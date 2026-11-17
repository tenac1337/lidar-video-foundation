"""Bounded silhouette-volume proposals and construction-only candidate selection.

Camera poses establish metric scale. A supplied support plane and height bound
are explicit hypotheses, not recovered object surfaces. This can add volume
missing from a learned mesh; it cannot recover invisible concavities or certify
appearance, instance identity, dynamics, or whole-scene completeness.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np
import trimesh
from scipy import ndimage

from lidar_video.camera_validation import rigid


@dataclass(frozen=True)
class VolumePolicy:
    pitch_m: float = 0.001
    max_height_m: float = 0.20
    mask_margin_pixels: int = 1
    max_cells: int = 12_000_000
    min_baseline_m: float = 0.01
    min_voxels: int = 16

    def __post_init__(self):
        if (
            not np.isfinite([self.pitch_m, self.max_height_m, self.min_baseline_m]).all()
            or not 0.0005 <= self.pitch_m <= 0.005
            or not 4 * self.pitch_m <= self.max_height_m <= 1
            or self.min_baseline_m <= 0
            or type(self.mask_margin_pixels) is not int
            or not 0 <= self.mask_margin_pixels <= 3
            or type(self.max_cells) is not int
            or not 100 <= self.max_cells <= 30_000_000
            or type(self.min_voxels) is not int
            or self.min_voxels < 1
        ):
            raise ValueError("Invalid bounded silhouette-volume policy")


def checked_views(views, *, allow_empty=False):
    """Validate masks, source identities and calibrated OpenCV cameras."""
    result = []
    for view in views:
        labels, k = np.asarray(view["labels"]), np.asarray(view["K"], float)
        label, source = view["label"], view["source_frame"]
        if (
            labels.ndim != 2
            or labels.dtype.kind not in "ui"
            or np.any(labels < 0)
            or type(label) is not int
            or label < 1
            or (not np.any(labels == label) and not (allow_empty and not np.any(labels)))
            or type(source) is not int
        ):
            raise ValueError("Nonempty indexed target masks and integer source identities required")
        if (
            k.shape != (3, 3)
            or not np.isfinite(k).all()
            or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1])
            or k[0, 1] != 0
            or k[1, 0] != 0
        ):
            raise ValueError("Invalid calibrated intrinsics")
        pose = rigid(view["world_from_camera"], "world_from_camera")
        result.append(dict(view, labels=labels, K=k, world_from_camera=pose))
    ids = [v["source_frame"] for v in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source viewpoints cannot supply independent evidence")
    return result


def distinct_centers(views, baseline):
    return any(
        np.linalg.norm(a["world_from_camera"][:3, 3] - b["world_from_camera"][:3, 3]) >= baseline
        for i, a in enumerate(views)
        for b in views[i + 1 :]
    )


def silhouette_score(mesh, view):
    """Rasterize at the original mask resolution; known other masks are unknown.

    Other-instance pixels are not positive target evidence. Their depth order is
    unresolved here, so they are excluded from the silhouette comparison only.
    """
    view = checked_views([view])[0]
    labels, k, pose = view["labels"], view["K"], view["world_from_camera"]
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("Finite triangle mesh required")
    camera = (mesh.vertices - pose[:3, 3]) @ pose[:3, :3]
    if np.any(camera[:, 2] <= 1e-6):
        return dict(iou=0.0, precision=0.0, recall=0.0, valid=False, reason="camera-plane crossing")
    uv = camera[:, :2] / camera[:, 2:] * np.diag(k)[:2] + k[:2, 2]
    if np.max(np.abs(uv)) > 1e7:
        return dict(iou=0.0, precision=0.0, recall=0.0, valid=False, reason="unbounded projection")
    pixels = np.rint(uv).astype(np.int32)
    raster = np.zeros(labels.shape, np.uint8)
    for face in mesh.faces:
        cv2.fillConvexPoly(raster, pixels[face], 1)
    target = labels == view["label"]
    visible = (labels == 0) | target
    predicted = (raster > 0) & visible
    intersection = int(np.count_nonzero(predicted & target))
    return dict(
        iou=intersection / max(1, np.count_nonzero(predicted | target)),
        precision=intersection / max(1, np.count_nonzero(predicted)),
        recall=intersection / max(1, np.count_nonzero(target)),
        valid=True,
        target_pixels=int(target.sum()),
        predicted_visible_pixels=int(predicted.sum()),
    )


def anchor_connected_volume(occupied, lower, pitch, landmarks, *, tolerance_m=0.003):
    """Identify one component using every supplied landmark, never component size.

    This explicitly assumes a single connected object. It is not a certificate
    that the landmarks, masks, support plane or physical prior are correct.
    """
    landmarks = np.asarray(landmarks, float)
    if (
        landmarks.ndim != 2
        or landmarks.shape[1] != 3
        or not 4 <= len(landmarks) <= 4096
        or not np.isfinite(landmarks).all()
        or len(np.unique(landmarks, axis=0)) < 4
        or not np.isfinite(tolerance_m)
        or not 0 < tolerance_m <= 0.003
    ):
        raise ValueError("Four or more distinct finite construction landmarks and <=3 mm tolerance required")
    if (
        np.asarray(occupied).ndim != 3
        or np.asarray(occupied).dtype != bool
        or not np.isfinite(pitch)
        or pitch <= 0
    ):
        raise ValueError("Boolean voxel volume and positive pitch required")
    labels, count = ndimage.label(occupied, structure=ndimage.generate_binary_structure(3, 1))
    radius = int(np.ceil(tolerance_m / pitch)) + 1
    offsets = np.array(np.meshgrid(*([np.arange(-radius, radius + 1)] * 3), indexing="ij")).reshape(3, -1).T
    shared = None
    for point in landmarks:
        center = np.floor((point - lower) / pitch).astype(int)
        candidates = center + offsets
        inside = np.all((candidates >= 0) & (candidates < np.array(occupied.shape)), axis=1)
        candidates = candidates[inside]
        centers = lower + (candidates + 0.5) * pitch
        candidates = candidates[np.linalg.norm(centers - point, axis=1) <= tolerance_m]
        ids = set(labels[tuple(candidates.T)].tolist()) - {0}
        if not ids:
            raise ValueError("A construction landmark has no nearby silhouette volume")
        shared = ids if shared is None else shared & ids
        if not shared:
            raise ValueError("Construction landmarks do not support one connected component")
    if len(shared) != 1:
        raise ValueError("Landmarks leave connected-component identity ambiguous")
    chosen = next(iter(shared))
    selected = labels == chosen
    return selected, dict(
        prior="one connected object containing every construction landmark",
        landmark_count=len(landmarks),
        tolerance_m=tolerance_m,
        input_components=int(count),
        selected_component=int(chosen),
        discarded_cells=int(occupied.sum() - selected.sum()),
        selection_by_size=False,
    )


def silhouette_volume(
    views, support_from_world, *, policy=None, static_scene=False, connected_landmarks=None
):
    """Construct a new visual hull from two or more views, without a learned seed.

    Each occupied cell must project into at least one target and must not project
    onto observed background in another construction view. Occluders are unknown.
    This proposal requires a fully framed object in every construction camera;
    masks at image edges abstain and the volume stays inside every camera.
    Disconnected components are retained by default. Explicit construction-only
    landmarks may identify one component under a connected-object hypothesis;
    no component is selected by size. Top/side truncation still abstains.
    """
    if static_scene is not True:
        raise ValueError("Explicit static_scene=True required")
    policy = policy or VolumePolicy()
    views = checked_views(views)
    if len(views) < 2 or not distinct_centers(views, policy.min_baseline_m):
        raise ValueError("Triangulation requires distinct camera centers")
    support = rigid(support_from_world, "support_from_world")
    boxes = []
    for view in views:
        yy, xx = np.where(view["labels"] == view["label"])
        if (
            yy.min() == 0
            or xx.min() == 0
            or yy.max() == view["labels"].shape[0] - 1
            or xx.max() == view["labels"].shape[1] - 1
        ):
            raise ValueError("Partially framed target cannot supply a bounded object silhouette")
        m = policy.mask_margin_pixels + 2
        corners = np.array(
            [[x, y, 1] for x in [xx.min() - m, xx.max() + m] for y in [yy.min() - m, yy.max() + m]]
        )
        pose = support @ view["world_from_camera"]
        rays = corners @ np.linalg.inv(view["K"]).T @ pose[:3, :3].T
        if np.any(rays[:, 2] >= -1e-6) or pose[2, 3] <= policy.max_height_m:
            raise ValueError("Construction camera does not see the bounded support slab from above")
        projected = np.concatenate(
            [
                pose[:3, 3] + rays * ((height - pose[2, 3]) / rays[:, 2])[:, None]
                for height in [0, policy.max_height_m]
            ]
        )
        boxes.append([projected.min(axis=0), projected.max(axis=0)])
    # The intersection of per-view target frustum bounds is only a workspace
    # bound; silhouette carving below determines occupancy within it.
    lower = np.max([b[0] for b in boxes], axis=0)
    upper = np.min([b[1] for b in boxes], axis=0)
    lower[2], upper[2] = 0, policy.max_height_m
    if np.any(upper <= lower):
        raise ValueError("Construction frusta have no common bounded volume")
    pitch = policy.pitch_m
    lower[:2] = np.floor(lower[:2] / pitch) * pitch - 2 * pitch
    upper[:2] = np.ceil(upper[:2] / pitch) * pitch + 2 * pitch
    shape = np.ceil((upper - lower) / pitch).astype(int)
    if np.prod(shape.astype(np.int64)) > policy.max_cells:
        raise ValueError("Visual hull exceeds the declared voxel budget")
    occupied = np.zeros(tuple(shape), bool)
    inverse = np.linalg.inv(support)
    allowed = []
    for view in views:
        target = view["labels"] == view["label"]
        margin = policy.mask_margin_pixels
        target = cv2.dilate(target.astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8)) > 0
        allowed.append((target, (view["labels"] != 0) & (view["labels"] != view["label"])))
    total = int(np.prod(shape))
    for start in range(0, total, 200_000):
        linear = np.arange(start, min(start + 200_000, total))
        ijk = np.array(np.unravel_index(linear, tuple(shape))).T
        points = lower + (ijk + 0.5) * pitch
        world = points @ inverse[:3, :3].T + inverse[:3, 3]
        keep, positive = np.ones(len(points), bool), np.zeros(len(points), np.uint16)
        for view, (target, other) in zip(views, allowed):
            pose, k = view["world_from_camera"], view["K"]
            camera = (world - pose[:3, 3]) @ pose[:3, :3]
            front = camera[:, 2] > 1e-6
            uv = np.zeros((len(points), 2), np.int64)
            uv[front] = np.rint(camera[front, :2] / camera[front, 2:] * np.diag(k)[:2] + k[:2, 2]).astype(
                np.int64
            )
            inside = (
                front
                & (uv[:, 0] >= 0)
                & (uv[:, 1] >= 0)
                & (uv[:, 0] < target.shape[1])
                & (uv[:, 1] < target.shape[0])
            )
            keep &= inside
            ids = np.flatnonzero(inside)
            hit = target[uv[ids, 1], uv[ids, 0]]
            keep[ids] &= hit | other[uv[ids, 1], uv[ids, 0]]
            positive[ids] += hit
        occupied.ravel()[linear] = keep & (positive > 0)
    count = int(occupied.sum())
    if count < policy.min_voxels:
        raise ValueError("Construction silhouettes leave insufficient occupied volume")
    component_selection = None
    if connected_landmarks is not None:
        landmarks = np.asarray(connected_landmarks, float)
        if landmarks.ndim != 2 or landmarks.shape[1] != 3:
            raise ValueError("World-frame construction landmarks must have shape Nx3")
        local_landmarks = landmarks @ support[:3, :3].T + support[:3, 3]
        occupied, component_selection = anchor_connected_volume(occupied, lower, pitch, local_landmarks)
        count = int(occupied.sum())
        if count < policy.min_voxels:
            raise ValueError("Anchored volume has insufficient occupied cells")
    if occupied[:, :, -1].any() or occupied[[0, -1], :, :].any() or occupied[:, [0, -1], :].any():
        raise ValueError("Shape reaches a declared bound; geometry is not resolved")
    _, components = ndimage.label(occupied, structure=ndimage.generate_binary_structure(3, 1))
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(occupied, pitch=pitch)
    mesh.apply_translation(lower + pitch / 2)
    mesh.apply_transform(inverse)
    if not mesh.is_watertight or not np.isfinite(mesh.vertices).all():
        raise ValueError("Visual hull failed finite closed-mesh validation")
    return mesh, dict(
        method="bounded_calibrated_silhouette_volume_v1",
        component_selection=component_selection,
        policy=asdict(policy),
        construction_sources=[v["source_frame"] for v in views],
        support_from_world=support.tolist(),
        occupied_cells=count,
        connected_components=int(components),
        grid_shape=shape.tolist(),
        touches_support=bool(occupied[:, :, 0].any()),
        shape_source="calibrated silhouettes; no learned mesh seed",
        physical_priors=[
            "static scene",
            "supplied support plane",
            "declared height bound",
            "target fully framed in each construction camera",
        ]
        + ([component_selection["prior"]] if component_selection else []),
        metric_object_accuracy_established=False,
        accepted=False,
    )


def select_candidate(
    candidates,
    incumbent,
    views,
    construction_sources,
    validation_sources,
    *,
    min_iou=0.45,
    max_regression=0.0,
):
    """Rank using construction only; validation can veto once, never rerank.

    Return a proposal without replacement permission if no reserved view exists.
    A silhouette pass is a geometry-screening result, not a qualitative win.
    """
    if not candidates or any(not isinstance(k, str) or not k for k in candidates):
        raise ValueError("Named candidate meshes required")
    if (
        not np.isfinite([min_iou, max_regression]).all()
        or not 0 <= min_iou <= 1
        or not 0 <= max_regression <= 0.05
    ):
        raise ValueError("Invalid candidate-screen thresholds")
    views = checked_views(views)
    ids = [v["source_frame"] for v in views]
    split = list(construction_sources) + list(validation_sources)
    if len(construction_sources) < 2 or len(split) != len(set(split)) or set(split) != set(ids):
        raise ValueError("Construction and validation must partition all unique observations")
    training = [v for v in views if v["source_frame"] in construction_sources]
    validation = [v for v in views if v["source_frame"] in validation_sources]
    if not distinct_centers(training, 0.01):
        raise ValueError("Construction requires a translational camera baseline")
    if any(
        np.linalg.norm(a["world_from_camera"][:3, 3] - b["world_from_camera"][:3, 3]) < 0.01
        for a in validation
        for b in training
    ):
        raise ValueError("Reserved camera centers must differ from construction centers")
    baseline = {v["source_frame"]: silhouette_score(incumbent, v) for v in views}
    report = dict(
        accepted=False,
        selected=None,
        construction={},
        validation={},
        incumbent=baseline,
        construction_sources=list(construction_sources),
        validation_sources=list(validation_sources),
        min_iou=min_iou,
        max_regression=max_regression,
        validation_used_for_ranking=False,
        independent_scene_evaluation=False,
        qualitative_win_established=False,
    )

    def passes(scores):
        return all(
            s["valid"] and s["iou"] >= max(min_iou, baseline[f]["iou"] - max_regression)
            for f, s in scores.items()
        )

    eligible = []
    for name, mesh in sorted(candidates.items()):
        scores = {v["source_frame"]: silhouette_score(mesh, v) for v in training}
        report["construction"][name] = scores
        if passes(scores):
            values = [s["iou"] for s in scores.values()]
            eligible.append((min(values), np.mean(values), name))
    if not eligible:
        report["reason"] = "No candidate clears every construction view without regression"
        return report
    chosen = max(eligible)[2]
    report["selected"] = chosen
    if not validation:
        report["reason"] = "Development proposal only: no reserved camera for replacement validation"
        return report
    scores = {v["source_frame"]: silhouette_score(candidates[chosen], v) for v in validation}
    report["validation"] = scores
    report["accepted"] = passes(scores)
    report["reason"] = (
        "Selected candidate passed the reserved-view veto"
        if report["accepted"]
        else "Reserved-view regression; rejected without trying another candidate"
    )
    return report


def refine_support_pose(
    incumbent, construction_views, support_from_world, *, max_translation_m=0.008, max_yaw_degrees=15.0
):
    """Bounded construction-only XY/yaw search preserving shape, scale and height."""
    return _refine_support(
        incumbent,
        construction_views,
        support_from_world,
        max_translation_m=max_translation_m,
        max_yaw_degrees=max_yaw_degrees,
    )


def refine_support_shape(
    incumbent,
    construction_views,
    support_from_world,
    *,
    max_translation_m=0.008,
    max_yaw_degrees=15.0,
    max_scale_fraction=0.10,
):
    """Bounded width/height refinement of a supplied prior; preserve its base height.

    This changes object geometry explicitly, not the camera calibration or global
    metric scale. Native collision assets must be rebuilt if a proposal is adopted.
    Require a separate-view veto and full-scene placement checks before adoption.
    """
    return _refine_support(
        incumbent,
        construction_views,
        support_from_world,
        max_translation_m=max_translation_m,
        max_yaw_degrees=max_yaw_degrees,
        max_scale_fraction=max_scale_fraction,
    )


def _refine_support(
    incumbent,
    construction_views,
    support_from_world,
    *,
    max_translation_m=0.008,
    max_yaw_degrees=15.0,
    max_scale_fraction=0.0,
):
    """Shared bounded search; optional width/height changes are explicit priors."""
    if (
        not np.isfinite([max_translation_m, max_yaw_degrees, max_scale_fraction]).all()
        or not 0 <= max_scale_fraction <= 0.15
        or not 0 < max_translation_m <= 0.02
        or not 0 < max_yaw_degrees <= 30
    ):
        raise ValueError("Invalid bounded support pose policy")
    views = checked_views(construction_views)
    if len(views) < 2 or not distinct_centers(views, 0.01):
        raise ValueError("Pose refinement requires two distinct construction cameras")
    support = rigid(support_from_world, "support_from_world")
    local = incumbent.copy()
    local.apply_transform(support)
    center = local.bounds.mean(axis=0)
    inverse = np.linalg.inv(support)
    limit = np.array([max_translation_m, max_translation_m, np.deg2rad(max_yaw_degrees)])
    if max_scale_fraction:
        limit = np.r_[limit, max_scale_fraction, max_scale_fraction]
    baseline = np.array([silhouette_score(incumbent, v)["iou"] for v in views])
    evaluations = 0

    def make(parameters):
        x, y, yaw = parameters[:3]
        rotation = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1], point=center)
        rotation[:2, 3] += [x, y]
        shape = np.eye(4)
        if max_scale_fraction:
            width, height = 1 + parameters[3:]
            shape[:3, :3] = np.diag([width, width, height])
            pivot = center.copy()
            pivot[2] = local.bounds[0, 2]
            shape[:3, 3] = pivot - shape[:3, :3] @ pivot
        mesh = local.copy()
        mesh.apply_transform(inverse @ rotation @ shape)
        return mesh, inverse @ rotation @ shape @ support

    def score(parameters):
        nonlocal evaluations
        mesh, _ = make(parameters)
        scores = [silhouette_score(mesh, v) for v in views]
        evaluations += 1
        if not all(s["valid"] for s in scores):
            return (-np.inf, -np.inf)
        values = np.array([s["iou"] for s in scores])
        return (float(np.min(values - baseline)), float(np.mean(values)))

    parameters = np.zeros(len(limit))
    best = score(parameters)
    # A fixed coarse-to-fine coordinate search; no reserved observations enter.
    for fraction in [0.5, 0.25, 0.125, 0.0625]:
        for _ in range(4):
            options = [(best, parameters)]
            for axis in range(len(limit)):
                for sign in [-1, 1]:
                    p = parameters.copy()
                    p[axis] += sign * fraction * limit[axis]
                    if np.all(np.abs(p) <= limit + 1e-12):
                        options.append((score(p), p))
            winner = max(range(len(options)), key=lambda i: options[i][0])
            if winner == 0:
                break
            best, parameters = options[winner]
    mesh, transform = make(parameters)
    return mesh, dict(
        method="bounded_support_shape_v1" if max_scale_fraction else "bounded_support_xy_yaw_v1",
        construction_sources=[v["source_frame"] for v in views],
        max_translation_m=max_translation_m,
        max_yaw_degrees=max_yaw_degrees,
        translation_support_xy_m=parameters[:2].tolist(),
        yaw_degrees=float(np.rad2deg(parameters[2])),
        world_from_original_world=transform.tolist(),
        evaluations=evaluations,
        max_scale_fraction=max_scale_fraction,
        width_scale=float(1 + parameters[3]) if max_scale_fraction else 1.0,
        height_scale=float(1 + parameters[4]) if max_scale_fraction else 1.0,
        support_height_preserved=True,
        shape_and_scale_preserved=not max_scale_fraction or bool(np.all(parameters[3:] == 0)),
        accepted=False,
    )
