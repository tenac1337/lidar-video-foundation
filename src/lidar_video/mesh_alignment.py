"""Scale-preserving support fit with per-instance, per-view acceptance.

Only the chosen reference trains the fit. Every other segmented view is held out.
No point pool, unlabelled square cloud, or free scale can satisfy this contract.
"""

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from lidar_video.object_evidence import assess_depth, distinct_view, reprojection_agreement


def silhouette_iou(mesh, labels, label, data, calibration, frame):
    h, w = labels.shape
    scale = min(1.0, 320 / w)
    width, height = round(w * scale), round(h * scale)
    pose = np.asarray(calibration["board_from_world"]) @ data["world_from_camera"][frame]
    vertices = (mesh.vertices - pose[:3, 3]) @ pose[:3, :3]
    k = data["K"] if data["K"].ndim == 2 else data["K"][frame]
    uv = (vertices[:, :2] / np.maximum(vertices[:, 2:], 1e-8) * np.diag(k)[:2] + k[:2, 2]) * scale
    if not np.isfinite(uv).all() or np.max(np.abs(uv)) > 1e7:
        return 0.0
    raster = np.zeros((height, width), np.uint8)
    for face in mesh.faces:
        if np.all(vertices[face, 2] > 0):
            cv2.fillConvexPoly(raster, np.rint(uv[face]).astype(np.int32), 1)
    target = cv2.resize(
        (labels == label).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    other = cv2.resize(
        ((labels != 0) & (labels != label)).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    visible = ~other
    union = ((raster > 0) | target) & visible
    return float(np.sum((raster > 0) & target & visible) / max(1, union.sum()))


def validate_and_align(mesh, entry, root, data, calibration):
    from lidar_video.reconstruction import MeshQualityError

    observations = entry.get("observations", [])
    frames = [o["frame"] for o in observations]
    reference = entry["frame"]
    if not entry.get("static_scene"):
        raise ValueError("Cross-view mesh validation requires an explicitly static scene")
    if len(frames) < 2 or len(set(frames)) != len(frames) or reference not in frames:
        raise MeshQualityError("Need distinct instance-masked reference and held-out views")
    if any(type(frame) is not int or not 0 <= frame < len(data["depth"]) for frame in frames):
        raise ValueError("Invalid observation frame")
    if not any(distinct_view(data, reference, frame) for frame in frames):
        raise MeshQualityError("Need a distinct held-out camera viewpoint")
    evidence = []
    for o in observations:
        frame, label = o["frame"], o["label"]
        if (
            type(frame) is not int
            or not 0 <= frame < len(data["depth"])
            or type(label) is not int
            or label < 1
        ):
            raise ValueError("Invalid observation frame or instance label")
        labels = np.asarray(Image.open(root / o["mask"]))
        if labels.ndim != 2 or labels.dtype.kind not in "iu":
            raise ValueError("Observation masks must be indexed integer images")
        quality, _, cloud = assess_depth(
            data, calibration, frame, labels == label, square=entry.get("square")
        )
        if not quality["passed"]:
            raise MeshQualityError(f"Unreliable object depth in frame {frame}: {quality['reasons']}")
        evidence.append(dict(frame=frame, label=label, labels=labels, cloud=cloud, quality=quality))
    association = []
    for i, left in enumerate(evidence):
        for right in evidence[i + 1 :]:
            forward = reprojection_agreement(
                left["cloud"], right["labels"] == right["label"], data, calibration, right["frame"]
            )
            backward = reprojection_agreement(
                right["cloud"], left["labels"] == left["label"], data, calibration, left["frame"]
            )
            if not forward["passed"] or not backward["passed"]:
                raise MeshQualityError("Held-out instance association fails mutual reprojection")
            association.append(
                dict(frames=[left["frame"], right["frame"]], forward=forward, backward=backward)
            )
    sampled, _ = trimesh.sample.sample_surface(mesh, 60000, seed=0)
    center = np.mean(mesh.vertices, axis=0)
    hull = mesh.convex_hull.vertices
    tree = cKDTree(sampled)
    visual = mesh.simplify_quadric_decimation(face_count=3000) if len(mesh.faces) > 3000 else mesh.copy()
    tolerance = float(calibration["square_size_m"]) * 0.25

    def transform(parameters):
        rotation = Rotation.from_rotvec(parameters[2:]).as_matrix()
        bottom = np.min(((hull - center) @ rotation.T + center)[:, 2])
        translation = np.array([parameters[0], parameters[1], -bottom])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = center + translation - rotation @ center
        return matrix

    def evaluate(matrix):
        moved = visual.copy()
        moved.apply_transform(matrix)
        views = []
        for o in evidence:
            inverse = (o["cloud"] - matrix[:3, 3]) @ matrix[:3, :3]
            distances = tree.query(inverse)[0]
            iou = silhouette_iou(moved, o["labels"], o["label"], data, calibration, o["frame"])
            median, p90 = map(float, np.quantile(distances, [0.5, 0.9]))
            views.append(
                dict(
                    frame=o["frame"],
                    role="training" if o["frame"] == reference else "held_out",
                    native_points=len(distances),
                    median_m=median,
                    p90_m=p90,
                    silhouette_iou=iou,
                    passed=median <= tolerance and p90 <= 2 * tolerance and iou >= 0.5,
                )
            )
        return views

    identity = np.eye(4)
    original_views = evaluate(identity)
    baseline = transform(np.zeros(5))
    baseline_views = evaluate(baseline)

    def acceptable(views):
        return all(
            v["passed"]
            and v["silhouette_iou"] >= old["silhouette_iou"] - 0.05
            and (
                v["role"] == "training"
                or (v["median_m"] <= old["median_m"] + 0.001 and v["p90_m"] <= old["p90_m"] + 0.002)
            )
            for v, old in zip(views, original_views)
        )

    attempts = []

    def record(method, matrix, views, success=True):
        accepted = bool(success and acceptable(views))
        attempts.append(
            dict(
                method=method,
                matrix=matrix.tolist(),
                views=views,
                optimizer_success=bool(success),
                accepted=accepted,
            )
        )
        return accepted

    method, chosen, views = "support_snap", baseline, baseline_views
    # A large initial base offset alone must not trigger a destructive rotation
    # when the simple support correction already passes every unchanged gate.
    if not record(method, chosen, views):
        training = next(o["cloud"] for o in evidence if o["frame"] == reference)
        training = training[np.linspace(0, len(training) - 1, min(1200, len(training)), dtype=int)]

        def depth_residual(parameters):
            matrix = transform(parameters)
            return tree.query((training - matrix[:3, 3]) @ matrix[:3, :3])[0]

        xy_bound = calibration["square_size_m"] / 2
        translation_fit = least_squares(
            lambda xy: depth_residual(np.r_[xy, np.zeros(3)]),
            np.zeros(2),
            bounds=(-np.full(2, xy_bound), np.full(2, xy_bound)),
            loss="soft_l1",
            f_scale=0.004,
            max_nfev=100,
        )
        parameters = np.r_[translation_fit.x, np.zeros(3)]
        method = "reference_only_translation_with_held_out_validation"
        chosen = transform(parameters)
        views = evaluate(chosen)
        if not record(method, chosen, views, translation_fit.success):
            # The original learned orientation is a prior, not a measured truth.
            # Express angular change as RMS surface displacement so a noisy small
            # patch cannot freely tilt the entire piece to fit the training view.
            radius = float(np.sqrt(np.mean(np.sum((hull - center) ** 2, axis=1))))

            def residual(parameters):
                rotation_prior = np.tile(radius * parameters[2:] / np.sqrt(3), len(training))
                return np.r_[depth_residual(parameters), rotation_prior]

            bounds = np.array([xy_bound] * 2 + [np.deg2rad(30)] * 3)
            fit = least_squares(
                residual, parameters, bounds=(-bounds, bounds), loss="soft_l1", f_scale=0.004, max_nfev=100
            )
            method = "reference_only_regularized_rigid_fit_with_held_out_validation"
            chosen = transform(fit.x)
            views = evaluate(chosen)
            if not record(method, chosen, views, fit.success):
                raise MeshQualityError(
                    f"Support-constrained fit failed per-view depth/silhouette validation: {views}",
                    evidence=dict(
                        original_views=original_views,
                        attempts=attempts,
                        reference_frame=reference,
                        scale_reapplied=False,
                    ),
                )
    mesh.apply_transform(chosen)
    return mesh, dict(
        validation="instance-multiview-v1",
        method=method,
        scale_reapplied=False,
        board_refinement=chosen.tolist(),
        original_views=original_views,
        attempts=attempts,
        views=views,
        association=association,
        rigid_translation_z_m=float(chosen[2, 3]),
        original_base_height_m=float(np.min(hull[:, 2])),
        final_base_height_m=float(mesh.bounds[0, 2]),
        reference_frame=reference,
        depth_residual_m=max(v["median_m"] for v in views),
    )
