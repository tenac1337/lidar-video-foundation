"""Bounded full-mesh prior refinement from RGB landmarks and silhouettes.

Sparse landmarks do not define a closed surface. This stage preserves a supplied
mesh's topology and unseen geometry, explicitly treating them as a prior. Only
construction views enter the search; additional views and native assets require
separate checks before replacement.
"""

import numpy as np
import trimesh

from lidar_video.camera_validation import rigid
from lidar_video.multiview_shape import checked_views, distinct_centers, silhouette_score


def refine_landmark_mesh(
    incumbent,
    world_points,
    construction_views,
    support_from_world,
    *,
    static_scene=False,
    max_translation_m=0.008,
    max_yaw_degrees=15.0,
    max_scale_fraction=0.15,
    landmark_cell_m=0.003,
    huber_delta_m=0.003,
    regularization_m2=1e-7,
):
    """Fit support-horizontal pose and width/height with silhouette nonregression.

    The minimum base height is fixed. Width scales both horizontal axes equally.
    One deterministic representative per landmark voxel limits clustered-feature
    dominance; this does not turn correlated tracks into independent evidence.
    The returned affine map must also be applied to collision geometry if adopted.
    """
    values = [
        max_translation_m,
        max_yaw_degrees,
        max_scale_fraction,
        landmark_cell_m,
        huber_delta_m,
        regularization_m2,
    ]
    if (
        static_scene is not True
        or not np.isfinite(values).all()
        or not 0 < max_translation_m <= 0.02
        or not 0 < max_yaw_degrees <= 30
        or not 0 < max_scale_fraction <= 0.15
        or not 0 < landmark_cell_m <= 0.005
        or not 0 < huber_delta_m <= 0.01
        or not 0 <= regularization_m2 <= 1e-5
    ):
        raise ValueError("Explicit static scene and bounded finite refinement policy required")
    views = checked_views(construction_views)
    if len(views) < 2 or not distinct_centers(views, 0.01):
        raise ValueError("Two distinct construction cameras required")
    support = rigid(support_from_world, "support_from_world")
    inverse = np.linalg.inv(support)
    points = np.asarray(world_points, float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not 4 <= len(points) <= 5000
        or not np.isfinite(points).all()
    ):
        raise ValueError("Four or more finite RGB landmarks required")
    points = points @ support[:3, :3].T + support[:3, 3]
    # Sort first so changing input order cannot change a voxel representative.
    points = points[np.lexsort(points.T[::-1])]
    _, first = np.unique(np.floor(points / landmark_cell_m), axis=0, return_index=True)
    selected = points[np.sort(first)]
    if len(selected) < 4 or np.linalg.svd(selected - selected.mean(axis=0), compute_uv=False)[1] < 0.002:
        raise ValueError("Insufficient spatially distributed RGB landmarks")
    local = incumbent.copy()
    local.apply_transform(support)
    if not len(local.faces) or not np.isfinite(local.vertices).all() or np.any(local.extents <= 0):
        raise ValueError("Finite nonzero full-mesh prior required")
    pivot = local.bounds.mean(axis=0)
    pivot[2] = local.bounds[0, 2]
    limits = np.array(
        [
            max_translation_m,
            max_translation_m,
            np.deg2rad(max_yaw_degrees),
            max_scale_fraction,
            max_scale_fraction,
        ]
    )
    baseline = np.array([silhouette_score(incumbent, v)["iou"] for v in views])
    evaluations = 0

    def make(parameters):
        x, y, yaw, width, height = parameters
        affine = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
        affine[:3, :3] = affine[:3, :3] @ np.diag([1 + width, 1 + width, 1 + height])
        affine[:3, 3] = pivot - affine[:3, :3] @ pivot + [x, y, 0]
        candidate = local.copy()
        candidate.apply_transform(affine)
        return candidate, inverse @ affine @ support

    def evaluate(parameters):
        nonlocal evaluations
        evaluations += 1
        mesh, transform = make(parameters)
        _, distances, _ = trimesh.proximity.closest_point(mesh, selected)
        if not np.isfinite(distances).all():
            raise ValueError("Nonfinite mesh-to-landmark distances")
        loss = np.mean(
            np.where(
                distances <= huber_delta_m,
                0.5 * distances**2,
                huber_delta_m * (distances - 0.5 * huber_delta_m),
            )
        )
        objective = loss + regularization_m2 * np.sum((parameters / limits) ** 2)
        mesh.apply_transform(inverse)
        scores = [silhouette_score(mesh, v) for v in views]
        allowed = all(s["valid"] for s in scores) and np.all(np.array([s["iou"] for s in scores]) >= baseline)
        return (
            (float(objective) if allowed else float("inf")),
            dict(
                landmark_huber_m2=float(loss),
                landmark_rmse_m=float(np.sqrt(np.mean(distances**2))),
                landmark_max_distance_m=float(distances.max()),
                construction_iou=[s["iou"] for s in scores],
            ),
            transform,
        )

    parameters = np.zeros(5)
    best, original, _ = evaluate(parameters)
    if not np.isfinite(best):
        raise ValueError("Incumbent cannot supply valid construction silhouettes")
    record = original
    for fraction in [0.5, 0.25, 0.125, 0.0625]:
        for _ in range(4):
            options = [(best, parameters, record)]
            for axis in range(5):
                for sign in [-1, 1]:
                    p = parameters.copy()
                    p[axis] += sign * fraction * limits[axis]
                    if np.any(np.abs(p) > limits + 1e-12):
                        continue
                    score, evidence, _ = evaluate(p)
                    options.append((score, p, evidence))
            winner = min(range(len(options)), key=lambda i: options[i][0])
            if winner == 0:
                break
            best, parameters, record = options[winner]
    mesh, transform = make(parameters)
    mesh.apply_transform(inverse)
    return mesh, dict(
        method="bounded_rgb_landmark_mesh_prior_v1",
        shape_source="supplied full mesh prior; RGB constraints do not recover unseen surfaces",
        input_landmarks=len(points),
        occupied_landmark_cells=len(selected),
        landmark_cell_m=landmark_cell_m,
        construction_sources=[v["source_frame"] for v in views],
        incumbent=original,
        candidate=record,
        translation_support_xy_m=parameters[:2].tolist(),
        yaw_degrees=float(np.rad2deg(parameters[2])),
        width_scale=float(1 + parameters[3]),
        height_scale=float(1 + parameters[4]),
        world_from_original_world=transform.tolist(),
        evaluations=evaluations,
        policy=dict(
            max_translation_m=max_translation_m,
            max_yaw_degrees=max_yaw_degrees,
            max_scale_fraction=max_scale_fraction,
            huber_delta_m=huber_delta_m,
            regularization_m2=regularization_m2,
        ),
        topology_preserved=True,
        support_height_preserved=True,
        accepted=False,
        scope="Construction-only prior refinement; additional-view, placement, visual and native checks required",
    )
