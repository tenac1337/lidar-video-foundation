"""Deterministic stereo pair selection from calibration, never image/depth fit.

Overlap is a rectification footprint proxy, not observed scene overlap. Occlusion,
moving objects, pose error and texture still require reconstruction-time checks.
"""

import cv2
import numpy as np

from lidar_video.camera_validation import rigid


def pair_geometry(k1, k2, pose1, pose2, size):
    """Measure rectification coverage in original images, including vertical pairs."""
    w, h = size
    relative = np.linalg.inv(pose2) @ pose1
    result = cv2.stereoRectify(
        k1, None, k2, None, (w, h), relative[:3, :3], relative[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    if abs(result[3][1, 3]) > abs(result[3][0, 3]):
        c = np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        converted = []
        for k in [k1, k2]:
            converted.append(np.array([[k[1, 1], 0, h - 1 - k[1, 2]], [0, k[0, 0], k[0, 2]], [0, 0, 1]]))
        k1, k2 = converted
        rotation = c.T @ relative[:3, :3] @ c
        translation = c.T @ relative[:3, 3]
        w, h = h, w
        result = cv2.stereoRectify(
            k1, None, k2, None, (w, h), rotation, translation, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
    r1, r2, p1, p2, _, roi1, roi2 = result
    fractions = []
    xx, yy = np.meshgrid(np.linspace(0, w - 1, 25), np.linspace(0, h - 1, 19))
    pixels = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    for k, r, p in [(k1, r1, p1), (k2, r2, p2)]:
        rays = pixels @ np.linalg.inv(p[:3, :3]).T @ r
        uv = rays @ k.T
        valid = uv[:, 2] > 0
        uv = uv[:, :2] / np.maximum(uv[:, 2:], 1e-12)
        inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if inside.sum() < 4:
            fractions.append(0.0)
        else:
            bounds = np.ptp(uv[inside], axis=0)
            fractions.append(float(np.prod(bounds) / (w * h) * inside.mean()))
    return dict(
        original_footprint_fraction=fractions,
        minimum_rectified_roi_fraction=min(x[2] * x[3] / (w * h) for x in [roi1, roi2]),
        rectified_focal_ratio=float(p1[0, 0] / k1[0, 0]),
    )


def select_stereo_neighbors(
    poses,
    intrinsics,
    size,
    anchor,
    candidates,
    *,
    excluded=(),
    min_baseline_m=0.02,
    max_baseline_m=0.20,
    target_baseline_m=0.06,
    max_rotation_degrees=25.0,
    min_footprint_fraction=0.20,
    min_neighbor_separation_m=0.02,
):
    """Select at most two separated partners, preserving all candidate failures.

    The API has no RGB or depth input. Candidates and excluded review-frame IDs
    are supplied explicitly; selection cannot retry on downstream stereo output.
    """
    poses = np.asarray(poses, float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("Rigid camera pose array required")
    n = len(poses)
    if type(anchor) is not int or not 0 <= anchor < n:
        raise ValueError("Valid anchor index required")
    if len(size) != 2 or any(type(x) is not int or x < 16 for x in size):
        raise ValueError("Positive image dimensions required")
    k = np.asarray(intrinsics, float)
    if k.shape not in [(3, 3), (n, 3, 3)] or not np.isfinite(k).all():
        raise ValueError("Finite camera intrinsics required")
    ks = np.broadcast_to(k, (n, 3, 3))
    if (
        np.any(ks[:, (0, 1), (0, 1)] <= 0)
        or not np.allclose(ks[:, 2], [0, 0, 1])
        or not np.allclose(ks[:, 0, 1], 0)
        or not np.allclose(ks[:, 1, 0], 0)
    ):
        raise ValueError("Unskewed positive pinhole intrinsics required")
    policy = dict(
        min_baseline_m=min_baseline_m,
        max_baseline_m=max_baseline_m,
        target_baseline_m=target_baseline_m,
        max_rotation_degrees=max_rotation_degrees,
        min_footprint_fraction=min_footprint_fraction,
        min_neighbor_separation_m=min_neighbor_separation_m,
    )
    if (
        not np.isfinite(list(policy.values())).all()
        or not 0 < min_baseline_m <= target_baseline_m <= max_baseline_m
        or not 0 < max_rotation_degrees < 90
        or not 0 < min_footprint_fraction <= 1
        or not 0 < min_neighbor_separation_m <= 2 * max_baseline_m
    ):
        raise ValueError("Finite bounded selection policy required")
    candidates, excluded = set(candidates), set(excluded)
    if any(type(x) is not int or not 0 <= x < n for x in candidates | excluded):
        raise ValueError("Candidate/excluded indices must exist")
    if anchor in excluded:
        raise ValueError("Anchor is excluded from construction")
    for i in candidates | {anchor}:
        rigid(poses[i], "world_from_camera")
    records = []
    for i in sorted(candidates):
        row = dict(frame=i, eligible=False)
        records.append(row)
        if i == anchor or i in excluded:
            row["reason"] = "anchor or excluded review view"
            continue
        baseline = float(np.linalg.norm(poses[i, :3, 3] - poses[anchor, :3, 3]))
        rotation = poses[i, :3, :3].T @ poses[anchor, :3, :3]
        angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
        row.update(baseline_m=baseline, rotation_degrees=angle)
        if not min_baseline_m <= baseline <= max_baseline_m or angle > max_rotation_degrees:
            row["reason"] = "baseline or rotation outside policy"
            continue
        try:
            geometry = pair_geometry(ks[anchor], ks[i], poses[anchor], poses[i], size)
        except (cv2.error, np.linalg.LinAlgError):
            row["reason"] = "rectification failed"
            continue
        row.update(geometry)
        footprint = min(geometry["original_footprint_fraction"])
        if (
            not np.isfinite(
                list(geometry["original_footprint_fraction"]) + [geometry["rectified_focal_ratio"]]
            ).all()
            or footprint < min_footprint_fraction
            or geometry["minimum_rectified_roi_fraction"] < 0.02
            or not 0.5 <= geometry["rectified_focal_ratio"] <= 4
        ):
            row["reason"] = "insufficient rectifiable footprint"
            continue
        row.update(
            eligible=True,
            score=float(footprint * min(baseline / target_baseline_m, target_baseline_m / baseline)),
        )
    selected = []
    for row in sorted((x for x in records if x["eligible"]), key=lambda x: (-x["score"], x["frame"])):
        if all(
            np.linalg.norm(poses[row["frame"], :3, 3] - poses[i, :3, 3]) >= min_neighbor_separation_m
            for i in selected
        ):
            selected.append(row["frame"])
        if len(selected) == 2:
            break
    return dict(
        anchor=anchor,
        neighbors=selected,
        complete=len(selected) == 2,
        policy=policy,
        candidates=records,
        selection_basis="Calibration-only footprint/baseline proxy; no RGB/depth quality ranking",
        independent_evaluation=False,
    )
