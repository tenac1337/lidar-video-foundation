"""Per-instance native-depth gates and conservative static cross-view association.

Thresholds are engineering policies for upright tabletop chess pieces, not sensor
accuracy guarantees. Board-like returns are unknown object depth, never filled.
"""

import chess
import cv2
import numpy as np

from lidar_video.native_depth import sample_ids

MIN_NATIVE_CORE = 24
MIN_VALID_FRACTION = 0.65
MIN_FOREGROUND_FRACTION = 0.70
MIN_HEIGHT_M = 0.015
MIN_SQUARE_FRACTION = 0.90


def unique_points(points, ids, selected):
    """One representative per native sample, regardless of RGB upsampling factor."""
    selected = selected & (ids >= 0)
    rows = np.flatnonzero(selected)
    _, first = np.unique(ids.ravel()[rows], return_index=True)
    return points.reshape(-1, 3)[rows[first]]


def assess_depth(data, calibration, frame, mask, *, points=None, square=None):
    from lidar_video.reconstruction import calibrated_points

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != data["depth"].shape[1:] or not mask.any():
        raise ValueError("Instance mask must be nonempty and registered to depth")
    if points is None:
        points = calibrated_points(data, calibration, frame)
    ids = sample_ids(data, frame)
    h, w = mask.shape
    nh, nw = data.get("native_depth_shape", (h, w))
    ry, rx = max(1, round(h / nh)), max(1, round(w / nw))
    core = cv2.erode(
        mask.astype(np.uint8),
        np.ones((2 * ry + 1, 2 * rx + 1), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    z = data["depth"][frame]
    valid = np.isfinite(z) & (z > 0) & (data["confidence"][frame] >= 1) & (ids >= 0)
    core_points = unique_points(points, ids, core & valid)
    raised = (points[..., 2] > MIN_HEIGHT_M) & (points[..., 2] < 0.2)
    trusted = core & valid & raised
    cloud = unique_points(points, ids, trusted)
    valid_fraction = float(valid[mask].mean())
    core_valid_fraction = float(valid[core].mean()) if core.any() else 0.0
    foreground_fraction = (
        float(np.mean((core_points[:, 2] > MIN_HEIGHT_M) & (core_points[:, 2] < 0.2)))
        if len(core_points)
        else 0.0
    )
    reasons = []
    if len(core_points) < MIN_NATIVE_CORE:
        reasons.append("insufficient native core samples")
    if min(valid_fraction, core_valid_fraction) < MIN_VALID_FRACTION:
        reasons.append("incomplete registered depth")
    if foreground_fraction < MIN_FOREGROUND_FRACTION:
        reasons.append("board-like or implausible depth inside object")
    if len(cloud) < MIN_NATIVE_CORE:
        reasons.append("insufficient trusted foreground samples")
    square_fraction = None
    if square is not None:
        index = chess.parse_square(square)
        size, gap = calibration["square_size_m"], calibration.get("center_gap_m", 0)
        center = (np.array([chess.square_file(index), chess.square_rank(index)]) - 3.5) * size
        center[1] += np.sign(center[1]) * gap / 2
        square_fraction = (
            float(np.mean(np.all(np.abs(cloud[:, :2] - center) <= size / 2, axis=1))) if len(cloud) else 0.0
        )
        if square_fraction < MIN_SQUARE_FRACTION:
            reasons.append("foreground depth spans multiple squares")
    quality = dict(
        passed=not reasons,
        reasons=reasons,
        native_core_samples=len(core_points),
        trusted_native_samples=len(cloud),
        square=square,
        native_square_fraction=square_fraction,
        valid_fraction=valid_fraction,
        core_valid_fraction=core_valid_fraction,
        foreground_fraction=foreground_fraction,
        median_core_height_m=float(np.median(core_points[:, 2])) if len(core_points) else None,
        policy=dict(
            min_native_samples=MIN_NATIVE_CORE,
            min_valid_fraction=MIN_VALID_FRACTION,
            min_foreground_fraction=MIN_FOREGROUND_FRACTION,
            min_height_m=MIN_HEIGHT_M,
            min_square_fraction=MIN_SQUARE_FRACTION,
        ),
    )
    return quality, trusted, cloud


def reprojection_agreement(cloud, target_mask, data, calibration, frame):
    """Trusted source points must project inside the target instance, bidirectionally.

    Occluded samples are excluded using target depth. Missing or board-like target
    depth does not invent agreement. One-native-pixel dilation tolerates rasterization.
    """
    pose = np.asarray(calibration["board_from_world"]) @ data["world_from_camera"][frame]
    camera = (cloud - pose[:3, 3]) @ pose[:3, :3]
    k = data["K"] if data["K"].ndim == 2 else data["K"][frame]
    uv = camera[:, :2] / np.maximum(camera[:, 2:], 1e-8) * np.diag(k)[:2] + k[:2, 2]
    xy = np.rint(uv).astype(int)
    h, w = target_mask.shape
    inside = (camera[:, 2] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    xy, camera = xy[inside], camera[inside]
    nh, nw = data.get("native_depth_shape", (h, w))
    radius = max(1, round(max(h / nh, w / nw)))
    dilated = cv2.dilate(target_mask.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    z = data["depth"][frame, xy[:, 1], xy[:, 0]]
    confidence = data["confidence"][frame, xy[:, 1], xy[:, 0]]
    visible = np.isfinite(z) & (z > 0) & (confidence >= 1) & (z >= camera[:, 2] - 0.015)
    hit = dilated[xy[:, 1], xy[:, 0]][visible]
    fraction = float(np.mean(hit > 0)) if len(hit) else 0.0
    return dict(
        passed=len(hit) >= 12 and fraction >= 0.75, visible_native_points=len(hit), mask_fraction=fraction
    )


def distinct_view(data, a, b):
    if "source_frame" in data and data["source_frame"][a] == data["source_frame"][b]:
        return False
    left, right = data["world_from_camera"][[a, b]]
    translation = np.linalg.norm(left[:3, 3] - right[:3, 3])
    angle = np.arccos(np.clip((np.trace(left[:3, :3].T @ right[:3, :3]) - 1) / 2, -1, 1))
    return bool(translation >= 0.005 or angle >= np.deg2rad(1))


def select_views(candidates, data, calibration):
    """Select a reliable reference only from mutually corroborating static views.

    Candidate grouping by square is only a proposal; reprojection checks are
    mandatory. Contradictory reliable views reject the group instead of being hidden.
    """
    selected, rejected = {}, {}
    groups = {}
    for c in candidates:
        groups.setdefault(c["square"], []).append(c)
    for square, group in groups.items():
        if len(group) < 2:
            rejected[square] = "needs another reliable segmented view"
            continue
        checks = []
        for i, left in enumerate(group):
            for right in group[i + 1 :]:
                a = reprojection_agreement(left["cloud"], right["mask"], data, calibration, right["frame"])
                b = reprojection_agreement(right["cloud"], left["mask"], data, calibration, left["frame"])
                checks.append(dict(frames=[left["frame"], right["frame"]], forward=a, backward=b))
        if not all(c["forward"]["passed"] and c["backward"]["passed"] for c in checks):
            rejected[square] = dict(reason="cross-view instance association unverified", checks=checks)
            continue
        best = max(
            group,
            key=lambda c: (
                c["quality"]["trusted_native_samples"] * c["quality"]["valid_fraction"],
                -c["frame"],
            ),
        )
        if not any(distinct_view(data, best["frame"], c["frame"]) for c in group):
            rejected[square] = "needs a distinct camera viewpoint, not repeated frames"
            continue
        selected[square] = dict(reference=best, observations=group, association=checks)
    return selected, rejected
