"""Select measured model-conditioning samples with per-ray multiview evidence.

This does not certify object surfaces: a background plane can agree in every
view. Original observations remain the independent mesh-validation input, so
filtering cannot conceal a candidate's disagreement with measured depth.
"""

import cv2
import numpy as np

from lidar_video.native_depth import native_shape, sample_ids, sensor_camera_points
from lidar_video.object_evidence import distinct_view


def corroborated_conditioning(data, observation, witnesses, *, depth_tolerance_m=0.015):
    """Keep native samples agreeing with another distinct, visible measurement.

    Visible silhouette or depth contradictions veto a sample. Occlusion, missing
    depth and off-image projections supply no positive evidence. A one-native-
    pixel silhouette margin accommodates rasterization. No depth is synthesized,
    and no object class, support-height threshold or reference pose is fitted.
    """
    if not np.isfinite(depth_tolerance_m) or depth_tolerance_m <= 0:
        raise ValueError("Depth tolerance must be finite and positive")
    frame = observation["frame"]
    trusted = np.asarray(observation["trusted"])
    mask = np.asarray(observation["mask"])
    shape = data["depth"].shape[1:]
    if trusted.dtype != bool or mask.dtype != bool or trusted.shape != shape or mask.shape != shape:
        raise ValueError("Registered boolean observation masks required")
    if np.any(trusted & ~mask):
        raise ValueError("Trusted samples must be inside the source instance")
    ids = sample_ids(data, frame)
    camera = sensor_camera_points(data, frame)
    usable = trusted & (ids >= 0) & np.isfinite(camera).all(axis=-1)
    usable &= (camera[..., 2] > 0) & (data["confidence"][frame] >= 1)
    rows = np.flatnonzero(usable)
    native_ids, first = np.unique(ids.ravel()[rows], return_index=True)
    points = camera.reshape(-1, 3)[rows[first]]
    pose = data["world_from_camera"][frame]
    world = points @ pose[:3, :3].T + pose[:3, 3]
    supported = np.zeros(len(points), bool)
    contradicted = np.zeros(len(points), bool)
    h, w = shape
    nh, nw = native_shape(data)
    ry, rx = max(1, round(h / nh)), max(1, round(w / nw))
    checks, seen = [], set()
    for witness in witnesses:
        target = witness["frame"]
        if target == frame:
            continue
        if target in seen:
            raise ValueError("At most one witness per source view required")
        seen.add(target)
        if not distinct_view(data, frame, target):
            continue
        target_mask = np.asarray(witness["mask"])
        if target_mask.dtype != bool or target_mask.shape != shape:
            raise ValueError("Registered boolean witness mask required")
        pose = data["world_from_camera"][target]
        camera = (world - pose[:3, 3]) @ pose[:3, :3]
        k = data["K"] if data["K"].ndim == 2 else data["K"][target]
        projected = camera @ k.T
        uv = projected[:, :2] / np.maximum(projected[:, 2:], 1e-8)
        xy = np.rint(uv).astype(np.int64)
        inside = (camera[:, 2] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w)
        inside &= (xy[:, 1] >= 0) & (xy[:, 1] < h)
        indices = np.flatnonzero(inside)
        x, y = xy[inside].T
        measured = data["depth"][target, y, x]
        valid = np.isfinite(measured) & (measured > 0) & (data["confidence"][target, y, x] >= 1)
        valid &= sample_ids(data, target)[y, x] >= 0
        delta = measured - camera[inside, 2]
        visible = valid & (delta >= -depth_tolerance_m)
        dilated = cv2.dilate(target_mask.astype(np.uint8), np.ones((2 * ry + 1, 2 * rx + 1), np.uint8))
        agrees = visible & (abs(delta) <= depth_tolerance_m) & (dilated[y, x] > 0)
        conflicts = visible & ~agrees
        supported[indices[agrees]] = True
        contradicted[indices[conflicts]] = True
        checks.append(
            dict(
                frame=target,
                source_frame=int(data["source_frame"][target]),
                agreeing_native_samples=int(agrees.sum()),
                conflicting_native_samples=int(conflicts.sum()),
                unknown_native_samples=int(len(points) - visible.sum()),
            )
        )
    keep = supported & ~contradicted
    result = usable & np.isin(ids, native_ids[keep])
    report = dict(
        frame=frame,
        source_frame=int(data["source_frame"][frame]),
        input_native_samples=len(points),
        retained_native_samples=int(keep.sum()),
        contradicted_native_samples=int(contradicted.sum()),
        uncorroborated_native_samples=int((~supported & ~contradicted).sum()),
        depth_tolerance_m=depth_tolerance_m,
        witnesses=checks,
        object_surface_confirmed=False,
        synthesized_depth=False,
        original_validation_observations_preserved=True,
    )
    return result, report


def choose_conditioning_reference(data, item, *, min_native_points=24):
    """Prefer the most corroborated input; abstain instead of falling back to raw depth."""
    if type(min_native_points) is not int or min_native_points < 3:
        raise ValueError("At least three native samples required")
    candidates, reports = [], []
    for obs in item["observations"]:
        trusted, report = corroborated_conditioning(data, obs, item["observations"])
        reports.append(report)
        if report["retained_native_samples"] >= min_native_points:
            candidates.append((obs, trusted, report))
    audit = dict(min_native_points=min_native_points, observations=reports, passed=bool(candidates))
    if not candidates:
        return None, audit
    obs, trusted, report = max(
        candidates, key=lambda x: (x[2]["retained_native_samples"], -x[0]["source_frame"])
    )
    audit["selected_source_frame"] = obs["source_frame"]
    # Copy only the conditioning mask. In particular, never replace obs['cloud'].
    return dict(obs, trusted=trusted), audit
