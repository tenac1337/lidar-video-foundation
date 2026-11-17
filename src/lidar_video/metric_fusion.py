"""Sensor-anchored scale and view-consistency gates for learned depth proposals.

These operations do not establish surface ownership or reference accuracy. Scale
calibration changes the modality: its result is RGB+LiDAR, never RGB-only.
"""

import cv2
import numpy as np

from lidar_video.camera_validation import rigid
from lidar_video.capture import camera_points


def _camera(k):
    k = np.asarray(k, float)
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or np.any(np.diag(k)[:2] <= 0)
        or not np.allclose(k[2], [0, 0, 1])
        or not np.allclose([k[0, 1], k[1, 0]], 0)
    ):
        raise ValueError("Finite unskewed pinhole intrinsics required")
    return k


def _depth(z):
    z = np.asarray(z, float)
    if z.ndim != 2 or not z.size or np.any(np.isfinite(z) & (z < 0)):
        raise ValueError("Nonnegative nonempty depth grid required")
    return z


def sample_registered_depth(depth, source_k, target_k, target_shape):
    """Sample co-located camera rays; out-of-image rays remain unknown, never clipped."""
    depth, source_k, target_k = _depth(depth), _camera(source_k), _camera(target_k)
    h, w = target_shape
    if h < 1 or w < 1:
        raise ValueError("Positive target dimensions required")
    yy, xx = np.indices((h, w))
    x = np.rint((xx - target_k[0, 2]) / target_k[0, 0] * source_k[0, 0] + source_k[0, 2]).astype(int)
    y = np.rint((yy - target_k[1, 2]) / target_k[1, 1] * source_k[1, 1] + source_k[1, 2]).astype(int)
    inside = (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
    out = np.zeros((h, w), float)
    values = depth[y[inside], x[inside]]
    out[inside] = np.where(np.isfinite(values) & (values > 0), values, 0)
    return out


def smooth_support(depth, valid, *, absolute_m=0.015, relative=0.0, radius=1):
    """Exclude missing neighborhoods and depth jumps; smoothness is not object ownership."""
    depth = _depth(depth)
    valid = np.asarray(valid, bool)
    if valid.shape != depth.shape or type(radius) is not int or not 1 <= radius <= 5:
        raise ValueError("Matching validity grid and radius 1..5 required")
    if not np.isfinite([absolute_m, relative]).all() or min(absolute_m, relative) < 0:
        raise ValueError("Finite nonnegative smoothness tolerances required")
    valid = valid & np.isfinite(depth) & (depth > 0)
    kernel = np.ones((2 * radius + 1,) * 2, np.uint8)
    complete = cv2.erode(valid.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    low = cv2.erode(np.where(valid, depth, 0), kernel)
    high = cv2.dilate(np.where(valid, depth, 0), kernel)
    return complete & (high - low <= absolute_m + relative * depth)


def fit_metric_scale(
    predicted,
    measured,
    support,
    *,
    min_samples=128,
    min_tiles=6,
    scale_bounds=(0.25, 4.0),
    tolerance_m=0.015,
    max_median_m=0.03,
    max_p90_m=0.08,
    min_agreement=0.5,
):
    """Fit one scale on alternating 8px blocks; validate on disjoint blocks.

    Sixteen spatial tile medians balance the fit. A bounded log-ratio MAD filter
    rejects outliers on fit blocks only. Validation never selects or refits scale.
    On rejection, scale is None and candidate_scale is diagnostic only.
    """
    predicted, measured = _depth(predicted), _depth(measured)
    support = np.asarray(support, bool)
    if predicted.shape != measured.shape or support.shape != measured.shape:
        raise ValueError("Matching native-ray grids required")
    if (
        type(min_samples) is not int
        or min_samples < 1
        or type(min_tiles) is not int
        or not 1 <= min_tiles <= 16
        or len(scale_bounds) != 2
        or not 0 < scale_bounds[0] < scale_bounds[1]
        or not np.isfinite([*scale_bounds, tolerance_m, max_median_m, max_p90_m, min_agreement]).all()
        or not 0 < tolerance_m <= max_median_m <= max_p90_m
        or not 0 <= min_agreement <= 1
    ):
        raise ValueError("Finite bounded scale acceptance policy required")
    h, w = measured.shape
    yy, xx = np.indices((h, w))
    valid = support & np.isfinite(predicted) & (predicted > 0) & np.isfinite(measured) & (measured > 0)
    split = ((xx // 8 + yy // 8) % 2) == 0
    fit, check = valid & split, valid & ~split
    tile = np.minimum(yy * 4 // h, 3) * 4 + np.minimum(xx * 4 // w, 3)
    fit_tiles = [i for i in range(16) if np.sum(fit & (tile == i)) >= 8]
    check_tiles = [i for i in range(16) if np.sum(check & (tile == i)) >= 8]
    report = dict(
        accepted=False,
        scale=None,
        candidate_scale=None,
        fit_samples=int(fit.sum()),
        check_samples=int(check.sum()),
        fit_tiles=len(fit_tiles),
        check_tiles=len(check_tiles),
    )
    if min(fit.sum(), check.sum()) < min_samples or min(len(fit_tiles), len(check_tiles)) < min_tiles:
        return dict(report, reason="insufficient spatial calibration support")
    ratios = np.zeros_like(predicted)
    ratios[valid] = np.log(measured[valid] / predicted[valid])
    initial = np.median([np.median(ratios[fit & (tile == i)]) for i in fit_tiles])
    mad = np.median(abs(ratios[fit] - initial))
    cutoff = max(0.02, min(0.15, 3 * 1.4826 * mad))
    inlier = fit & (abs(ratios - initial) <= cutoff)
    balanced = [
        np.median(ratios[inlier & (tile == i)]) for i in fit_tiles if np.sum(inlier & (tile == i)) >= 8
    ]
    if len(balanced) < min_tiles or inlier.sum() < min_samples:
        return dict(report, reason="insufficient robust calibration support")
    scale = float(np.exp(np.median(balanced)))
    residual = abs(predicted[check] * scale - measured[check])
    report.update(
        candidate_scale=scale,
        fit_inliers=int(inlier.sum()),
        validation_median_m=float(np.median(residual)),
        validation_p90_m=float(np.quantile(residual, 0.9)),
        validation_agreement=float(np.mean(residual <= tolerance_m)),
        log_ratio_mad=float(mad),
    )
    if not scale_bounds[0] <= scale <= scale_bounds[1]:
        return dict(report, reason="scale outside bounds")
    if (
        report["validation_median_m"] > max_median_m
        or report["validation_p90_m"] > max_p90_m
        or report["validation_agreement"] < min_agreement
    ):
        return dict(report, reason="disjoint calibration check failed")
    return dict(
        report, accepted=True, scale=scale, reason="passed calibration check; accuracy not established"
    )


def view_consistency(depth, k, pose, witnesses, *, tolerance_m=0.015, min_support=2, min_baseline_m=0.02):
    """Reproject into distinct witness cameras; retain corroboration without contradiction.

    Witnesses are (id, depth, K, world_from_camera). Farther observed depth
    contradicts the query point. Nearer depth is possible occlusion, hence unknown.
    Only explicit positive depth is evidence. No fitting or averaging occurs.
    """
    depth, k, pose = _depth(depth), _camera(k), rigid(pose, "pose")
    if (
        type(min_support) is not int
        or min_support < 1
        or not np.isfinite([tolerance_m, min_baseline_m]).all()
        or not 0 < tolerance_m <= 0.05
        or min_baseline_m < 0
    ):
        raise ValueError("Positive bounded consistency policy required")
    valid = np.isfinite(depth) & (depth > 0)
    points = camera_points(np.where(valid, depth, 0), k) @ pose[:3, :3].T + pose[:3, 3]
    support = np.zeros(depth.shape, np.uint32)
    contradictions = np.zeros(depth.shape, np.uint32)
    records, ids = [], set()
    for identity, z, kt, pt in witnesses:
        if identity in ids:
            raise ValueError("Witness identities must be unique")
        ids.add(identity)
        z, kt, pt = _depth(z), _camera(kt), rigid(pt, "witness pose")
        baseline = float(np.linalg.norm(pose[:3, 3] - pt[:3, 3]))
        if baseline < max(min_baseline_m, 1e-9):
            records.append(dict(id=identity, skipped="insufficient baseline"))
            continue
        camera = (points - pt[:3, 3]) @ pt[:3, :3]
        projected = camera @ kt.T
        uv = np.rint(projected[..., :2] / np.maximum(projected[..., 2:], 1e-8)).astype(int)
        h, w = z.shape
        inside = valid & (camera[..., 2] > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < w)
        inside &= (uv[..., 1] >= 0) & (uv[..., 1] < h)
        observed = z[np.clip(uv[..., 1], 0, h - 1), np.clip(uv[..., 0], 0, w - 1)]
        usable = inside & np.isfinite(observed) & (observed > 0)
        delta = np.where(usable, observed - camera[..., 2], 0)
        agree, conflict = usable & (abs(delta) <= tolerance_m), usable & (delta > tolerance_m)
        support += agree
        contradictions += conflict
        records.append(
            dict(
                id=identity,
                baseline_m=baseline,
                comparable=int(usable.sum()),
                agreement=int(agree.sum()),
                farther_contradiction=int(conflict.sum()),
                nearer_or_occluded=int((usable & (delta < -tolerance_m)).sum()),
            )
        )
    return dict(
        keep=valid & (support >= min_support) & (contradictions == 0),
        support=support,
        contradictions=contradictions,
        witnesses=records,
    )
