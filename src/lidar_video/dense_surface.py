"""Calibrated RGB plane sweep and reciprocal multi-view surface acceptance.

No shape completion or sensor depth is used. Photometric agreement is a
correspondence hypothesis, not an accuracy certificate. World poses determine
metric scale; callers must disclose any LiDAR used to estimate those poses.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from lidar_video.camera_validation import rigid
from lidar_video.capture import camera_points
from lidar_video.metric_fusion import _camera, _depth


@dataclass(frozen=True)
class DensePolicy:
    near_m: float = 0.1
    far_m: float = 2.0
    planes: int = 384
    window: int = 5
    min_std: float = 3.0
    min_ncc: float = 0.85
    uniqueness_margin: float = 0.04
    ambiguity_separation_px: float = 1.5
    min_baseline_m: float = 0.01
    min_parallax_degrees: float = 1.0
    max_depth_step_m: float = 0.02
    depth_tolerance_m: float = 0.015
    cycle_tolerance_px: float = 1.5

    def validate(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError("Finite dense policy required")
        if not 0 < self.near_m < self.far_m or type(self.planes) is not int or not 16 <= self.planes <= 2048:
            raise ValueError("Ordered positive metric range and 16..2048 planes required")
        if type(self.window) is not int or self.window not in (3, 5, 7, 9):
            raise ValueError("Odd patch window 3..9 required")
        if not (0 < self.min_std <= 255 and 0 < self.min_ncc < 1 and 0 < self.uniqueness_margin < 1):
            raise ValueError("Positive texture, NCC and ambiguity checks required")
        if not (0 < self.min_baseline_m and 0 < self.min_parallax_degrees < 45):
            raise ValueError("Positive baseline and bounded parallax required")
        if not (0 < self.depth_tolerance_m <= 0.05 and 0 < self.max_depth_step_m <= 0.05):
            raise ValueError("Depth checks must be positive and at most 50mm")
        if not 0 < self.cycle_tolerance_px <= 3:
            raise ValueError("Cycle check must be positive and at most 3px")
        if not 0 < self.ambiguity_separation_px <= 3:
            raise ValueError("Ambiguity separation must be positive and at most 3px")


def _rgb(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 10:
        raise ValueError("Nonempty uint8 RGB image at least 10x10 required")
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)


def plane_sweep_pair(
    reference, target, k, target_k, pose, target_pose, *, source_ids, policy=DensePolicy(), static_scene=False
):
    """Match directly in original image grids using frontoparallel plane warps.

    ZNCC removes affine brightness changes. The nonlocal second-best plane,
    texture, sampling resolution and triangulation angle all gate acceptance.
    No smoothing, hole filling, disparity regularization or learned prior occurs.
    The full cost volume is transient; output depths remain pair proposals.
    """
    policy.validate()
    if static_scene is not True:
        raise ValueError("Static-scene assumption must be explicit")
    if len(source_ids) != 2 or source_ids[0] == source_ids[1]:
        raise ValueError("Two distinct original source IDs required")
    a, b = _rgb(reference), _rgb(target)
    k, target_k = _camera(k), _camera(target_k)
    pose, target_pose = rigid(pose, "pose"), rigid(target_pose, "target_pose")
    baseline = float(np.linalg.norm(pose[:3, 3] - target_pose[:3, 3]))
    if baseline < policy.min_baseline_m:
        raise ValueError("Insufficient metric camera baseline")
    h, w = a.shape
    if policy.planes * h * w > 180_000_000:
        raise ValueError("Cost volume exceeds bounded CPU memory budget; resize images and intrinsics")
    yy, xx = np.indices((h, w), dtype=np.float32)
    rays = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(k).T
    relative = np.linalg.inv(target_pose) @ pose
    projected_rays = (rays @ relative[:3, :3].T) @ target_k.T
    projected_t = relative[:3, 3] @ target_k.T
    rho = np.linspace(1 / policy.far_m, 1 / policy.near_m, policy.planes)
    cost = np.full((policy.planes, h, w), 2.0, np.float32)
    window = (policy.window,) * 2

    def mean(x):
        return cv2.boxFilter(x, -1, window, normalize=True, borderType=cv2.BORDER_CONSTANT)

    ma = mean(a)
    va = np.maximum(mean(a * a) - ma * ma, 0)
    texture = va >= policy.min_std**2
    kernel = np.ones(window, np.uint8)
    for index, inverse_depth in enumerate(rho):
        projected = projected_rays + projected_t * inverse_depth
        z = projected[..., 2]
        u = (projected[..., 0] / np.maximum(z, 1e-9)).astype(np.float32)
        v = (projected[..., 1] / np.maximum(z, 1e-9)).astype(np.float32)
        inside = (z > 0) & (u >= 1) & (v >= 1) & (u < b.shape[1] - 2) & (v < b.shape[0] - 2)
        complete = (
            cv2.erode(inside.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        )
        warped = cv2.remap(b, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mb = mean(warped)
        vb = np.maximum(mean(warped * warped) - mb * mb, 0)
        ncc = (mean(a * warped) - ma * mb) / np.sqrt(np.maximum(va * vb, 1e-8))
        usable = complete & texture & (vb >= policy.min_std**2)
        cost[index] = np.where(usable, 1 - np.clip(ncc, -1, 1), 2)
    best = cost.argmin(0)
    y, x = np.indices((h, w))
    best_cost = cost[best, y, x].copy()
    # Define a different match in pixels, not plane indices: inverse-depth
    # oversampling must not turn the same subpixel match into an alternative.
    best_projected = projected_rays + projected_t * rho[best, None]
    best_uv = best_projected[..., :2] / np.maximum(best_projected[..., 2:], 1e-9)
    second = np.full((h, w), 2.0, np.float32)
    for index in range(policy.planes):
        projected = projected_rays + projected_t * rho[index]
        uv = projected[..., :2] / np.maximum(projected[..., 2:], 1e-9)
        separated = np.linalg.norm(uv - best_uv, axis=-1) >= policy.ambiguity_separation_px
        second = np.minimum(second, np.where(separated, cost[index], 2))
    low = cost[np.maximum(best - 1, 0), y, x]
    high = cost[np.minimum(best + 1, policy.planes - 1), y, x]
    curvature = low + high - 2 * best_cost
    delta = np.clip((low - high) / np.maximum(2 * curvature, 1e-8), -0.5, 0.5)
    inverse = rho[best] + delta * (rho[1] - rho[0])
    depth = 1 / inverse
    depth_step = abs(1 / rho[np.maximum(best - 1, 0)] - 1 / rho[np.minimum(best + 1, policy.planes - 1)]) / 2
    points = rays * depth[..., None]
    target_center = (target_pose[:3, 3] - pose[:3, 3]) @ pose[:3, :3]
    other_ray = points - target_center
    cosine = np.sum(points * other_ray, axis=-1) / np.maximum(
        np.linalg.norm(points, axis=-1) * np.linalg.norm(other_ray, axis=-1), 1e-9
    )
    angle = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    keep = (best > 0) & (best < policy.planes - 1) & (best_cost <= 1 - policy.min_ncc)
    keep &= (second - best_cost >= policy.uniqueness_margin) & (curvature > 1e-6)
    keep &= (depth_step <= policy.max_depth_step_m) & (angle >= policy.min_parallax_degrees)
    return dict(
        depth=np.where(keep, depth, 0).astype(np.float32),
        valid=keep,
        ncc=1 - best_cost,
        uniqueness=second - best_cost,
        depth_step_m=depth_step.astype(np.float32),
        metadata=dict(
            source_ids=list(source_ids),
            baseline_m=baseline,
            policy=asdict(policy),
            proposed_pixels=int(keep.sum()),
            geometry_checked=False,
            method="original-grid inverse-depth ZNCC plane sweep",
            metric_scale="supplied camera poses",
            completion=False,
        ),
    )


def reciprocal_check(depth, k, pose, witness_depth, witness_k, witness_pose, *, policy=DensePolicy()):
    """Check depth and pixel round trips without interpolating across missing rays.

    A nearer witness depth is possibly occluded, not supporting evidence. A
    farther depth is a free-space conflict. Both fail reciprocal acceptance.
    """
    policy.validate()
    depth, witness_depth = _depth(depth), _depth(witness_depth)
    k, witness_k = _camera(k), _camera(witness_k)
    pose, witness_pose = rigid(pose, "pose"), rigid(witness_pose, "witness_pose")
    if np.linalg.norm(pose[:3, 3] - witness_pose[:3, 3]) < policy.min_baseline_m:
        raise ValueError("Insufficient witness baseline")
    valid = np.isfinite(depth) & (depth > 0)
    point = camera_points(np.where(valid, depth, 0), k)
    relative = np.linalg.inv(witness_pose) @ pose
    projected = point @ relative[:3, :3].T + relative[:3, 3]
    pixel = projected @ witness_k.T
    uv = pixel[..., :2] / np.maximum(pixel[..., 2:], 1e-9)
    xy = np.rint(uv).astype(np.int64)
    h, w = witness_depth.shape
    inside = (
        valid
        & (projected[..., 2] > 0)
        & (xy[..., 0] >= 0)
        & (xy[..., 0] < w)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] < h)
    )
    z = witness_depth[np.clip(xy[..., 1], 0, h - 1), np.clip(xy[..., 0], 0, w - 1)]
    usable = inside & np.isfinite(z) & (z > 0)
    safe_z = np.where(usable, z, 0)
    ray = np.concatenate([xy, np.ones((*depth.shape, 1))], -1) @ np.linalg.inv(witness_k).T
    back = ((ray * safe_z[..., None]) - relative[:3, 3]) @ relative[:3, :3]
    back_pixel = back @ k.T
    back_uv = back_pixel[..., :2] / np.maximum(back_pixel[..., 2:], 1e-9)
    yy, xx = np.indices(depth.shape)
    cycle = np.linalg.norm(back_uv - np.stack([xx, yy], -1), axis=-1)
    delta = safe_z - projected[..., 2]
    keep = (
        usable
        & (abs(delta) <= policy.depth_tolerance_m)
        & (cycle <= policy.cycle_tolerance_px)
        & (back[..., 2] > 0)
    )
    return dict(
        keep=keep,
        comparable=usable,
        cycle_error_px=np.where(usable, cycle, np.nan),
        farther_conflict=usable & (delta > policy.depth_tolerance_m),
        nearer_or_occluded=usable & (delta < -policy.depth_tolerance_m),
    )


def corroborate_pairs(proposals, k, pose, *, reference_id, policy=DensePolicy()):
    """Require two separated source cameras and reciprocal pair agreement.

    Each proposal has id, depth, reverse_depth, K, and pose. Sources are unique;
    support counts cameras, never pixels or duplicate proposals. No LiDAR enters.
    Returned source-index maps preserve which pair justifies each kept sample.
    """
    policy.validate()
    if len(proposals) < 2:
        raise ValueError("At least two distinct witnesses required")
    ids = [p["id"] for p in proposals]
    if len(set(ids)) != len(ids) or reference_id in ids:
        raise ValueError("Distinct original witness IDs excluding reference required")
    arrays, witnesses = [], []
    for p in proposals:
        z = _depth(p["depth"])
        check = reciprocal_check(z, k, pose, p["reverse_depth"], p["K"], p["pose"], policy=policy)
        arrays.append(np.where(check["keep"], z, 0))
        witnesses.append(
            dict(
                id=p["id"],
                proposed=int(np.count_nonzero(z)),
                reciprocal=int(check["keep"].sum()),
                farther_conflict=int(check["farther_conflict"].sum()),
            )
        )
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Proposals must share the reference grid")
    depth = np.zeros_like(arrays[0], np.float32)
    best_spread = np.full(depth.shape, np.inf)
    sources = np.full((*depth.shape, 2), -1, np.int16)
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            if (
                np.linalg.norm(
                    np.asarray(proposals[i]["pose"])[:3, 3] - np.asarray(proposals[j]["pose"])[:3, 3]
                )
                < policy.min_baseline_m
            ):
                continue
            a, b = arrays[i], arrays[j]
            spread = abs(a - b)
            valid = (a > 0) & (b > 0) & (spread <= policy.depth_tolerance_m) & (spread < best_spread)
            depth[valid] = ((a + b) / 2)[valid]
            best_spread[valid] = spread[valid]
            sources[valid] = [i, j]
    # Any other reciprocal proposal with a materially different depth vetoes
    # acceptance; do not cherry-pick the closest pair in a conflicting cluster.
    conflict = np.zeros(depth.shape, bool)
    for z in arrays:
        conflict |= (z > 0) & (depth > 0) & (abs(z - depth) > policy.depth_tolerance_m)
    depth[conflict] = 0
    sources[depth == 0] = -1
    return dict(
        depth=depth,
        valid=depth > 0,
        source_indices=sources,
        metadata=dict(
            reference_id=reference_id,
            witness_ids=ids,
            witnesses=witnesses,
            accepted_pixels=int((depth > 0).sum()),
            conflict_pixels=int(conflict.sum()),
            independent_evaluation=False,
            completion=False,
        ),
    )


def close_view_triangles(depth, source_indices, k, pose, witnesses, edges, *, policy=DensePolicy()):
    """Require the selected witnesses to agree without reusing reference edges.

    witnesses follows corroborate_pairs' source order and contains id/K/pose.
    edges[(a,b)] is a depth map in camera a matched against b. No missing edge
    counts as evidence. The geometry checks reuse source images, so this is a
    consistency filter and is never an independent evaluation.
    """
    policy.validate()
    depth = _depth(depth)
    selected = np.asarray(source_indices)
    ids = [w["id"] for w in witnesses]
    if len(ids) != len(set(ids)):
        raise ValueError("Distinct witness IDs required")
    if selected.shape != (*depth.shape, 2) or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("Two integer witness indices per pixel required")
    valid = np.isfinite(depth) & (depth > 0)
    if np.any(
        valid
        & ((selected[..., 0] < 0) | (selected[..., 0] >= selected[..., 1]) | (selected[..., 1] >= len(ids)))
    ):
        raise ValueError("Ordered distinct witness indices required for every positive depth")
    keep = np.zeros(depth.shape, bool)
    for i in range(len(witnesses)):
        for j in range(i + 1, len(witnesses)):
            chosen = valid & (selected[..., 0] == i) & (selected[..., 1] == j)
            a, b = witnesses[i], witnesses[j]
            if not chosen.any() or (a["id"], b["id"]) not in edges or (b["id"], a["id"]) not in edges:
                continue
            if (
                np.linalg.norm(np.asarray(a["pose"])[:3, 3] - np.asarray(b["pose"])[:3, 3])
                < policy.min_baseline_m
            ):
                continue
            ca = reciprocal_check(depth, k, pose, edges[a["id"], b["id"]], a["K"], a["pose"], policy=policy)
            cb = reciprocal_check(depth, k, pose, edges[b["id"], a["id"]], b["K"], b["pose"], policy=policy)
            keep |= chosen & ca["keep"] & cb["keep"]
    return dict(
        depth=np.where(keep, depth, 0).astype(np.float32),
        keep=keep,
        metadata=dict(
            accepted_pixels=int(keep.sum()),
            input_pixels=int(valid.sum()),
            independent_evaluation=False,
            witness_to_witness_checks=True,
        ),
    )


def verify_depth_proposal(depth, k, pose, witnesses, edges, *, reference_id, policy=DensePolicy()):
    """Check an external depth proposal through two witnesses and their third edges.

    Each witness supplies id/K/pose and reverse_depth matched toward the
    reference. The same proposal is tested against multiple measurements; this
    must not be described as multiple independently estimated reference depths.
    The proposal's modality/provenance must travel with the result externally.
    No fitting, filling, scaling or mesh completion occurs here.
    """
    proposals = [dict(w, depth=depth) for w in witnesses]
    pair = corroborate_pairs(proposals, k, pose, reference_id=reference_id, policy=policy)
    triangle = close_view_triangles(
        pair["depth"], pair["source_indices"], k, pose, witnesses, edges, policy=policy
    )
    return dict(
        depth=triangle["depth"],
        keep=triangle["keep"],
        source_indices=pair["source_indices"],
        metadata=dict(
            reference_id=reference_id,
            pair=pair["metadata"],
            triangle=triangle["metadata"],
            external_proposal=True,
            completion=False,
        ),
    )
