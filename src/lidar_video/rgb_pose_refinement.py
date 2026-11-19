"""Bounded camera refinement from RGB tracks with disjoint-track acceptance.

Metric scale/gauge comes from fixed cameras, device-pose priors and optional
prechecked native depth anchors. RGB tracks remain correspondence hypotheses.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from lidar_video.camera_validation import rigid
from lidar_video.landmark_tracks import triangulate_landmark
from lidar_video.rgb_association import RGBPolicy


def _track_score(tracks, views):
    errors, invalid = [], 0
    for track in tracks:
        result = triangulate_landmark(track["observations"], views, RGBPolicy(max_reprojection_px=100))
        if not result["passed"]:
            invalid += 1
        errors.extend(result.get("reprojection_errors_px", [100.0]))
    return dict(median_px=float(np.median(errors)), p90_px=float(np.quantile(errors, 0.9)), invalid=invalid)


def refine_camera_component(views, tracks, *, static_scene=False, max_nfev=500):
    """Refine one connected component; use every fifth track only for checking.

    Each track has world_point, observations[{view,pixel}], and optional
    depth_anchors[{view,z}]. Caller must gate depth anchors before this operation.
    Rejected candidates are returned separately; applied poses remain original.
    """
    if static_scene is not True or type(max_nfev) is not int or not 1 <= max_nfev <= 2000:
        raise ValueError("Explicit static scene and bounded solve budget required")
    poses = np.array([rigid(v["world_from_camera"], "pose") for v in views])
    ks = np.array([v["K"] for v in views], float)
    n = len(views)
    if n < 3 or ks.shape != (n, 3, 3) or not np.isfinite(ks).all():
        raise ValueError("At least three finite calibrated cameras required")
    if (
        np.any(ks[:, (0, 1), (0, 1)] <= 0)
        or not np.allclose(ks[:, 2], [0, 0, 1])
        or not np.allclose(ks[:, 0, 1], 0)
        or not np.allclose(ks[:, 1, 0], 0)
    ):
        raise ValueError("Positive pinhole intrinsics required")
    if len({v["source_frame"] for v in views}) != n:
        raise ValueError("Distinct source cameras required")
    for t in tracks:
        obs = t["observations"]
        if len(obs) < 3 or len({o["view"] for o in obs}) != len(obs):
            raise ValueError("Three unique views per track required")
        if any(
            type(o["view"]) is not int
            or not 0 <= o["view"] < n
            or np.asarray(o["pixel"]).shape != (2,)
            or not np.isfinite(o["pixel"]).all()
            for o in obs
        ):
            raise ValueError("Invalid track observations")
        if np.asarray(t["world_point"]).shape != (3,) or not np.isfinite(t["world_point"]).all():
            raise ValueError("Finite initial world point required")
        for a in t.get("depth_anchors", []):
            if a["view"] not in {o["view"] for o in obs} or not np.isfinite(a["z"]) or a["z"] <= 0:
                raise ValueError("Depth anchor must match an observed view")
    check = [t for i, t in enumerate(tracks) if i % 5 == 0]
    fit = [t for i, t in enumerate(tracks) if i % 5 != 0]
    report = dict(
        accepted=False,
        fit_tracks=len(fit),
        check_tracks=len(check),
        fixed_camera=0,
        limits=dict(translation_norm_m=0.02, rotation_norm_degrees=2.0, point_axis_m=0.05),
        validation="Disjoint construction tracks, not independent evaluation",
    )
    if len(fit) < 32 or len(check) < 8:
        return dict(
            poses=poses, candidate_poses=poses.copy(), report=dict(report, reason="insufficient tracks")
        )
    fit_counts = np.bincount([o["view"] for t in fit for o in t["observations"]], minlength=n)
    check_counts = np.bincount([o["view"] for t in check for o in t["observations"]], minlength=n)
    if np.any(fit_counts < 12) or np.any(check_counts < 3):
        return dict(
            poses=poses,
            candidate_poses=poses.copy(),
            report=dict(report, reason="insufficient per-camera support"),
        )
    connected = {0}
    while True:
        extended = set(connected)
        for t in fit:
            ids = {o["view"] for o in t["observations"]}
            if ids & connected:
                extended |= ids
        if extended == connected:
            break
        connected = extended
    if len(connected) != n:
        return dict(
            poses=poses,
            candidate_poses=poses.copy(),
            report=dict(report, reason="fit camera graph is disconnected"),
        )
    points = np.array([t["world_point"] for t in fit])
    nc = (n - 1) * 6
    obs = [(i, o["view"], o["pixel"]) for i, t in enumerate(fit) for o in t["observations"]]
    pi = np.array([x[0] for x in obs])
    ci = np.array([x[1] for x in obs])
    pixels = np.array([x[2] for x in obs])
    anchors = [(i, a["view"], a["z"]) for i, t in enumerate(fit) for a in t.get("depth_anchors", [])]
    ai = np.array([x[0] for x in anchors], int)
    ac = np.array([x[1] for x in anchors], int)
    az = np.array([x[2] for x in anchors])

    def unpack(x):
        refined = poses.copy()
        delta = x[:nc].reshape(-1, 6)
        refined[1:, :3, :3] = poses[1:, :3, :3] @ Rotation.from_rotvec(delta[:, :3]).as_matrix()
        refined[1:, :3, 3] += delta[:, 3:]
        return refined, points + x[nc:].reshape(-1, 3)

    def residual(x):
        cameras, p = unpack(x)
        cp = np.einsum("ni,nij->nj", p[pi] - cameras[ci, :3, 3], cameras[ci, :3, :3])
        uv = np.einsum("ni,nji->nj", cp, ks[ci])
        image = (uv[:, :2] / np.maximum(uv[:, 2:], 0.01) - pixels).ravel()
        if len(anchors):
            ap = np.einsum("ni,nij->nj", p[ai] - cameras[ac, :3, 3], cameras[ac, :3, :3])
            measured = (ap[:, 2] - az) / 0.015
        else:
            measured = np.empty(0)
        prior = x[:nc].reshape(-1, 6) / np.array([*[np.deg2rad(0.5)] * 3, *[0.01] * 3])
        return np.r_[image, measured, prior.ravel(), x[nc:] / 0.02]

    rows = len(obs) * 2 + len(anchors) + nc + len(fit) * 3
    cols = nc + len(fit) * 3
    sparsity = lil_matrix((rows, cols), dtype=int)
    for index, (point, camera, _) in enumerate(obs):
        sparsity[2 * index : 2 * index + 2, nc + 3 * point : nc + 3 * point + 3] = 1
        if camera:
            sparsity[2 * index : 2 * index + 2, (camera - 1) * 6 : camera * 6] = 1
    for index, (point, camera, _) in enumerate(anchors):
        row = len(obs) * 2 + index
        sparsity[row, nc + 3 * point : nc + 3 * point + 3] = 1
        if camera:
            sparsity[row, (camera - 1) * 6 : camera * 6] = 1
    sparsity[-cols:, :] = np.eye(cols, dtype=int)
    limits = np.r_[np.tile([*[np.deg2rad(2.0)] * 3, *[0.02] * 3], n - 1), np.full(len(fit) * 3, 0.05)]
    initial = _track_score(check, views)
    solve = least_squares(
        residual,
        np.zeros(cols),
        jac_sparsity=sparsity.tocsr(),
        bounds=(-limits, limits),
        x_scale=limits,
        tr_options=dict(atol=1e-10, btol=1e-10, maxiter=1000),
        loss="huber",
        f_scale=1.5,
        max_nfev=max_nfev,
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
    )
    candidate, _ = unpack(solve.x)
    updated = [dict(v, world_from_camera=p) for v, p in zip(views, candidate)]
    after = _track_score(check, updated)
    delta = solve.x[:nc].reshape(-1, 6)
    translation = np.linalg.norm(delta[:, 3:], axis=1)
    rotation = np.degrees(np.linalg.norm(delta[:, :3], axis=1))
    reasons = []
    if not solve.success:
        reasons.append("solver did not converge")
    if np.any(translation >= 0.019) or np.any(rotation >= 1.9):
        reasons.append("pose update approaches bound")
    if np.any(abs(solve.x[nc:]) >= 0.0475):
        reasons.append("landmark update approaches bound")
    if after["invalid"] > initial["invalid"]:
        reasons.append("invalid check tracks increased")
    if after["p90_px"] > 2.0:
        reasons.append("absolute check reprojection exceeds 2 pixels")
    if after["median_px"] > 0.95 * initial["median_px"] or after["p90_px"] > 1.05 * initial["p90_px"]:
        reasons.append("disjoint-track improvement not established")
    report.update(
        accepted=not reasons,
        reason="; ".join(reasons) if reasons else "passed construction-track check",
        before=initial,
        after=after,
        nfev=solve.nfev,
        native_depth_anchors=len(anchors),
        landmark_prior_sigma_m=0.02,
        max_landmark_displacement_m=float(np.linalg.norm(solve.x[nc:].reshape(-1, 3), axis=1).max()),
        max_translation_m=float(translation.max()),
        max_rotation_degrees=float(rotation.max()),
    )
    return dict(poses=candidate if not reasons else poses, candidate_poses=candidate, report=report)
