"""Support-relative observed fragments, never semantic object completion."""

import numpy as np

from lidar_video.geometry import clusters, fit_support


def support_fragment_hypotheses(
    points, *, z_up=False, height_range=(0.015, 0.3), radius_m=0.008, min_points=20
):
    """Separate a dominant horizontal plane from raised surface fragments.

    Intended for tabletop recordings with a supplied gravity-aligned frame.
    Plane fitting and checking use disjoint 6mm spatial blocks; both remain
    construction data. Connected fragments can merge objects or split one
    object, so fragment IDs must not be promoted to instance/scene-graph IDs.
    Hidden bottoms, footprints and unobserved points are never synthesized.
    """
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("Finite world points (N,3) required")
    if z_up is not True:
        raise ValueError("Gravity-aligned z-up frame must be explicit")
    if (
        len(height_range) != 2
        or not np.isfinite([*height_range, radius_m]).all()
        or not 0.003 < height_range[0] < height_range[1]
        or not 0 < radius_m <= 0.02
        or type(min_points) is not int
        or min_points < 3
    ):
        raise ValueError("Bounded support-fragment policy required")
    report = dict(
        accepted=False,
        semantic_instances=False,
        completion=False,
        scope="Geometric fragment hypotheses; not independent object accuracy",
        plane_tolerance_m=0.003,
        height_range_m=list(height_range),
        radius_m=radius_m,
        min_points=min_points,
    )
    empty = dict(labels=np.full(len(p), -1, np.int32), fragments=[], report=report)
    if len(p) < 400:
        report["reason"] = "insufficient surface evidence"
        return empty
    blocks = np.floor(p / 0.006).astype(np.int64)
    split = (blocks.sum(axis=1) % 2) == 0
    if min(split.sum(), (~split).sum()) < 200:
        report["reason"] = "insufficient disjoint spatial blocks"
        return empty
    try:
        transform, rmse = fit_support(p[split])
    except ValueError as error:
        report["reason"] = str(error)
        return empty
    q = np.einsum("ij,kj->ik", p, transform[:3, :3]) + transform[:3, 3]
    fit_plane = split & (abs(q[:, 2]) < 0.003)
    check_plane = (~split) & (abs(q[:, 2]) < 0.003)
    fraction = float(check_plane.sum() / (~split).sum())
    report.update(fit_rmse_m=rmse, check_inliers=int(check_plane.sum()), check_support_fraction=fraction)
    if check_plane.sum() < 100 or fraction < 0.2:
        report["reason"] = "disjoint plane-support check failed"
        return empty
    low, high = np.quantile(q[fit_plane, :2], [0.005, 0.995], axis=0)
    if np.any(high - low < 0.05):
        report["reason"] = "insufficient spatial extent for support"
        return empty
    eligible = (
        (q[:, 2] > height_range[0])
        & (q[:, 2] < height_range[1])
        & np.all((q[:, :2] >= low) & (q[:, :2] <= high), axis=1)
    )
    source = np.flatnonzero(eligible)
    # Cluster only existing points. No plane projection, proxy, or base creation.
    groups = clusters(q[eligible], radius=radius_m, min_points=min_points)
    groups.sort(key=lambda x: tuple(np.mean(q[source[x]], axis=0)))
    labels = empty["labels"]
    fragments = []
    for identity, group in enumerate(groups):
        indices = source[group]
        labels[indices] = identity
        fragments.append(
            dict(
                fragment_id=identity,
                point_indices=indices.tolist(),
                observed_points=len(indices),
                observed_bounds_world_m=[p[indices].min(0).tolist(), p[indices].max(0).tolist()],
                semantic_class="unknown",
                complete_object=False,
            )
        )
    report.update(
        accepted=True,
        reason="plane check passed; object identities unresolved",
        scene_from_world=transform.tolist(),
        support_bounds_xy_m=[low.tolist(), high.tolist()],
        raised_samples=int(eligible.sum()),
        clustered_samples=int(np.sum(labels >= 0)),
        fragment_count=len(groups),
    )
    return dict(labels=labels, fragments=fragments, report=report)
