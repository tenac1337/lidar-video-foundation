"""Compare independently triangulated RGB landmarks with native depth evidence.

Agreement between depth maps can describe a background plane rather than the
visible object. This diagnostic keeps RGB geometry and depth support separate.
It never alters cameras, measurements, instance identity, or reconstruction assets.
"""

import cv2
import numpy as np

from lidar_video.capture import validate
from lidar_video.landmark_tracks import triangulate_landmark
from lidar_video.native_depth import native_intrinsics, sample_ids
from lidar_video.rgb_association import RGBPolicy


def audit_landmark_depth(
    data, observations, labels, *, static_scene=False, depth_tolerance_m=0.015, min_confidence=2, policy=None
):
    """Retriangulate a 3+ view RGB feature before consulting its depth samples.

    Each observation supplies capture-frame index ``view``, RGB ``pixel`` and
    positive instance ``label``. Labels are registered to the supplied capture.
    A native-pixel footprint plus reprojection margin excludes mask boundaries.
    Two separated agreeing depth views are needed; any farther depth return
    vetoes support. Nearer returns are occlusion/geometry ambiguity, not support.
    This is point-level consistency only, never permission to carve foreground.
    """
    validate(data)
    if (
        static_scene is not True
        or not np.isfinite(depth_tolerance_m)
        or not 0 < depth_tolerance_m <= 0.015
        or type(min_confidence) is not int
        or min_confidence not in (1, 2)
    ):
        raise ValueError("Explicit static scene, confidence level and <=15 mm finite tolerance required")
    policy = policy or RGBPolicy()
    labels = np.asarray(labels)
    if labels.shape != data["depth"].shape or labels.dtype.kind not in "ui":
        raise ValueError("Indexed labels must match the capture pixel grid")
    n, h, w = labels.shape
    observations = list(observations)
    for o in observations:
        pixel = np.asarray(o["pixel"], float)
        if (
            type(o["view"]) is not int
            or not 0 <= o["view"] < n
            or type(o["label"]) is not int
            or o["label"] < 1
            or pixel.shape != (2,)
            or not np.isfinite(pixel).all()
            or np.any(pixel < 0)
            or pixel[0] >= w - 0.5
            or pixel[1] >= h - 0.5
        ):
            raise ValueError("Finite in-image feature pixels, capture-frame indices and labels required")
        x, y = np.rint(pixel).astype(int)
        if labels[o["view"], y, x] != o["label"]:
            raise ValueError("RGB feature does not belong to its declared mask")
    ks = np.broadcast_to(data["K"], (n, 3, 3))
    views = [dict(K=ks[f], world_from_camera=data["world_from_camera"][f]) for f in range(n)]
    geometry = triangulate_landmark(observations, views, policy)
    report = dict(
        rgb_geometry=geometry,
        depth_supports_landmark=False,
        observations=[],
        depth_tolerance_m=depth_tolerance_m,
        min_confidence=min_confidence,
        foreground_carving_authorized=False,
        scope="Point-level cross-modal consistency; no object identity, surface completeness or calibration proof",
    )
    if not geometry["passed"]:
        report["reason"] = "RGB triangulation failed; depth cannot rescue the feature"
        return report
    point = np.asarray(geometry["world_point"])
    native_ks = native_intrinsics(data)
    margins = np.maximum(ks[:, 0, 0] / native_ks[:, 0, 0], ks[:, 1, 1] / native_ks[:, 1, 1])
    margins += policy.max_reprojection_px
    agreeing = []
    farther = False
    for o in observations:
        f = o["view"]
        x, y = np.rint(o["pixel"]).astype(int)
        pose = data["world_from_camera"][f]
        camera = (point - pose[:3, 3]) @ pose[:3, :3]
        observed = float(data["depth"][f, y, x])
        confidence = int(data["confidence"][f, y, x])
        sid = int(sample_ids(data, f)[y, x])
        record = dict(
            frame=f,
            source_frame=int(data.get("source_frame", np.arange(n))[f]),
            pixel=list(map(float, o["pixel"])),
            label=o["label"],
            native_sample_id=sid,
            confidence=confidence,
            rgb_camera_depth_m=float(camera[2]),
            measured_depth_m=observed if np.isfinite(observed) and observed > 0 else None,
            mask_interior_margin_px=float(margins[f]),
        )
        target = (labels[f] == o["label"]).astype(np.uint8)
        distance = cv2.distanceTransform(np.pad(target, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        if distance[y, x] <= margins[f]:
            record["status"] = "unknown_mask_boundary"
        elif sid < 0 or not np.isfinite(observed) or observed <= 0 or confidence < min_confidence:
            record["status"] = "unknown_depth"
        else:
            residual = observed - camera[2]
            record["depth_minus_rgb_m"] = float(residual)
            if residual > depth_tolerance_m:
                record["status"] = "depth_behind_rgb_surface"
                farther = True
            elif residual < -depth_tolerance_m:
                record["status"] = "nearer_return_or_geometry_disagreement"
            else:
                record["status"] = "agrees"
                agreeing.append(f)
        report["observations"].append(record)
    separated = any(
        np.linalg.norm(data["world_from_camera"][a, :3, 3] - data["world_from_camera"][b, :3, 3])
        >= policy.min_baseline_m
        for i, a in enumerate(agreeing)
        for b in agreeing[i + 1 :]
    )
    report["depth_supports_landmark"] = bool(separated and not farther)
    report["reason"] = (
        "Depth lies behind the RGB landmark; do not infer empty foreground"
        if farther
        else "Separated depth views agree with the RGB landmark"
        if separated
        else "Insufficient separated depth agreement; retain RGB-only uncertainty"
    )
    return report
