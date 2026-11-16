"""Development RGB stereo surface proposals, never relabelled as sensor depth.

Metric scale comes from the supplied camera poses. Left/right consistency,
texture and disparity uniqueness reject some failures, but repeated patterns,
reflections and pose errors still require independent-view validation.
"""

import cv2
import numpy as np

from lidar_video.capture import validate


def stereo_surface(data, left, right, *, disparities=256, static_scene=False):
    validate(data)
    n, h, w = data["depth"].shape
    if static_scene is not True:
        raise ValueError("Static-scene assumption must be explicit")
    if any(type(f) is not int or not 0 <= f < n for f in (left, right)) or left == right:
        raise ValueError("Two distinct source views required")
    if type(disparities) is not int or disparities < 16 or disparities % 16 or disparities >= w:
        raise ValueError("Disparities must be a positive multiple of 16 below image width")
    source_ids = data.get("source_frame")
    if source_ids is None or source_ids[left] == source_ids[right]:
        raise ValueError("Distinct original source frame IDs required")
    poses = data["world_from_camera"]
    relative = np.linalg.inv(poses[right]) @ poses[left]
    baseline = float(np.linalg.norm(relative[:3, 3]))
    if baseline < 0.005:
        raise ValueError("Insufficient metric camera baseline")
    ks = np.broadcast_to(data["K"], (n, 3, 3))
    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
        ks[left],
        None,
        ks[right],
        None,
        (w, h),
        relative[:3, :3],
        relative[:3, 3],
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=1,
    )
    if abs(p2[1, 3]) > abs(p2[0, 3]):
        raise ValueError("Vertical rectification is not supported by this horizontal stereo proposal")
    if any(roi[2] * roi[3] < 0.02 * w * h for roi in (roi1, roi2)):
        raise ValueError("Insufficient rectified field of view for reliable stereo")
    maps = [
        cv2.initUndistortRectifyMap(k, None, r, p, (w, h), cv2.CV_32FC1)
        for k, r, p in [(ks[left], r1, p1), (ks[right], r2, p2)]
    ]
    colors = [cv2.remap(data["rgb"][f], *m, cv2.INTER_LINEAR) for f, m in zip((left, right), maps)]
    gray = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in colors]
    minimum = 0 if p2[0, 3] < 0 else -disparities

    def match(a, b, low):
        matcher = cv2.StereoSGBM_create(
            minDisparity=low,
            numDisparities=disparities,
            blockSize=5,
            P1=8 * 25,
            P2=32 * 25,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        return matcher.compute(a, b).astype(np.float32) / 16

    dl = match(gray[0], gray[1], minimum)
    right_minimum = -minimum - disparities
    dr = match(gray[1], gray[0], right_minimum)
    yy, xx = np.indices((h, w))
    xr = np.rint(xx - dl).astype(int)
    inside = (xr >= 0) & (xr < w)
    clipped_x = np.clip(xr, 0, w - 1)
    right_disparity = dr[yy, clipped_x]
    valid = inside & (dl > minimum - 1) & (right_disparity > right_minimum - 1)
    valid &= abs(dl + right_disparity) <= 1
    for roi, columns in [(roi1, xx), (roi2, xr)]:
        x, y, rw, rh = roi
        valid &= (columns >= x + 3) & (columns < x + rw - 3) & (yy >= y + 3) & (yy < y + rh - 3)
    for (mx, my), columns in zip(maps, (xx, clipped_x)):
        valid &= (mx[yy, columns] >= 3) & (mx[yy, columns] < w - 3)
        valid &= (my[yy, columns] >= 3) & (my[yy, columns] < h - 3)
    g = gray[0].astype(np.float32)
    variance = cv2.blur(g * g, (7, 7)) - cv2.blur(g, (7, 7)) ** 2
    valid &= variance >= 9
    rectified_points = cv2.reprojectImageTo3D(dl, q)
    camera = rectified_points @ r1
    valid &= np.isfinite(camera).all(axis=2) & (camera[:, :, 2] > 0.1) & (camera[:, :, 2] < 10)
    points = camera[valid]
    uv = points[:, :2] / points[:, 2:] * np.diag(ks[left])[:2] + ks[left][:2, 2]
    pixels = np.rint(uv).astype(int)
    keep = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    pixels, points = pixels[keep], points[keep]
    depth = np.full((h, w), np.inf, np.float32)
    np.minimum.at(depth, (pixels[:, 1], pixels[:, 0]), points[:, 2])
    depth[~np.isfinite(depth)] = 0
    world = points @ poses[left, :3, :3].T + poses[left, :3, 3]
    return dict(
        depth=depth,
        valid=depth > 0,
        world_points=world,
        source_pixels=pixels,
        colors=colors[0][valid][keep],
        disparity=dl,
        rectified_valid=valid,
        metadata=dict(
            schema="rgb-stereo-surface-proposal-v1",
            scope="development; not sensor measurements",
            source_frames=[int(source_ids[f]) for f in (left, right)],
            frames=[left, right],
            baseline_m=baseline,
            disparities=disparities,
            disparity_min=minimum,
            rectification_alpha=1,
            rectified_focal_pixels=float(p1[0, 0]),
            valid_rectified_rois=[list(map(int, roi1)), list(map(int, roi2))],
            left_right_tolerance_pixels=1,
            texture_min_std_gray=3,
            uniqueness_ratio=15,
            inferred_depth_pixels=int(np.count_nonzero(depth)),
            independent_view_validation_passed=False,
            usable_for_native_depth_validation=False,
            limitation="Static scene, supplied camera calibration, texture and Lambertian correspondence assumptions",
        ),
    )
