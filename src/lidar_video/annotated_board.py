"""Reviewable grid annotations for occluded boards, including a folding seam.

Annotations are measured image landmarks, not inferred piece identities. Camera
poses may subsequently be refined against the same stationary board landmarks.
All fit thresholds are fixed; failed views remain excluded and reported.
"""

from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from lidar_video.board_calibration import fit_grid, project


def sample_landmarks(data, frame, pixels):
    k = data["K"] if data["K"].ndim == 2 else data["K"][frame]
    pts, used = [], []
    h, w = data["depth"].shape[1:]
    for index, (u, v) in enumerate(pixels):
        x, y = np.rint([u, v]).astype(int)
        if not 1 <= x < w - 1 or not 1 <= y < h - 1:
            continue
        z = data["depth"][frame, y - 1 : y + 2, x - 1 : x + 2]
        level = data["confidence"][frame, y - 1 : y + 2, x - 1 : x + 2]
        good = np.isfinite(z) & (z > 0) & (level >= 1)
        if good.sum() >= 5:
            pts.append(np.median(z[good]) * np.linalg.solve(k, [u, v, 1]))
            used.append(index)
    return np.asarray(pts), np.asarray(used, dtype=int)


def metric_grid(grid, size, gap):
    points = np.asarray(grid, float) * size
    points[:, 1] += np.sign(points[:, 1]) * gap / 2
    return points


def fit_annotated_grid(grid, points, *, center_gap=False):
    grid, points = np.asarray(grid, float), np.asarray(points, float)
    if grid.shape != points.shape or grid.ndim != 2 or grid.shape[1] != 3 or len(grid) < 10:
        raise ValueError("Need at least ten corresponding planar grid and depth landmarks")
    if not np.isfinite(grid).all() or not np.isfinite(points).all() or np.any(grid[:, 2] != 0):
        raise ValueError("Landmarks must be finite with grid z=0")
    if center_gap and (not np.any(grid[:, 1] < 0) or not np.any(grid[:, 1] > 0)):
        raise ValueError("Gap fitting needs landmarks on both halves")
    size, transform = fit_grid(grid, points)
    initial = np.r_[Rotation.from_matrix(transform[:3, :3]).as_rotvec(), transform[:3, 3], size]
    if center_gap:
        initial = np.r_[initial, 0.0]

    def prediction(x):
        local = metric_grid(grid, x[6], x[7] if center_gap else 0)
        return local @ Rotation.from_rotvec(x[:3]).as_matrix().T + x[3:6]

    result = least_squares(lambda x: (prediction(x) - points).ravel(), initial)
    size, gap = float(result.x[6]), float(result.x[7]) if center_gap else 0.0
    residual = np.linalg.norm(prediction(result.x) - points, axis=1)
    rms = float(np.sqrt(np.mean(residual**2)))
    if not result.success or not 0.01 <= size <= 0.2 or not 0 <= gap <= size:
        raise ValueError("Invalid fitted board dimensions")
    if rms > 0.003 or residual.max() > 0.01:
        raise ValueError(f"Annotated grid fails metric fit: RMS {rms:.6f} m")
    transform[:3, :3] = Rotation.from_rotvec(result.x[:3]).as_matrix()
    transform[:3, 3] = result.x[3:6]
    return size, gap, transform, rms


def calibrate_annotations(data, image, annotation):
    frame = annotation["frame"]
    if type(frame) is not int or not 0 <= frame < len(data["depth"]):
        raise ValueError("Annotation frame outside capture")
    if annotation.get("source_frame") != int(data["source_frame"][frame]):
        raise ValueError("Annotation source frame disagrees with capture mapping")
    if image is None:
        raise ValueError("Annotation source image is missing")
    pixels = np.asarray(annotation["pixels"], np.float32)
    grid = np.asarray(annotation["grid"], float)
    if pixels.shape != (len(grid), 2):
        raise ValueError("Grid and pixel annotations disagree")
    if image.shape[:2] != tuple(annotation["image_size_hw"]):
        raise ValueError("Annotation image dimensions disagree")
    refined = cv2.cornerSubPix(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        pixels.reshape(-1, 1, 2).copy(),
        (19, 19),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01),
    ).reshape(-1, 2)
    if np.max(np.linalg.norm(refined - pixels, axis=1)) > 20:
        raise ValueError("Corner refinement moved too far from annotation")
    scale = np.array(data["depth"].shape[1:][::-1]) / np.array(image.shape[:2][::-1])
    corners = refined * scale
    camera, used = sample_landmarks(data, frame, corners)
    pose = data["world_from_camera"][frame]
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    size, gap, transform, rms = fit_annotated_grid(
        grid[used], world, center_gap=annotation.get("center_gap", False)
    )
    k = data["K"] if data["K"].ndim == 2 else data["K"][frame]
    predicted = metric_grid(grid[used], size, gap) @ transform[:3, :3].T + transform[:3, 3]
    pixel_rms = float(np.sqrt(np.mean(np.sum((project(predicted, pose, k) - corners[used]) ** 2, axis=1))))
    if pixel_rms > 1:
        raise ValueError("Annotated grid exceeds one depth-grid pixel RMS")
    if np.dot(transform[:3, 2], pose[:3, 3] - transform[:3, 3]) < 0:
        transform[:3, 1:3] *= -1
        grid[:, 1] *= -1
    return dict(
        square_size_m=size,
        center_gap_m=gap,
        board_from_world=np.linalg.inv(transform).tolist(),
        reference_frame=frame,
        source_frame=int(data["source_frame"][frame]),
        evidence="agent-reviewed image grid annotations + subpixel refinement + metric LiDAR fit",
        orientation=dict(resolved=False, source="provisional annotation axes; no verified a1"),
        corners=corners.tolist(),
        grid=grid.tolist(),
        quality=dict(depth_rms_m=rms, reprojection_rms_depth_px=pixel_rms, landmarks=len(used)),
        limits=[
            "absolute sensor accuracy unverified",
            "planar board with central seam assumed",
            "annotation-assisted",
        ],
    )


def align_views(data, rgb_directory, calibration, *, step=4):
    """Return accepted board-anchored poses and a complete attempt audit.

    SIFT matches a reference band covering the empty center ranks. The band must
    remain unobstructed and the board stationary. No moving-object fusion claim.
    """
    reference = calibration["reference_frame"]
    h, w = data["depth"].shape[1:]
    size, gap = calibration["square_size_m"], calibration.get("center_gap_m", 0)
    world_from_board = np.linalg.inv(calibration["board_from_world"])
    corners = np.asarray(calibration["corners"])
    local = metric_grid(calibration["grid"], size, gap)

    def read(frame):
        path = Path(rgb_directory) / f"{int(data['source_frame'][frame]):06d}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Missing RGB frame {path.name}")
        return cv2.resize(image, (960, 720))

    # Native intrinsics are defined in scaled image coordinates. OpenCV resize
    # additionally shifts pixel centers by half a pixel.
    preview_scale = np.array([960 / w, 720 / h])
    corners_preview = corners * preview_scale - 0.25
    sift = cv2.SIFT_create(nfeatures=5000)
    ref_image = read(reference)
    band = metric_grid([[-4, -2, 0], [4, -2, 0], [4, 2, 0], [-4, 2, 0]], size, gap)
    uv = (
        project(
            band @ world_from_board[:3, :3].T + world_from_board[:3, 3],
            data["world_from_camera"][reference],
            data["K"][reference],
        )
        * preview_scale
        - 0.25
    )
    mask = np.zeros(ref_image.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.rint(uv).astype(int)], 255)
    ref_keys, ref_descriptors = sift.detectAndCompute(ref_image, mask)
    if ref_descriptors is None or len(ref_keys) < 10:
        raise ValueError("Insufficient reference features in board center")
    accepted, audit = [], []
    for frame in range(0, len(data["depth"]), step):
        record = dict(frame=frame, source_frame=int(data["source_frame"][frame]))
        try:
            image = read(frame)
            keys, descriptors = sift.detectAndCompute(image, None)
            if descriptors is None or len(keys) < 2:
                raise ValueError("Insufficient image features")
            pairs = cv2.BFMatcher().knnMatch(ref_descriptors, descriptors, k=2)
            matches = [a for a, b in pairs if a.distance < 0.75 * b.distance]
            if len(matches) < 10:
                raise ValueError("Fewer than ten feature matches")
            a = np.float32([ref_keys[m.queryIdx].pt for m in matches])
            b = np.float32([keys[m.trainIdx].pt for m in matches])
            homography, inliers = cv2.findHomography(a, b, cv2.USAC_MAGSAC, 3)
            if homography is None or inliers.sum() < 10:
                raise ValueError("No supported board homography")
            predicted = cv2.perspectiveTransform(
                corners_preview.astype(np.float32).reshape(-1, 1, 2), homography
            ).reshape(-1, 2)
            inside = np.all((predicted > [12, 12]) & (predicted < [948, 708]), axis=1)
            if inside.sum() < 10:
                raise ValueError("Too few landmarks inside frame")
            refined = cv2.cornerSubPix(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                predicted[inside].copy().reshape(-1, 1, 2),
                (7, 7),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01),
            ).reshape(-1, 2)
            valid = np.linalg.norm(refined - predicted[inside], axis=1) < 5
            pixels = (refined[valid] + 0.25) / preview_scale
            camera, used = sample_landmarks(data, frame, pixels)
            if len(used) < 10:
                raise ValueError("Too few refined landmarks with depth")
            grid = local[inside][valid][used]
            _, camera_from_board = fit_grid(grid, camera, fixed_size=1)
            predicted_camera = grid @ camera_from_board[:3, :3].T + camera_from_board[:3, 3]
            rms = float(np.sqrt(np.mean(np.sum((predicted_camera - camera) ** 2, axis=1))))
            pixels = pixels[used]
            pixel_rms = float(
                np.sqrt(
                    np.mean(
                        np.sum((project(predicted_camera, np.eye(4), data["K"][frame]) - pixels) ** 2, axis=1)
                    )
                )
            )
            if rms > 0.003 or pixel_rms > 1:
                raise ValueError(f"Grid fit failed: {rms:.6f} m / {pixel_rms:.3f} px RMS")
            ok, rotation, translation = cv2.solvePnP(
                grid, pixels, data["K"][frame], None, flags=cv2.SOLVEPNP_ITERATIVE
            )
            pnp = grid @ cv2.Rodrigues(rotation)[0].T + translation.reshape(3)
            depth_rms = float(np.sqrt(np.mean((pnp[:, 2] - camera[:, 2]) ** 2)))
            if not ok or depth_rms > 0.008:
                raise ValueError("Independent image pose disagrees with metric depth")
            pose = world_from_board @ np.linalg.inv(camera_from_board)
            correction = float(np.linalg.norm(pose[:3, 3] - data["world_from_camera"][frame, :3, 3]))
            record.update(
                status="accepted",
                landmarks=len(used),
                depth_rms_m=rms,
                reprojection_rms_depth_px=pixel_rms,
                pnp_depth_rms_m=depth_rms,
                translation_correction_m=correction,
                world_from_camera=pose.tolist(),
            )
            accepted.append(record.copy())
        except (ValueError, cv2.error) as error:
            record.update(status="rejected", reason=str(error))
        audit.append(record)
    return accepted, audit
