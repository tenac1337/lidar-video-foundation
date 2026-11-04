"""Detect an 8x8 board and estimate its metric frame from registered RGB-D.

Intrinsics and camera poses must already be supplied. This estimates board pose,
not camera intrinsics. A symmetric board cannot reveal the physical a1 corner.
"""

from pathlib import Path

import cv2
import numpy as np

from lidar_video.capture import validate, write_json


def fit_grid(grid, world, fixed_size=None):
    """Proper similarity fit, with scale estimated from depth unless fixed."""
    a, b = grid - grid.mean(0), world - world.mean(0)
    if np.linalg.matrix_rank(a, tol=1e-8) < 2:
        raise ValueError("Collinear grid samples")
    u, singular, vt = np.linalg.svd(a.T @ b)
    sign = np.diag([1, 1, np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ sign @ u.T
    size = float(np.sum(singular * np.diag(sign)) / np.sum(a * a))
    if fixed_size is not None:
        size = float(fixed_size)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = world.mean(0) - rotation @ (size * grid.mean(0))
    return size, transform


def project(points, world_from_camera, k):
    camera = (np.asarray(points) - world_from_camera[:3, 3]) @ world_from_camera[:3, :3]
    if np.any(camera[:, 2] <= 0):
        raise ValueError("Board projects behind the camera")
    return camera[:, :2] / camera[:, 2:] * np.diag(k)[:2] + k[:2, 2]


def detect_frame(data, frame, square_size=None):
    gray = cv2.cvtColor(data["rgb"][frame], cv2.COLOR_RGB2GRAY)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, (7, 7), flags=flags)
    if not found:
        raise ValueError("Full 7x7 inner-corner grid not visible; use an unobstructed board frame")
    corners = corners.reshape(-1, 2).astype(float)
    grid = np.array([[x, y, 0] for y in range(-3, 4) for x in range(-3, 4)], float)
    z, confidence = data["depth"][frame], data["confidence"][frame]
    k = data["K"] if data["K"].ndim == 2 else data["K"][frame]
    pose = data["world_from_camera"][frame]
    points, used = [], []
    for i, (u, v) in enumerate(corners):
        x, y = np.rint([u, v]).astype(int)
        patch = z[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]
        levels = confidence[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]
        good = np.isfinite(patch) & (patch > 0) & (levels >= 1)
        if good.sum() < 5:
            continue
        depth = float(np.median(patch[good]))
        camera = depth * np.array([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], 1])
        points.append(pose[:3, :3] @ camera + pose[:3, 3])
        used.append(i)
    if len(used) < 35:
        raise ValueError(f"Only {len(used)}/49 grid corners have usable depth; need at least 35")
    points, used = np.asarray(points), np.asarray(used)
    grid = grid[used]
    # RANSAC excludes depth measured on a piece or a depth-discontinuity outlier.
    best = np.zeros(len(points), bool)
    rng = np.random.default_rng(17)
    for _ in range(128):
        subset = rng.choice(len(points), 3, replace=False)
        try:
            size, world_from_board = fit_grid(grid[subset], points[subset])
        except ValueError:
            continue
        predicted = size * grid @ world_from_board[:3, :3].T + world_from_board[:3, 3]
        inliers = np.linalg.norm(predicted - points, axis=1) <= 0.003
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 35:
        raise ValueError(f"Only {int(best.sum())}/49 corners agree with a metric square grid within 3 mm")
    estimated_size, world_from_board = fit_grid(grid[best], points[best])
    if not 0.01 <= estimated_size <= 0.2:
        raise ValueError("Estimated square size outside supported 1–20 cm range")
    if square_size is not None:
        if not np.isfinite(square_size) or square_size <= 0:
            raise ValueError("Supplied square size must be finite and positive")
        if abs(estimated_size / square_size - 1) > 0.05:
            raise ValueError(
                "Measured square size disagrees with supplied size by more than 5%; check depth units"
            )
    size, world_from_board = fit_grid(grid[best], points[best], square_size)
    predicted = size * grid @ world_from_board[:3, :3].T + world_from_board[:3, 3]
    residual = np.linalg.norm(predicted - points, axis=1)
    reprojection = np.linalg.norm(project(predicted, pose, k) - corners[used], axis=1)
    rms = float(np.sqrt(np.mean(residual[best] ** 2)))
    pixel_rms = float(np.sqrt(np.mean(reprojection[best] ** 2)))
    if rms > 0.003 or pixel_rms > 1.5:
        raise ValueError(f"Grid fit failed: depth RMS {rms:.4f} m, reprojection RMS {pixel_rms:.2f} px")
    # The top normal points to the observing camera, fixing reflection ambiguity.
    if np.dot(world_from_board[:3, 2], pose[:3, 3] - world_from_board[:3, 3]) < 0:
        world_from_board[:3, 1:3] *= -1
    return dict(
        frame=int(frame),
        square_size_m=size,
        estimated_square_size_m=estimated_size,
        world_from_board=world_from_board,
        corners=corners,
        valid_depth_corners=len(used),
        inlier_corners=int(best.sum()),
        depth_rms_m=rms,
        reprojection_rms_px=pixel_rms,
    )


def orient_from_a1(world_from_board, pixel, pose, k, square_size):
    """An annotation anywhere inside physical a1 fixes the remaining quarter turn."""
    pixel = np.asarray(pixel, float)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError("a1 pixel must be a finite [u, v]")
    ray = pose[:3, :3] @ np.linalg.solve(k, [*pixel, 1])
    normal, center = world_from_board[:3, 2], world_from_board[:3, 3]
    denominator = np.dot(normal, ray)
    if abs(denominator) < 1e-8:
        raise ValueError("a1 ray is parallel to board")
    distance = np.dot(normal, center - pose[:3, 3]) / denominator
    if distance <= 0:
        raise ValueError("a1 ray does not intersect the board in front of the camera")
    world_point = pose[:3, 3] + distance * ray
    for turn in range(4):
        angle = turn * np.pi / 2
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        candidate = world_from_board.copy()
        candidate[:3, :3] = world_from_board[:3, :3] @ rotation
        local = (world_point - center) @ candidate[:3, :3] / square_size
        if np.all((local[:2] > -4) & (local[:2] < -3)):
            return candidate
    raise ValueError("a1 annotation must lie inside a corner square, away from its edges")


def calibrate(data, out=None, *, frame=None, a1_pixel=None, square_size=None, max_frames=12):
    validate(data)
    if frame is not None and (type(frame) is not int or not 0 <= frame < len(data["depth"])):
        raise ValueError("Calibration frame outside capture")
    if a1_pixel is not None and frame is None:
        raise ValueError("--a1-pixel requires --frame so its image is unambiguous")
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    frames = (
        [frame]
        if frame is not None
        else np.unique(
            np.linspace(0, len(data["depth"]) - 1, min(max_frames, len(data["depth"]))).astype(int)
        ).tolist()
    )
    successes, failures = [], []
    for index in frames:
        try:
            successes.append(detect_frame(data, index, square_size))
        except ValueError as error:
            failures.append(dict(frame=int(index), reason=str(error)))
    if not successes:
        raise ValueError(
            "No usable board calibration: "
            + "; ".join(f"frame {f['frame']}: {f['reason']}" for f in failures)
        )
    chosen = successes[0]
    transform = chosen["world_from_board"].copy()
    # Camera poses must place the board consistently in a single world frame.
    checks = []
    for other in successes[1:]:
        center_error = float(np.linalg.norm(transform[:3, 3] - other["world_from_board"][:3, 3]))
        normal_angle = float(
            np.degrees(np.arccos(np.clip(np.dot(transform[:3, 2], other["world_from_board"][:3, 2]), -1, 1)))
        )
        scale_error = abs(other["square_size_m"] / chosen["square_size_m"] - 1)
        if center_error > 0.01 or normal_angle > 5 or scale_error > 0.05:
            raise ValueError("Board is inconsistent across camera poses; check pose drift or board motion")
        checks.append(
            dict(
                frame=other["frame"],
                center_difference_m=center_error,
                normal_difference_deg=normal_angle,
                relative_scale_difference=scale_error,
            )
        )
    index = chosen["frame"]
    k = data["K"] if data["K"].ndim == 2 else data["K"][index]
    if a1_pixel is not None:
        transform = orient_from_a1(
            transform, a1_pixel, data["world_from_camera"][index], k, chosen["square_size_m"]
        )
    result = dict(
        square_size_m=chosen["square_size_m"],
        board_from_world=np.linalg.inv(transform).tolist(),
        origin="detected board center at top surface",
        axes="x toward assigned h-file, y toward assigned rank 8, z toward observing camera",
        evidence="OpenCV 7x7 inner corners + registered metric depth; no supplied board pose",
        orientation=dict(
            resolved=a1_pixel is not None,
            source="supplied a1 pixel"
            if a1_pixel is not None
            else "provisional detector axes; fourfold ambiguity",
            a1_pixel=None if a1_pixel is None else list(map(float, a1_pixel)),
        ),
        reference_frame=index,
        quality={
            k: chosen[k]
            for k in (
                "valid_depth_corners",
                "inlier_corners",
                "depth_rms_m",
                "reprojection_rms_px",
                "estimated_square_size_m",
            )
        },
        checked_frames=[s["frame"] for s in successes],
        rejected_frames=failures,
        cross_view_checks=checks,
        limits=[
            "one planar 8x8 board",
            "camera intrinsics and poses are supplied",
            "no real iPhone validation",
            "board stationary across checked views",
        ],
    )
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "board_calibration.json", result)
        overlay = cv2.cvtColor(data["rgb"][index], cv2.COLOR_RGB2BGR)
        cv2.drawChessboardCorners(
            overlay, (7, 7), chosen["corners"].astype(np.float32).reshape(-1, 1, 2), True
        )
        local = (
            np.array([[-4, -4, 0], [4, -4, 0], [4, 4, 0], [-4, 4, 0], [-3.5, -3.5, 0]])
            * chosen["square_size_m"]
        )
        uv = (
            project(local @ transform[:3, :3].T + transform[:3, 3], data["world_from_camera"][index], k)
            .round()
            .astype(int)
        )
        cv2.polylines(overlay, [uv[:4]], True, (0, 255, 0), 2)
        cv2.putText(
            overlay,
            "a1" if a1_pixel is not None else "a1?",
            tuple(uv[4]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
        cv2.imwrite(str(out / "overlay.png"), overlay)
    return result
