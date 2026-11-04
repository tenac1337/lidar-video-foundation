"""Stray Scanner directory import; preserve frame association and sensor times.

Format: https://github.com/strayrobots/scanner/blob/main/docs/format.md
OdometryEncoder already rotates ARKit cameras into OpenCV coordinates. Only the
world's y-up to z-up rotation is applied here. FFmpeg ignores MP4 edit lists so a
trimmed leading sample cannot silently shift every RGB/depth association.
"""

import csv
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lidar_video.capture import validate, write_json


def read_odometry(path):
    with Path(path).open() as file:
        rows = list(csv.DictReader(file, skipinitialspace=True))
    fields = "timestamp frame x y z qx qy qz qw fx fy cx cy".split()
    try:
        values = {name: np.array([float(row[name]) for row in rows]) for name in fields}
    except (KeyError, ValueError) as error:
        raise ValueError("Expected Stray Scanner odometry with per-frame intrinsics") from error
    if not rows or any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("Empty or non-finite odometry")
    if not np.array_equal(values["frame"], np.arange(len(rows))):
        raise ValueError("Frame IDs must be consecutive from zero; cannot infer missing RGB association")
    if np.any(np.diff(values["timestamp"]) <= 0):
        raise ValueError("Sensor timestamps must increase")
    if any(
        row.get(name, "").strip() for row in rows for name in ("distortion_center_x", "distortion_center_y")
    ):
        raise ValueError("Lens distortion metadata needs an explicit rectification adapter")
    q = np.stack([values[name] for name in ("qx", "qy", "qz", "qw")], axis=1)
    if not np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-3):
        raise ValueError("Invalid camera quaternion")
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(q).as_matrix()
    poses[:, :3, 3] = np.stack([values[name] for name in ("x", "y", "z")], axis=1)
    zup = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    k = np.tile(np.eye(3), (len(rows), 1, 1))
    for name, a, b in (("fx", 0, 0), ("fy", 1, 1), ("cx", 0, 2), ("cy", 1, 2)):
        k[:, a, b] = values[name]
    return values["timestamp"], zup @ poses, k


def uniform_selection(times, fps):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive")
    target = times[0] + np.arange(int(np.floor((times[-1] - times[0]) * fps)) + 1) / fps
    right = np.clip(np.searchsorted(times, target), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(times[left] - target) < abs(times[right] - target), left, right)
    if np.any(np.diff(indices) <= 0):
        raise ValueError("Requested sampling rate repeats source frames; choose a lower FPS")
    return indices, target


def import_stray(source, out, *, fps=2.0):
    source, out = Path(source), Path(out)
    times, poses, intrinsics = read_odometry(source / "odometry.csv")
    if (source / "distortion").exists():
        raise ValueError("Distortion lookup tables are not yet supported")
    expected = {f"{i:06d}.png" for i in range(len(times))}
    for folder in ("depth", "confidence"):
        if {p.name for p in (source / folder).glob("*.png")} != expected:
            raise ValueError(f"{folder} frame IDs disagree with odometry")
    indices, targets = uniform_selection(times, fps)
    out.mkdir(parents=True, exist_ok=False)
    (out / "rgb").mkdir()
    rgb, depths, levels, ks = [], [], [], []
    wanted = set(indices.tolist())
    decoded = 0
    with tempfile.TemporaryDirectory(prefix="stray-import-") as temporary:
        video = Path(temporary) / "untrimmed.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ignore_editlist",
                "1",
                "-i",
                str(source / "rgb.mp4"),
                "-map",
                "0:v:0",
                "-c:v",
                "copy",
                "-an",
                str(video),
            ],
            check=True,
        )
        reader = cv2.VideoCapture(str(video))
        while True:
            ok, image = reader.read()
            if not ok:
                break
            frame = decoded
            decoded += 1
            if frame not in wanted:
                continue
            depth = cv2.imread(str(source / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
            confidence = cv2.imread(str(source / "confidence" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
            if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
                raise ValueError(f"Invalid uint16 millimetre depth at {frame}")
            if confidence is None or confidence.shape != depth.shape or confidence.dtype != np.uint8:
                raise ValueError(f"Invalid confidence map at {frame}")
            h, w = depth.shape
            ih, iw = image.shape[:2]
            if not np.isclose(iw / ih, w / h):
                raise ValueError("RGB/depth aspect ratios differ; registration is ambiguous")
            sx, sy = w / iw, h / ih
            # ARKit sceneDepth uses scaled native image coordinates, not a new
            # optical sensor. Sample RGB at those exact coordinate locations.
            scale = np.diag([sx, sy, 1.0])
            yy, xx = np.indices((h, w), dtype=np.float32)
            registered = cv2.remap(image, xx / sx, yy / sy, cv2.INTER_LINEAR)
            rgb.append(cv2.cvtColor(registered, cv2.COLOR_BGR2RGB))
            depths.append(depth.astype(np.float32) / 1000)
            levels.append(confidence)
            ks.append(scale @ intrinsics[frame])
            cv2.imwrite(str(out / "rgb" / f"{frame:06d}.jpg"), image)
        reader.release()
    if decoded != len(times) or len(rgb) != len(indices):
        raise ValueError(f"Decoded {decoded} video frames, expected {len(times)}; refusing shifted pairing")
    data = dict(
        rgb=np.stack(rgb),
        depth=np.stack(depths),
        confidence=np.stack(levels),
        K=np.stack(ks),
        world_from_camera=poses[indices],
        rgb_time=targets,
        depth_time=targets,
        source_frame=indices,
        source_time=times[indices],
        source_kind=np.array("real_iphone_stray"),
    )
    report = validate(data)
    np.savez_compressed(out / "capture.npz", **data)
    report.update(
        source=str(source),
        source_format="Stray Scanner per-frame-intrinsics PNG export",
        source_frames=len(times),
        decoded_video_frames=decoded,
        source_duration_s=float(times[-1] - times[0]),
        sampling="nearest complete RGB-D-pose tuple on explicit regular timeline; no pose/depth interpolation",
        output_fps=fps,
        max_time_displacement_s=float(np.max(abs(times[indices] - targets))),
        source_frame=indices.tolist(),
        source_time_s=times[indices].tolist(),
        independent_depth_timestamps=False,
        timing_limit="Depth association is supplied by the app; hardware RGB/depth skew is not measured",
        video_edit_list_ignored=True,
        per_frame_intrinsics=True,
        camera_axes="already OpenCV in Stray odometry; no second camera flip",
        geometric_accuracy="not established without measured ground truth",
    )
    write_json(out / "import_report.json", report)
    return report
