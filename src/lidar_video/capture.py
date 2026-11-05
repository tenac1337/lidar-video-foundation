"""Validated RGB-D interchange and a deliberately strict Record3D importer.

Canonical camera: OpenCV, x right, y down, z forward. World: right-handed, z up.
Depth means camera-axis Z, NOT Euclidean ray range. RGB/depth share a pixel grid.
"""

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from lidar_video.native_depth import native_intrinsics, native_shape, sample_ids

REQUIRED = {"rgb", "depth", "confidence", "K", "world_from_camera", "rgb_time", "depth_time"}


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate(data, max_skew=0.02):
    missing = REQUIRED - data.keys()
    if missing:
        raise ValueError(f"Missing capture fields: {sorted(missing)}")
    depth = np.asarray(data["depth"])
    if not np.issubdtype(depth.dtype, np.floating):
        raise ValueError("depth must use floating-point metres, not an integer sensor encoding")
    if depth.ndim != 3 or min(depth.shape) < 1:
        raise ValueError("depth must be nonempty (N,H,W) camera-axis Z in metres")
    n, h, w = depth.shape
    if data["rgb"].shape != (n, h, w, 3) or data["rgb"].dtype != np.uint8:
        raise ValueError("rgb must be uint8 (N,H,W,3), registered to depth")
    confidence = data["confidence"]
    if confidence.shape != depth.shape or not np.isin(confidence, [0, 1, 2]).all():
        raise ValueError("confidence must match depth, with levels 0, 1, or 2")
    if np.any(np.isfinite(depth) & (depth < 0)):
        raise ValueError("Negative depth is not valid; use zero/NaN for missing observations")
    k = np.asarray(data["K"])
    if k.shape not in ((3, 3), (n, 3, 3)) or not np.isfinite(k).all():
        raise ValueError("K must be finite (3,3) or (N,3,3)")
    ks = np.broadcast_to(k, (n, 3, 3))
    if not (np.all(ks[:, (0, 1), (0, 1)] > 0) and np.allclose(ks[:, 2], [0, 0, 1])):
        raise ValueError("Invalid pinhole intrinsics")
    if not np.allclose(ks[:, 0, 1], 0) or not np.allclose(ks[:, 1, 0], 0):
        raise ValueError("Skewed intrinsics are not supported")
    for frame in range(n):
        ids = sample_ids(data, frame)
        usable = np.isfinite(depth[frame]) & (depth[frame] > 0) & (confidence[frame] >= 1)
        if np.any(usable & (ids < 0)):
            raise ValueError("Usable depth has no native sample ID")
    native_intrinsics(data)
    transforms = data["world_from_camera"]
    if transforms.shape != (n, 4, 4) or not np.isfinite(transforms).all():
        raise ValueError("world_from_camera must be finite (N,4,4)")
    rot = transforms[:, :3, :3]
    if not (
        np.allclose(transforms[:, 3], [0, 0, 0, 1], atol=1e-5)
        and np.allclose(rot.transpose(0, 2, 1) @ rot, np.eye(3), atol=1e-4)
        and np.allclose(np.linalg.det(rot), 1, atol=1e-4)
    ):
        raise ValueError("Poses must be proper rigid transforms (no reflection or scale)")
    for name in ("rgb_time", "depth_time"):
        t = data[name]
        if t.shape != (n,) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError(f"{name} must contain strictly increasing finite timestamps in seconds")
    skew = np.abs(data["rgb_time"] - data["depth_time"])
    if skew.max() > max_skew:
        raise ValueError(f"RGB/depth timestamp skew {skew.max():.6f}s exceeds {max_skew}s")
    usable = np.isfinite(depth) & (depth > 0) & (confidence >= 1)
    if np.any(usable.sum(axis=(1, 2)) == 0):
        raise ValueError("At least one frame has no usable depth")
    return {
        "frames": n,
        "width": w,
        "height": h,
        "max_timestamp_skew_s": float(skew.max()),
        "valid_depth_fraction": float(usable.mean()),
        "units": "metres",
        "camera_convention": "OpenCV",
        "world_convention": "right-handed z-up",
    }


def load_capture(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    validate(data)
    return data


def camera_points(depth, k):
    v, u = np.indices(depth.shape)
    return np.stack([(u - k[0, 2]) * depth / k[0, 0], (v - k[1, 2]) * depth / k[1, 1], depth], axis=-1)


def world_points(depth, k, transform):
    p = camera_points(depth, k)
    return p @ transform[:3, :3].T + transform[:3, 3]


def normalize(data, min_confidence=1):
    """Rectify changing intrinsics to the first frame. Never interpolate missing depth.

    Both streams use identical maps. Nearest-neighbor depth keeps invalid boundaries
    explicit instead of presenting interpolated gaps as measurements.
    """
    validate(data)
    if min_confidence not in (1, 2):
        raise ValueError("min_confidence must be 1 or 2")
    n, h, w = data["depth"].shape
    ks = np.broadcast_to(data["K"], (n, 3, 3))
    target = ks[0].copy()
    v, u = np.indices((h, w), dtype=np.float32)

    rgb, depths, levels, sample_maps = [], [], [], []
    for i, k in enumerate(ks):
        mx = ((u - target[0, 2]) / target[0, 0] * k[0, 0] + k[0, 2]).astype(np.float32)
        my = ((v - target[1, 2]) / target[1, 1] * k[1, 1] + k[1, 2]).astype(np.float32)
        z = np.array(data["depth"][i], dtype=np.float32, copy=True)
        good = np.isfinite(z) & (z > 0) & (data["confidence"][i] >= min_confidence)
        z[~good] = 0
        depths.append(cv2.remap(z, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT))
        levels.append(
            cv2.remap(
                data["confidence"][i].astype(np.uint8),
                mx,
                my,
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
        )
        rgb.append(cv2.remap(data["rgb"][i], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
        mapped_ids = cv2.remap(
            sample_ids(data, i).astype(np.float64),
            mx,
            my,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1,
        ).astype(np.int64)
        mapped_ids[depths[-1] <= 0] = -1
        sample_maps.append(mapped_ids)
    out = dict(
        data,
        rgb=np.stack(rgb),
        depth=np.stack(depths),
        confidence=np.stack(levels),
        depth_sample_id=np.stack(sample_maps),
        native_K=native_intrinsics(data).copy(),
        native_depth_shape=np.array(native_shape(data)),
        K=target,
    )
    validate(out)
    return out


def import_r3d(path, out):
    """Import fixed-intrinsic Record3D LiDAR archives; fail on unrecognized metadata.

    This reader needs validation against a real export from the user's app version.
    It does NOT fabricate timestamps from a nominal FPS or silently discard varying K.
    """
    import lzfse

    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata"))
        if metadata.get("cameraType") != 1:
            raise ValueError("Expected Record3D cameraType=1 (rear LiDAR)")
        if "perFrameIntrinsicCoeffs" in metadata:
            raise ValueError(
                "Per-frame Record3D intrinsics need an explicit schema adapter; use canonical NPZ"
            )
        names = set(archive.namelist())
        indices = sorted(int(Path(p).stem) for p in names if p.startswith("rgbd/") and p.endswith(".depth"))
        if not indices:
            raise ValueError("Archive has no rgbd/<index>.depth frames")
        if "frameTimestamps" not in metadata:
            raise ValueError("Record3D frameTimestamps missing; nominal FPS is not a replacement")
        timestamps = np.asarray(metadata["frameTimestamps"], dtype=float)
        poses = np.asarray(metadata["poses"], dtype=float)
        if poses.ndim != 2 or poses.shape[1] != 7 or indices[-1] >= min(len(poses), len(timestamps)):
            raise ValueError("Pose/timestamp metadata does not cover the RGB-D frames")
        h, w = int(metadata["dh"]), int(metadata["dw"])
        k = np.array(metadata["K"], dtype=float).reshape(3, 3).T
        # Full RGB pixel grid -> depth grid. Pixel-center convention: x'=(x+.5)*scale-.5.
        sx, sy = w / int(metadata["w"]), h / int(metadata["h"])
        scale = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
        k = scale @ k
        rgb, depth, confidence = [], [], []
        for i in indices:
            prefix = f"rgbd/{i}"
            image = np.array(Image.open(io.BytesIO(archive.read(prefix + ".jpg"))).convert("RGB"))
            if image.shape[:2] != (int(metadata["h"]), int(metadata["w"])):
                raise ValueError(
                    "RGB dimensions disagree with metadata; orientation must be resolved explicitly"
                )
            rgb.append(cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA))
            depth.append(
                np.frombuffer(lzfse.decompress(archive.read(prefix + ".depth")), dtype="<f4").reshape(h, w)
            )
            confidence.append(
                np.frombuffer(lzfse.decompress(archive.read(prefix + ".conf")), dtype=np.uint8).reshape(h, w)
            )
        arkit_from_camera = np.repeat(np.eye(4)[None], len(indices), axis=0)
        q = poses[indices, :4]
        if not np.isfinite(q).all() or not np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-3):
            raise ValueError("Invalid camera-pose quaternions")
        arkit_from_camera[:, :3, :3] = Rotation.from_quat(q).as_matrix()
        arkit_from_camera[:, :3, 3] = poses[indices, 4:]
        zup_from_arkit = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        camera_flip = np.diag([1, -1, -1, 1])
        data = dict(
            rgb=np.stack(rgb),
            depth=np.stack(depth),
            confidence=np.stack(confidence),
            K=k,
            world_from_camera=zup_from_arkit @ arkit_from_camera @ camera_flip,
            rgb_time=timestamps[indices],
            depth_time=timestamps[indices],
        )
    report = validate(data)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    np.savez_compressed(out, **data)
    write_json(
        out.with_suffix(".json"),
        dict(
            report,
            source=str(path),
            source_kind="Record3D",
            timestamp_basis="Record3D associated-frame timestamps; separate sensor exposure times unavailable",
            calibration_status="parser fixture-tested; real-device export unverified",
        ),
    )
    return report
