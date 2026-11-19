"""Foreground-depth disagreement diagnostics and observed open-surface meshes.

RGB corroboration is a caller-supplied evidence mask, not a learned-depth
confidence score. A foreground hypothesis is not certified object ownership.
"""

import numpy as np
import trimesh

from lidar_video.camera_validation import rigid
from lidar_video.capture import camera_points


def depth_ownership(sensor_depth, rgb_depth, rgb_corroborated, *, tolerance_m=0.015):
    """Classify same-camera rays without forcing RGB geometry onto background depth."""
    sensor, rgb = np.asarray(sensor_depth, float), np.asarray(rgb_depth, float)
    corroborated = np.asarray(rgb_corroborated)
    if sensor.ndim != 2 or sensor.shape != rgb.shape or corroborated.shape != rgb.shape:
        raise ValueError("Matching registered nonempty depth grids required")
    if not sensor.size or corroborated.dtype != bool:
        raise ValueError("Explicit boolean RGB corroboration required")
    if not np.isfinite(tolerance_m) or not 0 < tolerance_m <= 0.05:
        raise ValueError("Finite positive depth tolerance <=50mm required")
    if np.any(np.isfinite(sensor) & (sensor < 0)) or np.any(np.isfinite(rgb) & (rgb < 0)):
        raise ValueError("Negative depth is invalid")
    sv = np.isfinite(sensor) & (sensor > 0)
    rv = corroborated & np.isfinite(rgb) & (rgb > 0)
    joint = sv & rv
    delta = np.zeros(rgb.shape)
    np.subtract(sensor, rgb, out=delta, where=joint)
    return dict(
        agreement=joint & (abs(delta) <= tolerance_m),
        possible_background_return=joint & (delta > tolerance_m),
        nearer_sensor_or_mismatch=joint & (delta < -tolerance_m),
        rgb_without_sensor=rv & ~sv,
        sensor_without_rgb=sv & ~rv,
        unknown=~sv & ~rv,
    )


def open_depth_mesh(depth, k, pose, keep, *, stride=2, max_edge_m=0.02, max_depth_jump_m=0.015):
    """Triangulate only fully observed adjacent cells; never bridge missing samples.

    This produces open visible-surface patches. No capping, watertight completion,
    category shape, object identity, collision or physics claim is added.
    """
    depth = np.asarray(depth, float)
    keep = np.asarray(keep)
    k = np.asarray(k, float)
    pose = rigid(pose, "world_from_camera")
    if (
        depth.ndim != 2
        or not depth.size
        or keep.shape != depth.shape
        or keep.dtype != bool
        or k.shape != (3, 3)
        or not np.isfinite(k).all()
        or np.any(np.diag(k)[:2] <= 0)
        or not np.allclose(k[2], [0, 0, 1])
    ):
        raise ValueError("Registered depth, boolean support and pinhole calibration required")
    if (
        type(stride) is not int
        or not 1 <= stride <= 8
        or not np.isfinite([max_edge_m, max_depth_jump_m]).all()
        or not 0 < max_edge_m <= 0.05
        or not 0 < max_depth_jump_m <= 0.05
    ):
        raise ValueError("Bounded surface triangulation policy required")
    valid = keep & np.isfinite(depth) & (depth > 0)
    points = camera_points(np.where(valid, depth, 0), k) @ pose[:3, :3].T + pose[:3, 3]
    h, w = depth.shape
    y, x = np.meshgrid(np.arange(0, h - stride, stride), np.arange(0, w - stride, stride), indexing="ij")
    corners = np.stack(
        [y * w + x, y * w + x + stride, (y + stride) * w + x, (y + stride) * w + x + stride], -1
    ).reshape(-1, 4)
    # The entire source pixel cell must be observed, including samples skipped by stride.
    complete = np.ones(len(corners), bool)
    for dy in range(stride + 1):
        for dx in range(stride + 1):
            complete &= valid[(y + dy).ravel(), (x + dx).ravel()]
    corners = corners[complete]
    faces = np.concatenate([corners[:, [0, 1, 2]], corners[:, [1, 3, 2]]])
    vertices = points.reshape(-1, 3)
    z = depth.ravel()
    if len(faces):
        edge = np.stack(
            [
                np.linalg.norm(vertices[faces[:, a]] - vertices[faces[:, b]], axis=1)
                for a, b in [(0, 1), (1, 2), (2, 0)]
            ],
            1,
        )
        ok = (edge.max(1) <= max_edge_m) & (np.ptp(z[faces], axis=1) <= max_depth_jump_m)
        faces = faces[ok]
    used, inverse = np.unique(faces, return_inverse=True)
    mesh = trimesh.Trimesh(vertices=vertices[used], faces=inverse.reshape(-1, 3), process=False)
    return mesh, dict(
        vertices=len(used),
        faces=len(faces),
        observed_pixels=int(valid.sum()),
        completion=False,
        representation="open observed surface; no object identity or physics claim",
    )
