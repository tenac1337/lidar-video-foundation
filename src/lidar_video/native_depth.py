"""Native sensor rays retained separately from the registered RGB pixel grid.

Only origin-aligned legacy inputs without IDs permit deriving native intrinsics.
Explicit IDs can have undergone arbitrary remapping, so their source K is required.
This module uses NumPy only, including in the isolated SAM 3D environment.
"""

import numpy as np


def native_shape(data):
    shape = np.asarray(data.get("native_depth_shape", data["depth"].shape[1:]))
    if shape.shape != (2,) or shape.dtype.kind not in "iu" or np.any(shape <= 0):
        raise ValueError("native_depth_shape must contain two positive integers")
    return tuple(map(int, shape))


def sample_ids(data, frame):
    """Preserve sensor sample identity through the project's nearest-neighbor mapping.

    Legacy native_depth_shape means the project's origin-aligned mapping:
    native pixel = round(model pixel / scale). Other resamplers must supply IDs.
    """
    h, w = data["depth"].shape[1:]
    nh, nw = native_shape(data)
    if "depth_sample_id" in data:
        ids = np.asarray(data["depth_sample_id"])
        if ids.shape != data["depth"].shape or ids.dtype.kind not in "iu":
            raise ValueError("depth_sample_id must be integer (N,H,W)")
        if np.any(ids[frame] < -1) or np.any(ids[frame] >= nh * nw):
            raise ValueError("depth_sample_id must be -1 or a native-grid index")
        return ids[frame]
    if nh > h or nw > w:
        raise ValueError("Invalid native_depth_shape; supply explicit depth_sample_id for other mappings")
    y, x = np.indices((h, w))
    return (np.clip(np.rint(y * nh / h), 0, nh - 1) * nw + np.clip(np.rint(x * nw / w), 0, nw - 1)).astype(
        np.int32
    )


def native_intrinsics(data):
    n, h, w = data["depth"].shape
    nh, nw = native_shape(data)
    if "native_K" in data:
        k = np.asarray(data["native_K"], dtype=float)
    elif "depth_sample_id" in data:
        raise ValueError(
            "Explicit depth_sample_id requires original native_K; restore it from the source capture"
        )
    else:
        # The documented legacy mapping has no pixel-center offset.
        k = np.diag([nw / w, nh / h, 1]) @ np.asarray(data["K"])
    if k.shape not in ((3, 3), (n, 3, 3)) or not np.isfinite(k).all():
        raise ValueError("native_K must be finite (3,3) or (N,3,3)")
    ks = np.broadcast_to(k, (n, 3, 3))
    if (
        not np.all(ks[:, (0, 1), (0, 1)] > 0)
        or not np.allclose(ks[:, 2], [0, 0, 1])
        or not np.allclose(ks[:, 0, 1], 0)
        or not np.allclose(ks[:, 1, 0], 0)
    ):
        raise ValueError("native_K must contain unskewed pinhole intrinsics")
    return ks


def sensor_camera_points(data, frame):
    """Backproject each stored depth using its ORIGINAL sample's camera ray.

    Duplicate stored pixels now produce identical XYZ, rather than a fan of rays.
    Invalid IDs produce NaN and must not become usable geometric evidence.
    """
    ids = sample_ids(data, frame)
    k = native_intrinsics(data)[frame]
    _, nw = native_shape(data)
    v, u = ids // nw, ids % nw
    z = data["depth"][frame]
    points = np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=-1)
    points[ids < 0] = np.nan
    return points
