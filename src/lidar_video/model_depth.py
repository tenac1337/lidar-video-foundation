"""Object evidence filtering without discarding SAM 3D's scene-scale context."""

import numpy as np


def object_conditioning_depth(archive, frame, mask, trusted):
    """Keep valid measured context; restrict only the target mask to trusted depth.

    The pinned SAM 3D ObjectCentricSSI normalizer uses scene-wide depth to
    estimate scale, while its center and RoboSkills' score use the target mask.
    Removing all context silently changes that scene statistic to an object
    statistic. Missing/invalid source samples are never made valid here.
    """
    frames = np.asarray(archive["frames"])
    rows = np.flatnonzero(frames == frame)
    if len(rows) != 1:
        raise ValueError("Object frame must uniquely match depth archive frames")
    row = int(rows[0])
    depth, valid = np.asarray(archive["depth"]), np.asarray(archive["valid"])
    mask, trusted = np.asarray(mask), np.asarray(trusted)
    if depth.ndim != 3 or valid.shape != depth.shape or valid.dtype != bool:
        raise ValueError("Depth archive requires depth and boolean valid arrays of shape (N,H,W)")
    if len(frames) != len(depth) or mask.shape != depth.shape[1:] or trusted.shape != mask.shape:
        raise ValueError("Object masks must be registered to the depth archive")
    if mask.dtype != bool or trusted.dtype != bool or not mask.any():
        raise ValueError("Object masks must be boolean and the target mask nonempty")
    if np.any(trusted & ~mask):
        raise ValueError("Trusted object depth must lie inside its instance mask")
    usable = valid[row] & np.isfinite(depth[row]) & (depth[row] > 0)
    if not np.any(usable & trusted):
        raise ValueError("Object has no valid trusted depth")
    result = {key: np.array(value, copy=True) for key, value in archive.items()}
    result["valid"][row] = usable & (~mask | trusted)
    result["depth"][row] = np.where(result["valid"][row], depth[row], 0)
    return result
