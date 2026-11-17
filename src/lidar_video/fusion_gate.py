"""Conservative cross-modal ray gate; disagreement is not averaged away.

Inputs share pinhole coordinates after native-ray registration. Unknown evidence
is retained with its modality, not promoted to cross-modal confirmation. This
gate does not establish correspondence, calibration, or surface completeness.
"""

import numpy as np


def gate_depths(lidar, rgb, *, tolerance_m=0.015):
    """Return per-modality masks, retaining single-modality and agreeing samples.

    Zero/NaN means unknown. Sensor confidence, native sampling, RGB validation,
    and registration must be handled by the caller. Values are never changed.
    Both modalities abstain on disagreement, regardless of which is nearer.
    """
    lidar, rgb = np.asarray(lidar), np.asarray(rgb)
    if lidar.shape != rgb.shape or lidar.ndim != 2 or not lidar.size:
        raise ValueError("Matching nonempty registered depth grids required")
    if not np.isfinite(tolerance_m) or not 0 < tolerance_m <= 0.05:
        raise ValueError("Positive finite tolerance <=50 mm required")
    if np.any(np.isfinite(lidar) & (lidar < 0)) or np.any(np.isfinite(rgb) & (rgb < 0)):
        raise ValueError("Negative depth is invalid; use zero/NaN for unknown")
    lvalid = np.isfinite(lidar) & (lidar > 0)
    rvalid = np.isfinite(rgb) & (rgb > 0)
    joint = lvalid & rvalid
    residual = np.zeros(lidar.shape, float)
    np.subtract(lidar, rgb, out=residual, where=joint)
    conflict = joint & (abs(residual) > tolerance_m)
    agreement = joint & ~conflict
    return dict(
        lidar_keep=lvalid & ~conflict,
        rgb_keep=rvalid & ~conflict,
        agreement=agreement,
        conflict=conflict,
        lidar_only=lvalid & ~rvalid,
        rgb_only=rvalid & ~lvalid,
    )
