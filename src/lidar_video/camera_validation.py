"""Small camera-frame validators without reconstruction or simulator imports."""

import numpy as np


def rigid(value, name):
    """Require a finite proper rigid transform, preserving calibrated metric scale."""
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError(f"{name} must be a finite proper rigid transform")
    return matrix
