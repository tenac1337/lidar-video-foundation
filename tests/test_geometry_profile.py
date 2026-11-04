"""A similarity fit must report world distances, not distances divided by its scale."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

pytest.importorskip("matplotlib")

spec = importlib.util.spec_from_file_location(
    "profile_mesh_lidar", Path(__file__).parents[1] / "scripts/profile_mesh_lidar.py"
)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


@pytest.mark.parametrize("scale", [2 / 3, 1, 1.5])
def test_inverse_query_matches_direct_world_distance(scale):
    rng = np.random.default_rng(24)
    surface = rng.normal(size=(400, 3)) * 0.03
    observed = rng.normal(size=(70, 3)) * 0.06
    center = np.array([0.03, -0.02, 0.05])
    translation = np.array([-0.012, 0.02, 0.015])
    rotvec = np.array([0.3, -0.2, 0.1])
    parameters = np.r_[translation, rotvec, np.log(scale)]
    moved_surface = (surface - center) @ Rotation.from_rotvec(rotvec).as_matrix().T * scale
    moved_surface += center + translation
    expected = cKDTree(moved_surface).query(observed)[0]
    actual = profile.transformed_distances(cKDTree(surface), observed, center, parameters)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
