import numpy as np
import pytest

from lidar_video.fusion_gate import gate_depths


def test_conflicts_abstain_in_both_directions_unknowns_preserve_provenance():
    lidar = np.array([[1.0, 2.0, 0.5, 0.0, 1.0, np.nan]])
    rgb = np.array([[1.005, 1.0, 1.0, 0.8, 0.0, np.nan]])
    before = lidar.copy(), rgb.copy()
    r = gate_depths(lidar, rgb)
    np.testing.assert_array_equal(r["agreement"], [[True, False, False, False, False, False]])
    np.testing.assert_array_equal(r["conflict"], [[False, True, True, False, False, False]])
    np.testing.assert_array_equal(r["lidar_keep"], [[True, False, False, False, True, False]])
    np.testing.assert_array_equal(r["rgb_keep"], [[True, False, False, True, False, False]])
    assert r["lidar_only"].sum() == r["rgb_only"].sum() == 1
    np.testing.assert_array_equal(lidar, before[0])
    np.testing.assert_array_equal(rgb, before[1])


def test_invalid_inputs_cannot_silently_change_gate_policy():
    with pytest.raises(ValueError):
        gate_depths(np.zeros((2, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        gate_depths(np.array([[-1.0]]), np.ones((1, 1)))
    for tolerance in [0, np.nan, 0.1]:
        with pytest.raises(ValueError):
            gate_depths(np.ones((1, 1)), np.ones((1, 1)), tolerance_m=tolerance)
