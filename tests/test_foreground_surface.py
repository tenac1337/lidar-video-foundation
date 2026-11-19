import numpy as np
import pytest

from lidar_video.foreground_surface import depth_ownership, open_depth_mesh


def test_background_is_not_used_as_foreground_agreement():
    sensor = np.array([[1.0, 1.0, 0.7, 0.0, 1.0]])
    rgb = np.array([[0.8, 1.0, 1.0, 0.8, 0.8]])
    support = np.array([[True, True, True, True, False]])
    r = depth_ownership(sensor, rgb, support)
    assert r["possible_background_return"].tolist() == [[True, False, False, False, False]]
    assert r["agreement"].sum() == 1 and r["nearer_sensor_or_mismatch"].sum() == 1
    assert r["rgb_without_sensor"].sum() == 1 and r["sensor_without_rgb"].sum() == 1
    with pytest.raises(ValueError):
        depth_ownership(sensor, rgb, support.astype(int))


def test_open_mesh_does_not_bridge_missing_interior_or_depth_jump():
    z = np.ones((5, 5))
    k = np.array([[100.0, 0, 2], [0, 100.0, 2], [0, 0, 1]])
    full, report = open_depth_mesh(z, k, np.eye(4), z > 0, stride=2, max_edge_m=0.05)
    assert len(full.faces) == 8 and not full.is_watertight and not report["completion"]
    keep = z > 0
    keep[1, 1] = False
    cut, _ = open_depth_mesh(z, k, np.eye(4), keep, stride=2, max_edge_m=0.05)
    assert len(cut.faces) == 6
    z[:, 2:] = 1.2
    discontinuous, _ = open_depth_mesh(z, k, np.eye(4), z > 0, stride=2, max_edge_m=0.05)
    assert len(discontinuous.faces) == 4
    empty, _ = open_depth_mesh(z, k, np.eye(4), np.zeros_like(z, bool))
    assert len(empty.faces) == len(empty.vertices) == 0
