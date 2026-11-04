"""Ambiguous detections must never be silently assigned to one chess square."""

import numpy as np
import pytest

from lidar_video.gpu_reconstruction import assign_instances


def test_skip_excludes_both_distinct_instances_in_one_square(monkeypatch):
    points = np.zeros((10, 12, 3))
    points[..., 0] = 0.025
    points[..., 1] = 0.025
    points[..., 2] = 0.04
    points[:, 8:, 0] = 0.075
    monkeypatch.setattr("lidar_video.gpu_reconstruction.calibrated_points", lambda *args: points)
    masks = []
    for lo, hi in [(0, 4), (4, 8), (8, 12)]:
        mask = np.zeros((10, 12), bool)
        mask[:, lo:hi] = True
        masks.append(dict(mask=mask, score=0.9))
    data = dict(depth=np.ones((1, 10, 12)), confidence=np.ones((1, 10, 12)))
    cal = dict(square_size_m=0.05)
    with pytest.raises(ValueError, match="Multiple SAM 3"):
        assign_instances(data, cal, 0, masks)
    labels, accepted, audit = assign_instances(data, cal, 0, masks, "skip")
    assert [item["square"] for item in accepted] == ["f5"]
    assert audit["ambiguous_squares"] == ["e5"]
    assert {item["index"] for item in audit["excluded"]} == {0, 1}
    assert not labels[:, :8].any()
    assert (labels[:, 8:] == 1).all()


def test_wrong_registration_still_fails_under_skip(monkeypatch):
    monkeypatch.setattr("lidar_video.gpu_reconstruction.calibrated_points", lambda *args: np.zeros((5, 5, 3)))
    data = dict(depth=np.ones((1, 5, 5)), confidence=np.ones((1, 5, 5)))
    with pytest.raises(ValueError, match="unregistered"):
        assign_instances(data, dict(square_size_m=0.05), 0, [dict(mask=np.ones((2, 2)))], "skip")
