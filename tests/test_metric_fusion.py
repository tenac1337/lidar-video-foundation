import numpy as np
import pytest

from lidar_video.metric_fusion import (
    fit_metric_scale,
    sample_registered_depth,
    smooth_support,
    view_consistency,
)


def test_scale_recovery_with_outliers_and_disjoint_validation():
    y, x = np.indices((96, 128))
    truth = 0.5 + 0.003 * x + 0.002 * y
    predicted = truth * 1.8
    measured = truth.copy()
    measured[(x % 7 == 0) & (y % 3 == 0)] += 0.2
    support = np.ones_like(truth, bool)
    result = fit_metric_scale(predicted, measured, support)
    assert result["accepted"] and result["scale"] == pytest.approx(1 / 1.8)
    changed = measured.copy()
    changed[((x // 8 + y // 8) % 2) == 1] += 0.2
    rejected = fit_metric_scale(predicted, changed, support)
    assert rejected["candidate_scale"] == result["candidate_scale"]
    assert not rejected["accepted"] and rejected["scale"] is None


def test_scale_rejects_wrong_shape_and_concentrated_support():
    y, x = np.indices((96, 128))
    truth = np.ones_like(x, float)
    pred = np.where(x < 64, 1.0, 2.0)
    assert not fit_metric_scale(pred, truth, truth > 0)["accepted"]
    support = (x < 30) & (y < 30)
    assert not fit_metric_scale(truth, truth, support)["accepted"]
    for kwargs in [dict(scale_bounds=(1, 1)), dict(min_samples=0), dict(min_agreement=np.nan)]:
        with pytest.raises(ValueError):
            fit_metric_scale(truth, truth, truth > 0, **kwargs)


def test_registration_does_not_copy_border_depth_into_unknown_rays():
    k = np.array([[10.0, 0, 2], [0, 10, 2], [0, 0, 1]])
    shifted = k.copy()
    shifted[0, 2] += 2
    out = sample_registered_depth(np.ones((5, 5)), k, shifted, (5, 5))
    assert np.all(out[:, :2] == 0) and np.all(out[:, 2:] == 1)


def test_support_excludes_edges_missing_neighbors_and_nonfinite_values():
    z = np.ones((10, 10))
    z[:, 5:] = 2
    z[2, 2] = np.nan
    support = smooth_support(z, np.ones_like(z, bool))
    assert not support[1:4, 1:4].any()
    assert not support[:, 4:6].any()
    assert support[6, 2] and support[6, 7]
    assert not support[0].any()


def test_view_gate_distinguishes_occlusion_contradiction_and_unknown():
    z = np.ones((16, 20))
    k = np.array([[20.0, 0, 10], [0, 20, 8], [0, 0, 1]])
    pose = np.eye(4)
    left = pose.copy()
    left[0, 3] = -0.05
    right = pose.copy()
    right[0, 3] = 0.05
    witnesses = [("left", z, k, left), ("right", z, k, right)]
    result = view_consistency(z, k, pose, witnesses)
    assert result["keep"][8, 10] and result["support"][8, 10] == 2
    assert not result["keep"][:, 0].any()
    occluder = ("occluder", z * 0.7, k, right)
    assert view_consistency(z, k, pose, [*witnesses, occluder])["keep"][8, 10]
    farther = ("farther", z * 1.2, k, right)
    assert not view_consistency(z, k, pose, [*witnesses, farther])["keep"][8, 10]
    missing = ("missing", z * 0, k, right)
    assert not view_consistency(z, k, pose, [witnesses[0], missing])["keep"][8, 10]
    with pytest.raises(ValueError):
        view_consistency(z, k, pose, [witnesses[0], witnesses[0]])
    assert not view_consistency(z, k, pose, [("self", z, k, pose)])["keep"].any()


def test_depth_and_transform_validation():
    z = np.ones((10, 10))
    k = np.eye(3)
    pose = np.eye(4)
    pose[2, 2] = 2
    with pytest.raises(ValueError):
        view_consistency(z, k, pose, [])
    with pytest.raises(ValueError):
        sample_registered_depth(-z, k, k, z.shape)


def test_rotated_camera_uses_camera_axis_depth_not_world_z():
    z = np.ones((48, 64))
    k = np.array([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]])
    theta = 0.2
    pose = np.eye(4)
    witness = np.eye(4)
    witness[:3, :3] = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    witness[0, 3] = 0.05
    y, x = np.indices(z.shape)
    # Intersection with world plane Z=1, analytically expressed in witness camera Z.
    depth = 1 / (np.cos(theta) - np.sin(theta) * (x - 32) / 60)
    result = view_consistency(z, k, pose, [("rotated", depth, k, witness)], min_support=1)
    assert result["keep"][16:32, 24:40].all()
    wrong = view_consistency(z, k, pose, [("wrong_world_z", z, k, witness)], min_support=1)
    assert not wrong["keep"][24, 24]
