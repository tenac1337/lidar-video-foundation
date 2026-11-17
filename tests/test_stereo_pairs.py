import numpy as np
import pytest

from lidar_video.stereo_pairs import select_stereo_neighbors


def fixture():
    poses = np.tile(np.eye(4), (6, 1, 1))
    poses[:, 0, 3] = [0, 0.001, 0.06, -0.06, 0.061, 0.5]
    k = np.array([[160.0, 0, 128], [0, 160.0, 64], [0, 0, 1]])
    return poses, k


def test_selects_separated_baselines_and_excludes_reviews_deterministically():
    p, k = fixture()
    a = select_stereo_neighbors(p, k, (256, 128), 0, [5, 4, 3, 2, 1], excluded=[4])
    b = select_stereo_neighbors(p, k, (256, 128), 0, [1, 2, 3, 4, 5], excluded=[4])
    assert a == b and set(a["neighbors"]) == {2, 3}
    assert a["complete"]
    assert not next(r for r in a["candidates"] if r["frame"] == 1)["eligible"]
    assert not next(r for r in a["candidates"] if r["frame"] == 5)["eligible"]


def test_vertical_motion_and_missing_second_partner():
    p, k = fixture()
    p[:, 1, 3] = p[:, 0, 3]
    p[:, 0, 3] = 0
    assert set(select_stereo_neighbors(p, k, (256, 128), 0, [2, 3])["neighbors"]) == {2, 3}
    result = select_stereo_neighbors(p, k, (256, 128), 0, [1, 2, 4])
    assert not result["complete"] and len(result["neighbors"]) == 1


def test_invalid_camera_policy_and_excluded_anchor_fail():
    p, k = fixture()
    for kwargs in [dict(excluded=[0]), dict(min_baseline_m=0.3), dict(max_rotation_degrees=np.nan)]:
        with pytest.raises(ValueError):
            select_stereo_neighbors(p, k, (256, 128), 0, [2, 3], **kwargs)
    p[2, 0, 0] = 2
    with pytest.raises(ValueError):
        select_stereo_neighbors(p, k, (256, 128), 0, [2, 3])
