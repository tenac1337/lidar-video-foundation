import copy

import numpy as np
import pytest
from test_landmark_tracks import fixture

from lidar_video.rgb_depth_audit import audit_landmark_depth


def inputs():
    views, pixels, _, _ = fixture()
    labels = np.stack([v["labels"] for v in views])
    data = dict(
        rgb=np.zeros((*labels.shape, 3), np.uint8),
        depth=np.full(labels.shape, 0.8, np.float32),
        confidence=np.full(labels.shape, 2, np.uint8),
        K=views[0]["K"],
        world_from_camera=np.stack([v["world_from_camera"] for v in views]),
        rgb_time=np.arange(3, dtype=float),
        depth_time=np.arange(3, dtype=float),
    )
    observations = [dict(view=i, pixel=pixels[i][0].tolist(), label=1) for i in range(3)]
    return data, observations, labels


def test_metric_rgb_surface_and_depth_agree_without_modifying_inputs():
    data, obs, labels = inputs()
    before = copy.deepcopy(data)
    r = audit_landmark_depth(data, obs, labels, static_scene=True)
    assert r["rgb_geometry"]["passed"] and r["depth_supports_landmark"]
    assert {v["status"] for v in r["observations"]} == {"agrees"}
    assert not r["foreground_carving_authorized"]
    for k in data:
        np.testing.assert_array_equal(data[k], before[k])


def test_background_plane_agreement_is_not_object_surface_support():
    data, obs, labels = inputs()
    # A high-confidence plane behind the visible RGB feature is coherent in
    # every depth view, but must not erase that foreground feature.
    data["depth"][:] = 1.0
    r = audit_landmark_depth(data, obs, labels, static_scene=True)
    assert r["rgb_geometry"]["passed"] and not r["depth_supports_landmark"]
    assert all(v["status"] == "depth_behind_rgb_surface" for v in r["observations"])
    assert not r["foreground_carving_authorized"]


def test_one_background_return_vetoes_two_agreeing_views():
    data, obs, labels = inputs()
    data["depth"][2] = 0.9
    r = audit_landmark_depth(data, obs, labels, static_scene=True)
    assert not r["depth_supports_landmark"]
    assert sum(v["status"] == "agrees" for v in r["observations"]) == 2


@pytest.mark.parametrize("kind", ["nearer", "missing", "low_confidence", "boundary"])
def test_unknown_or_occluded_depth_does_not_supply_positive_support(kind):
    data, obs, labels = inputs()
    if kind == "nearer":
        data["depth"][:] = 0.6
    elif kind == "missing":
        for o in obs:
            x, y = np.rint(o["pixel"]).astype(int)
            data["depth"][o["view"], y, x] = 0
    elif kind == "low_confidence":
        data["confidence"][:] = 1
    else:
        labels[:] = 0
        for o in obs:
            x, y = np.rint(o["pixel"]).astype(int)
            labels[o["view"], y, x] = 1
    r = audit_landmark_depth(data, obs, labels, static_scene=True)
    assert not r["depth_supports_landmark"]
    assert not any(v["status"] == "agrees" for v in r["observations"])


def test_depth_cannot_rescue_inconsistent_rgb_triangulation():
    data, obs, labels = inputs()
    obs[2]["pixel"][1] += 15
    r = audit_landmark_depth(data, obs, labels, static_scene=True)
    assert not r["rgb_geometry"]["passed"] and not r["depth_supports_landmark"]
    assert r["observations"] == []


def test_undeclared_static_scene_bad_pixels_and_wrong_masks_fail():
    data, obs, labels = inputs()
    with pytest.raises(ValueError, match="static scene"):
        audit_landmark_depth(data, obs, labels)
    with pytest.raises(ValueError, match="three unique"):
        audit_landmark_depth(data, obs[:2], labels, static_scene=True)
    bad = copy.deepcopy(obs)
    bad[0]["pixel"] = [-1, 30]
    with pytest.raises(ValueError, match="in-image"):
        audit_landmark_depth(data, bad, labels, static_scene=True)
    labels[:] = 2
    with pytest.raises(ValueError, match="declared mask"):
        audit_landmark_depth(data, obs, labels, static_scene=True)
