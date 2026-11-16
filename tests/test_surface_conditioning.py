import copy

import numpy as np
import pytest
from test_general_scene import fixture

from lidar_video.general_scene import InstancePolicy, observation
from lidar_video.model_depth import object_conditioning_depth
from lidar_video.surface_conditioning import choose_conditioning_reference, corroborated_conditioning


def observations(data, detections):
    result = []
    for frame, items in detections.items():
        obs = observation(data, frame, items[0]["mask"], InstancePolicy())
        obs["source_frame"] = int(data["source_frame"][frame])
        result.append(obs)
    return result


def test_clean_metric_surface_survives_without_mutating_validation_evidence():
    data, detections, _ = fixture()
    group = observations(data, detections)
    before = copy.deepcopy(group)
    chosen, report = choose_conditioning_reference(data, dict(observations=group))
    assert report["passed"]
    assert all(r["retained_native_samples"] > 300 for r in report["observations"])
    assert not any(r["object_surface_confirmed"] for r in report["observations"])
    for original, obs in zip(before, group):
        for key in ("cloud", "trusted", "mask"):
            np.testing.assert_array_equal(original[key], obs[key])
    assert any(chosen["cloud"] is obs["cloud"] for obs in group)


def test_visible_depth_error_is_removed_from_model_input_not_validation_cloud():
    data, detections, _ = fixture()
    # One patch floats 5 cm ahead of the real object in the source view.
    mask = detections[0][0]["mask"]
    yy, xx = np.where(mask)
    patch = np.zeros_like(mask)
    patch[
        int(np.median(yy)) - 2 : int(np.median(yy)) + 3, int(np.median(xx)) - 2 : int(np.median(xx)) + 3
    ] = True
    data["depth"][0][patch] -= 0.05
    group = observations(data, detections)
    original_cloud = group[0]["cloud"].copy()
    trusted, report = corroborated_conditioning(data, group[0], group)
    assert report["contradicted_native_samples"] >= patch.sum()
    assert not trusted[patch].any()
    assert trusted.sum() > 100
    np.testing.assert_array_equal(original_cloud, group[0]["cloud"])
    archive = dict(frames=np.arange(3), depth=data["depth"].copy(), valid=data["depth"] > 0)
    conditioned = object_conditioning_depth(archive, 0, mask, trusted)
    assert not conditioned["valid"][0][patch].any()
    np.testing.assert_array_equal(conditioned["depth"][0][~mask], data["depth"][0][~mask])


def test_occlusion_missing_depth_and_repeated_camera_never_count_as_support():
    data, detections, _ = fixture()
    group = observations(data, detections)
    for bad in ("occluded", "missing", "repeat"):
        changed = copy.deepcopy(data)
        if bad == "occluded":
            changed["depth"][1:] = 0.1
        elif bad == "missing":
            changed["confidence"][1:] = 0
        else:
            changed["world_from_camera"][:] = changed["world_from_camera"][0]
        trusted, report = corroborated_conditioning(changed, group[0], group)
        assert not trusted.any(), bad
        assert report["retained_native_samples"] == 0


def test_visible_silhouette_contradiction_vetoes_other_view_agreement():
    data, detections, _ = fixture()
    group = observations(data, detections)
    group[2]["mask"][:] = False
    trusted, report = corroborated_conditioning(data, group[0], group)
    assert report["witnesses"][0]["agreeing_native_samples"] > 100
    assert report["contradicted_native_samples"] > 100
    assert not trusted.any()


def test_thin_surface_has_no_class_or_support_height_cutoff():
    data, detections, _ = fixture()
    # A 2 mm surface is retained if other views actually measure that surface.
    data["depth"][:] = 0.798
    group = observations(data, detections)
    trusted, report = corroborated_conditioning(data, group[1], group)
    assert trusted.sum() > 100
    assert report["retained_native_samples"] > 100


def test_no_reference_fallback_and_invalid_settings_fail():
    data, detections, _ = fixture()
    group = observations(data, detections)
    chosen, report = choose_conditioning_reference(data, dict(observations=group[:1]))
    assert chosen is None and not report["passed"]
    with pytest.raises(ValueError, match="tolerance"):
        corroborated_conditioning(data, group[0], group, depth_tolerance_m=np.nan)
    with pytest.raises(ValueError, match="one witness"):
        corroborated_conditioning(data, group[0], [group[1], group[1]])


def test_upsampling_does_not_inflate_native_evidence():
    data, detections, _ = fixture()
    group = observations(data, detections)
    _, base = corroborated_conditioning(data, group[1], group)
    h, w = data["depth"].shape[1:]
    up = copy.deepcopy(data)
    up["native_depth_shape"] = np.array([h, w])
    up["native_K"] = data["K"].copy()
    up["K"][:2] *= 2
    for key in ("rgb", "depth", "confidence"):
        up[key] = np.repeat(np.repeat(data[key], 2, axis=1), 2, axis=2)
    ids = np.broadcast_to(np.arange(h * w).reshape(h, w), data["depth"].shape)
    up["depth_sample_id"] = np.repeat(np.repeat(ids, 2, axis=1), 2, axis=2)
    up_group = copy.deepcopy(group)
    for obs in up_group:
        for key in ("mask", "trusted"):
            obs[key] = np.repeat(np.repeat(obs[key], 2, axis=0), 2, axis=1)
    _, report = corroborated_conditioning(up, up_group[1], up_group)
    assert report["input_native_samples"] == base["input_native_samples"]
    assert report["retained_native_samples"] == base["retained_native_samples"]
