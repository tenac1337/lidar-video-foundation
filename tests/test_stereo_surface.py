import copy

import cv2
import numpy as np
import pytest

from lidar_video.stereo_surface import stereo_surface


def textured_plane():
    # Independent analytic stereo pair: 0.08 m baseline, f=160 px, Z=0.8 m => 16 px disparity.
    rng = np.random.default_rng(8)
    h, w = 128, 256
    left = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-16] = left[:, 16:]
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[1, 0, 3] = 0.08
    return dict(
        rgb=np.stack([left, right]),
        depth=np.ones((2, h, w), np.float32) * 2,
        confidence=np.ones((2, h, w), np.uint8),
        K=np.array([[160.0, 0, 128], [0, 160, 64], [0, 0, 1]]),
        world_from_camera=poses,
        rgb_time=np.array([0.0, 1.0]),
        depth_time=np.array([0.0, 1.0]),
        source_frame=np.array([12, 48]),
    )


def test_metric_stereo_ignores_wrong_lidar_and_preserves_source_capture():
    data = textured_plane()
    original = copy.deepcopy(data)
    result = stereo_surface(data, 0, 1, disparities=32, static_scene=True)
    core = result["depth"][15:-15, 50:-20]
    assert np.mean(core > 0) > 0.9
    assert np.median(abs(core[core > 0] - 0.8)) < 0.005
    assert not result["metadata"]["usable_for_native_depth_validation"]
    assert result["metadata"]["source_frames"] == [12, 48]
    for key in original:
        np.testing.assert_array_equal(original[key], data[key])


def test_reverse_camera_order_recovers_negative_disparity():
    data = textured_plane()
    result = stereo_surface(data, 1, 0, disparities=32, static_scene=True)
    assert result["metadata"]["disparity_min"] == -32
    assert np.count_nonzero(result["depth"]) > 10000
    assert np.median(abs(result["depth"][result["valid"]] - 0.8)) < 0.005


def test_rgb_path_does_not_require_or_consult_lidar_fields():
    data = textured_plane()
    baseline = stereo_surface(data, 0, 1, disparities=32, static_scene=True)
    for key in ["depth", "confidence", "depth_time"]:
        del data[key]
    independent = stereo_surface(data, 0, 1, disparities=32, static_scene=True)
    np.testing.assert_array_equal(baseline["depth"], independent["depth"])


def test_vertical_motion_rotation_preserves_world_geometry_and_source_pixels():
    data = textured_plane()
    data["rgb"] = np.rot90(data["rgb"], 1, axes=(1, 2)).copy()
    data["K"] = np.array([[160.0, 0, 64], [0, 160.0, 127], [0, 0, 1]])
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    data["world_from_camera"][:, :3, :3] = rotation
    result = stereo_surface(data, 0, 1, disparities=32, static_scene=True, rectification_alpha=0)
    assert result["depth"].shape == (256, 128)
    assert result["metadata"]["vertical_image_rotation_degrees"] == 90
    assert len(result["world_points"]) > 10000
    assert np.median(abs(result["world_points"][:, 2] - 0.8)) < 0.005
    points = result["world_points"] @ rotation
    uv = points @ data["K"].T
    uv = np.rint(uv[:, :2] / uv[:, 2:]).astype(int)
    np.testing.assert_array_equal(uv, result["source_pixels"])


def test_textureless_or_repeated_view_never_fills_surface():
    data = textured_plane()
    data["rgb"][:] = 127
    result = stereo_surface(data, 0, 1, disparities=32, static_scene=True)
    assert not result["valid"].any()
    with pytest.raises(ValueError, match="Static"):
        stereo_surface(data, 0, 1, disparities=32)
    with pytest.raises(ValueError, match="distinct"):
        stereo_surface(data, 0, 0, disparities=32, static_scene=True)
    data["source_frame"][1] = data["source_frame"][0]
    with pytest.raises(ValueError, match="Distinct original"):
        stereo_surface(data, 0, 1, disparities=32, static_scene=True)


def test_rotated_camera_recovers_metric_plane_after_rectification():
    data = textured_plane()
    angle = np.deg2rad(3)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    data["world_from_camera"][1, :3, :3] = rotation
    yy, xx = np.indices((128, 256))
    rays = np.stack([(xx - 128) / 160, (yy - 64) / 160, np.ones_like(xx)], axis=-1) @ rotation.T
    world = rays * (0.8 / rays[..., 2, None]) + np.array([0.08, 0, 0])
    # Independently render the right image by intersecting rays with the plane
    # and sampling the left camera's texture; no rectification/Q code is reused.
    uv = world[..., :2] / 0.8 * 160 + np.array([128, 64])
    data["rgb"][1] = cv2.remap(
        data["rgb"][0], uv[..., 0].astype("float32"), uv[..., 1].astype("float32"), cv2.INTER_LINEAR
    )
    result = stereo_surface(data, 0, 1, disparities=32, static_scene=True)
    assert np.count_nonzero(result["valid"]) > 5000
    assert np.median(abs(result["depth"][result["valid"]] - 0.8)) < 0.01
