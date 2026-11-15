import json

import numpy as np
import pytest
import trimesh
from test_general_scene import fixture

from lidar_video.capture import normalize, write_json
from lidar_video.general_perception import model_transport, world_prior


def test_irregular_frame_transport_preserves_sensor_times_and_native_rays(tmp_path):
    data, _, _ = fixture()
    data["rgb_time"] = np.array([0.0, 0.48, 1.07])
    data["depth_time"] = data["rgb_time"].copy()
    data = normalize(data)
    out = model_transport(data, tmp_path / "transport")
    meta = np.load(out / "sensor_metadata.npz")
    np.testing.assert_array_equal(meta["rgb_time"], data["rgb_time"])
    np.testing.assert_array_equal(meta["source_frame"], [11, 43, 72])
    depth = np.load(out / "depth.npz")
    np.testing.assert_array_equal(depth["depth"], data["depth"])
    np.testing.assert_array_equal(depth["depth_sample_id"], data["depth_sample_id"])
    assert not json.loads((out / "transport.json").read_text())["temporal_inference_allowed"]


def test_pose_is_applied_once_and_reported_scale_never_doubled(tmp_path):
    data, _, _ = fixture()
    mesh = trimesh.creation.box(extents=[0.08, 0.04, 0.12])
    mesh.export(tmp_path / "mesh.ply")
    pose = np.eye(4)
    pose[:3, 3] = [0.1, 0.05, 0.5]
    write_json(tmp_path / "pose.json", dict(camera_from_object=pose.tolist(), scale=13.0))

    def fake_decompose(path, out, threshold, faces, hulls):
        assert (threshold, faces, hulls) == (0.05, 20000, 32)
        trimesh.load(path).export(out / "part.obj")
        write_json(out / "parts.json", dict(parts=["part.obj"]))

    entry = world_prior(tmp_path, data, 1, fake_decompose)
    result = trimesh.load(tmp_path / "world" / entry["visual"])
    expected = mesh.copy()
    expected.apply_transform(data["world_from_camera"][1] @ pose)
    np.testing.assert_allclose(result.bounds, expected.bounds, atol=1e-7)
    assert entry["provenance"]["source_frame"] == 43
    assert not entry["provenance"]["extra_scale_applied"]


def test_nonrigid_model_pose_rejected(tmp_path):
    data, _, _ = fixture()
    pose = np.diag([2.0, 2, 2, 1])
    write_json(tmp_path / "pose.json", dict(camera_from_object=pose.tolist()))
    with pytest.raises(ValueError, match="rigid"):
        world_prior(tmp_path, data, 0, None)
