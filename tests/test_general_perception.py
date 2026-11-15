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


def test_mask_duplicates_suppressed_and_unrelated_instance_survives():
    from lidar_video.general_perception import disambiguate_masks

    a = np.zeros((20, 20), bool)
    a[1:8, 1:8] = True
    b = np.zeros_like(a)
    b[10:18, 10:18] = True
    items = [dict(mask=a, score=0.8), dict(mask=a.copy(), score=0.9), dict(mask=b, score=0.7)]
    result, audit = disambiguate_masks(items)
    assert [r["raw_detection"] for r in result] == [1, 2]
    assert audit["events"][0]["raw_detection"] == 0
    np.testing.assert_array_equal(items[0]["mask"], a)
    np.testing.assert_array_equal(result[1]["mask"], b)


def test_disputed_pixels_unassigned_even_when_nested_detection_rejected():
    from lidar_video.general_perception import disambiguate_masks

    outer = np.zeros((20, 20), bool)
    outer[2:18, 2:18] = True
    inner = np.zeros_like(outer)
    inner[6:9, 6:9] = True
    result, audit = disambiguate_masks([dict(mask=outer, score=0.9), dict(mask=inner, score=0.8)])
    assert len(result) == 1
    assert not (result[0]["mask"] & inner).any()
    assert audit["disputed_pixels"] == 9
    assert any(e["action"] == "rejected ambiguous mask" for e in audit["events"])


def test_mask_overlap_abstention_and_invalid_inputs():
    from lidar_video.general_perception import disambiguate_masks

    a = np.zeros((20, 20), bool)
    b = a.copy()
    a[2:12, 2:12] = True
    b[3:13, 3:13] = True
    result, audit = disambiguate_masks([dict(mask=a, score=0.9), dict(mask=b, score=0.8)])
    assert result == []  # Neither retains 80%; no arbitrary winner.
    assert len(audit["events"]) == 2
    assert disambiguate_masks([])[0] == []
    with pytest.raises(ValueError, match="confidence"):
        disambiguate_masks([dict(mask=a, score=float("nan"))])


def test_interrupted_model_attempt_preserves_terminal_status(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from lidar_video.general_perception import run_development

    def interrupt():
        raise KeyboardInterrupt()

    monkeypatch.setitem(sys.modules, "roboskills.tools.sam3.sam3", SimpleNamespace(Sam=interrupt))
    monkeypatch.setitem(sys.modules, "roboskills.tools.coacd.coacd", SimpleNamespace(decompose=None))
    original_is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda p: str(p) == "/opt/venv/bin/python" or original_is_file(p))
    native = tmp_path / "colab_sam3d.py"
    native.write_text("# stub; never executed")
    native.with_name("sam3d_native_rays.py").write_text("# stub; never executed")
    data, _, _ = fixture()
    capture = tmp_path / "capture.npz"
    np.savez_compressed(capture, **data)
    out = tmp_path / "attempt"
    with pytest.raises(KeyboardInterrupt):
        run_development(capture, out, native, frames=[0, 1])
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "interrupted"
    assert status["error_type"] == "KeyboardInterrupt"
    assert status["elapsed_s"] >= 0
    with pytest.raises(FileExistsError):
        run_development(capture, out, native, frames=[0, 1])


def test_batched_first_hits_preserve_front_surface_and_misses():
    from lidar_video.general_scene import first_hit_ranges

    mesh = trimesh.creation.box(extents=[2, 2, 2])
    mesh.apply_translation([0, 0, 3])
    camera = np.zeros(3)
    rays = np.tile([0.0, 0.0, 1.0], (65, 1))
    rays[-1] = [1.0, 0.0, 0.0]
    actual = mesh.ray.intersects_location
    sizes = []

    def bounded(origins, directions, **kwargs):
        sizes.append(len(directions))
        return actual(origins, directions, **kwargs)

    from unittest.mock import patch

    with patch.object(mesh.ray, "intersects_location", side_effect=bounded):
        ranges = first_hit_ranges(mesh, camera, rays)
    assert sizes == [32, 32, 1]
    np.testing.assert_allclose(ranges[:-1], 2.0)
    assert np.isinf(ranges[-1])
