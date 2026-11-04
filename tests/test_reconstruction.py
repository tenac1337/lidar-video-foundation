"""Geometry, coordinate and shared-output contracts independent of neural inference."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

from lidar_video.capture import load_capture
from lidar_video.reconstruction import collect_squares, import_learned_mesh, reconstruct, rigid
from lidar_video.scenic_export import validate_sample

DATA = Path(__file__).resolve().parent / "fixtures/chess-demo"


def test_calibration_rejects_scale_and_reflection():
    for matrix in (np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1])):
        with pytest.raises(ValueError, match="proper rigid"):
            rigid(matrix, "calibration")


def test_mesh_pose_is_composed_and_scale_is_not_applied_twice(tmp_path):
    mesh = trimesh.creation.box(extents=[0.02, 0.02, 0.04])
    mesh.export(tmp_path / "mesh.ply")
    board_from_world = np.eye(4)
    board_from_world[:3, :3] = Rotation.from_euler("z", 0.4).as_matrix()
    board_from_world[:3, 3] = [0.1, -0.2, 0.03]
    world_from_camera = np.eye(4)
    world_from_camera[:3, :3] = Rotation.from_euler("x", 0.6).as_matrix()
    world_from_camera[:3, 3] = [-0.3, 0.1, 0.5]
    board_from_object = np.eye(4)
    board_from_object[:3, 3] = [0.0225, 0.0225, 0.02]
    pose = np.linalg.inv(board_from_world @ world_from_camera) @ board_from_object
    (tmp_path / "pose.json").write_text(json.dumps({"camera_from_object": pose.tolist(), "scale": [9, 9, 9]}))
    observed, _ = trimesh.sample.sample_surface(mesh, 100, seed=3)
    observed += board_from_object[:3, 3]
    data = dict(depth=np.ones((1, 2, 2)), world_from_camera=[world_from_camera])
    calibration = dict(board_from_world=board_from_world.tolist(), square_size_m=0.045)
    imported, record = import_learned_mesh(
        dict(mesh="mesh.ply", pose="pose.json"), tmp_path, data, calibration, observed
    )
    np.testing.assert_allclose(imported.extents, [0.02, 0.02, 0.04], atol=1e-7)
    np.testing.assert_allclose(imported.bounds.mean(0), [0.0225, 0.0225, 0.02], atol=1e-7)
    assert record["scale_reapplied"] is False
    wrong = copy.deepcopy(data)
    wrong["world_from_camera"] = [np.eye(4)]
    with pytest.raises(ValueError, match="depth agreement"):
        import_learned_mesh(dict(mesh="mesh.ply", pose="pose.json"), tmp_path, wrong, calibration, observed)


def test_registered_masks_are_required(tmp_path):
    data = load_capture(DATA / "capture.npz")
    calibration = json.loads((DATA / "board_calibration.json").read_text())
    Image.new("L", (4, 4), 1).save(tmp_path / "00000.png")
    with pytest.raises(ValueError, match="registered"):
        collect_squares(data, calibration, [0], tmp_path)
    Image.fromarray(np.zeros(data["depth"].shape[1:], np.uint8)).save(tmp_path / "00000.png")
    points, _, _ = collect_squares(data, calibration, [0], tmp_path)
    assert sum(map(len, points.values())) == 0


def test_quality_fallback_is_explicit_and_preserves_rejection_evidence(tmp_path):
    trimesh.creation.box(extents=[0.02, 0.02, 0.04]).export(tmp_path / "bad.ply")
    pose = np.eye(4)
    pose[:3, 3] = [5, 5, 5]
    (tmp_path / "pose.json").write_text(json.dumps(dict(camera_from_object=pose.tolist())))
    manifest = tmp_path / "meshes.json"
    manifest.write_text(
        json.dumps(
            dict(
                schema="roboskills-chess-meshes-v1",
                objects={"a1": dict(mesh="bad.ply", pose="pose.json", frame=0)},
            )
        )
    )
    kwargs = dict(mesh_manifest=manifest, static_scene=True, samples=2, simulation=False)
    with pytest.raises(ValueError, match="depth agreement"):
        reconstruct(DATA / "capture.npz", DATA / "board_calibration.json", tmp_path / "strict", **kwargs)
    report = reconstruct(
        DATA / "capture.npz",
        DATA / "board_calibration.json",
        tmp_path / "fallback",
        mesh_policy="fallback",
        **kwargs,
    )
    assert report["imported_learned_meshes"] == 0
    assert report["geometric_completions"] == 32
    assert "depth agreement" in report["rejected_learned_meshes"]["a1"]
    graph = json.loads((tmp_path / "fallback/scene_graph.json").read_text())
    piece = next(n for n in graph["nodes"] if n["id"] == "piece_a1")
    assert "rejected_learned_mesh" in piece["evidence"]
    # Invalid coordinate contracts must still fail, even under fallback policy.
    pose[0, 0] = 2
    (tmp_path / "pose.json").write_text(json.dumps(dict(camera_from_object=pose.tolist())))
    with pytest.raises(ValueError, match="proper rigid"):
        reconstruct(
            DATA / "capture.npz",
            DATA / "board_calibration.json",
            tmp_path / "invalid",
            mesh_policy="fallback",
            **kwargs,
        )


def test_rgbd_to_shared_scene_and_scenic_without_fixture_labels(tmp_path):
    out = tmp_path / "reconstruction"
    report = reconstruct(
        DATA / "capture.npz",
        DATA / "board_calibration.json",
        out,
        static_scene=True,
        samples=3,
        simulation=False,
    )
    assert report["objects"] == 32
    assert report["imported_learned_meshes"] == 0
    graph = json.loads((out / "scene_graph.json").read_text())
    assert len(graph["nodes"]) == 97
    for node in graph["nodes"]:
        if node["kind"] != "object":
            continue
        assert node["symbol"] == "?"  # No truth or chess starting-position prior supplied.
        mesh = trimesh.load(out / node["mesh_path"], force="mesh")
        np.testing.assert_allclose(mesh.extents, node["size_m"], atol=1e-7)
        np.testing.assert_allclose(mesh.bounds.mean(0), 0, atol=1e-7)
    variants = json.loads((out / "scenic_samples.json").read_text())["samples"]
    for variant in variants:
        validate_sample(variant)
        assert {n["id"] for n in graph["nodes"]} == {n["id"] for n in variant["nodes"]}
        for before, after in zip(graph["nodes"], variant["nodes"]):
            if before["kind"] == "object":
                assert before["mesh_path"] == after["mesh_path"]
                assert before["square"] == after["square"]
                np.testing.assert_allclose(before["size_m"], after["size_m"])
    # A malformed externally changed sample cannot silently violate occupancy.
    bad = copy.deepcopy(variants[0])
    piece = next(n for n in bad["nodes"] if n["kind"] == "object")
    piece["position_m"][0] += 0.045
    with pytest.raises(ValueError):
        validate_sample(bad)
