import json

import numpy as np
import pytest
from PIL import Image

from lidar_video import reconstruction_run as runner
from lidar_video.creation_audit import read_json, save
from lidar_video.scenic_export import export_scenic, sample_scenic


@pytest.fixture
def recipe(tmp_path):
    capture = tmp_path / "capture.npz"
    np.savez_compressed(
        capture, rgb=np.zeros((2, 4, 4, 3), np.uint8), depth=np.ones((2, 4, 4), np.float32),
        confidence=np.full((2, 4, 4), 2, np.uint8), K=np.array([[10., 0, 2], [0, 10, 2], [0, 0, 1]]),
        world_from_camera=np.tile(np.eye(4), (2, 1, 1)), rgb_time=np.array([0., 1.]),
        depth_time=np.array([0., 1.]), source_frame=np.array([101, 205]),
    )
    save(tmp_path / "calibration.json", dict(board_from_world=np.eye(4).tolist(), square_size_m=.05,
                                           reference_frame=0, source_frame=101))
    masks = tmp_path / "masks"
    masks.mkdir()
    for i in range(2):
        Image.fromarray(np.ones((4, 4), np.uint8)).save(masks / f"{i:05d}.png")
    save(masks / "instances.json", dict(schema="chess-instance-assignments-v1",
                                       frames={"0": {"1": "a1"}, "1": {"1": "a1"}}))
    config = dict(schema="reconstruction-run-v1", scene_id="test", evaluation_scope="development",
                  capture="capture.npz", calibration="calibration.json", masks="masks", mesh_manifest=None,
                  expected_objects=["a1"], fusion_source_frames=[205, 101], reference_source_frame=101,
                  static_scene=True, mesh_policy="strict", min_height_m=.006, samples=2, seed=13,
                  simulation=False)
    path = tmp_path / "recipe.json"
    save(path, config)
    return path


def emit_scene(capture, calibration, out, **kwargs):
    assert kwargs["fusion_frames"] == [1, 0]
    assert kwargs["reference_frame"] == 0
    assert kwargs["samples"] == 2 and kwargs["seed"] == 13
    assert kwargs["simulation"] is False and kwargs["mesh_policy"] == "strict"
    out.mkdir()
    graph = dict(units="metres", nodes=[
        dict(id="board", kind="support", position_m=[0, 0, -.01], size_m=[.4, .4, .02]),
        dict(id="piece_a1", square="a1", kind="object", position_m=[-.06, 0, .021],
             size_m=[.03, .03, .04]),
    ], edges=[dict(source="piece_a1", relation="on", target="board")])
    save(out / "scene_graph.json", graph)
    export_scenic(graph, out / "scenario.scenic", jitter=.001)
    sample_scenic(out / "scenario.scenic", graph, out / "scenic_samples.json",
                  count=kwargs["samples"], seed=kwargs["seed"])
    return dict(status="passed", objects=1)


def test_frozen_run_preserves_source_selection_and_checks_actual_scenic(recipe, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "reconstruct", emit_scene)
    out = tmp_path / "run"
    receipt = runner.freeze_reconstruction(recipe, out)
    assert receipt["selection"] == dict(fusion_frames=[1, 0], reference_frame=0)
    with pytest.raises(FileExistsError):
        runner.freeze_reconstruction(recipe, out)
    result = runner.run_reconstruction(out, expected_lock_sha256=receipt["lock_sha256"])
    assert result["status"] == "completed"
    assert all(r["decision"] == "accept" for r in result["comparison"]["decisions"])
    assert (out / "comparison/consistency/replayed_scenic_samples.json").exists()
    with pytest.raises(FileExistsError):
        runner.run_reconstruction(out, expected_lock_sha256=receipt["lock_sha256"])


@pytest.mark.parametrize("change", ["mask", "assignments", "new_mask", "config", "code", "runtime", "lock"])
def test_changed_inputs_are_blocked_before_reconstruction(recipe, tmp_path, monkeypatch, change):
    out = tmp_path / "run"
    receipt = runner.freeze_reconstruction(recipe, out)
    def forbidden(*args, **kwargs):
        pytest.fail("Reconstruction must not start for changed inputs")
    monkeypatch.setattr(runner, "reconstruct", forbidden)
    if change == "mask":
        Image.fromarray(np.zeros((4, 4), np.uint8)).save(tmp_path / "masks/00000.png")
    elif change == "new_mask":
        (tmp_path / "masks/extra.txt").write_text("new input")
    elif change in ("config", "assignments", "lock"):
        path = {"config": recipe, "assignments": tmp_path / "masks/instances.json",
                "lock": out / "lock.json"}[change]
        path.write_text(path.read_text() + "\n")
    elif change == "code":
        monkeypatch.setattr(runner, "policy_hashes", lambda: {"changed": "code"})
    else:
        monkeypatch.setattr(runner, "_runtime", lambda: {"changed": "runtime"})
    with pytest.raises(ValueError, match="changed|hash mismatch"):
        runner.run_reconstruction(out, expected_lock_sha256=receipt["lock_sha256"])
    assert not (out / "execution.json").exists()


@pytest.mark.parametrize("change", ["missing_assignments", "unexpected_object", "missing_label",
                                    "wrong_shape", "source_index", "duplicate_sources", "calibration",
                                    "unknown_option", "scope"])
def test_preflight_rejects_bad_recipe_without_creating_run(recipe, tmp_path, change):
    config = read_json(recipe)
    assignments = tmp_path / "masks/instances.json"
    if change == "missing_assignments":
        assignments.unlink()
    elif change == "unexpected_object":
        data = read_json(assignments)
        data["frames"]["0"]["1"] = "b1"
        save(assignments, data)
    elif change == "missing_label":
        Image.fromarray(np.zeros((4, 4), np.uint8)).save(tmp_path / "masks/00000.png")
    elif change == "wrong_shape":
        Image.fromarray(np.ones((3, 4), np.uint8)).save(tmp_path / "masks/00000.png")
    elif change == "source_index":
        config.update(fusion_source_frames=[0, 1], reference_source_frame=0)
    elif change == "duplicate_sources":
        config["fusion_source_frames"] = [101, 101]
    elif change == "calibration":
        cal = read_json(tmp_path / "calibration.json")
        cal["source_frame"] = 999
        save(tmp_path / "calibration.json", cal)
    elif change == "unknown_option":
        config["typo_samples"] = 3
    else:
        config["evaluation_scope"] = "confirmatory"
    save(recipe, config)
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.freeze_reconstruction(recipe, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_output_cannot_be_inside_mask_inventory(recipe, tmp_path):
    with pytest.raises(ValueError, match="separate"):
        runner.freeze_reconstruction(recipe, tmp_path / "masks/run")
    assert not (tmp_path / "masks/run").exists()


def test_input_change_during_reconstruction_records_failure(recipe, tmp_path, monkeypatch):
    out = tmp_path / "run"
    receipt = runner.freeze_reconstruction(recipe, out)
    def mutate(*args, **kwargs):
        result = emit_scene(*args, **kwargs)
        recipe.write_text(recipe.read_text() + "\n")
        return result
    monkeypatch.setattr(runner, "reconstruct", mutate)
    with pytest.raises(ValueError, match="partial outputs preserved"):
        runner.run_reconstruction(out, expected_lock_sha256=receipt["lock_sha256"])
    assert read_json(out / "result.json")["status"] == "failed"
    assert not (out / "comparison").exists()


def test_reconstruction_inventory_mismatch_cannot_report_completion(recipe, tmp_path, monkeypatch):
    out = tmp_path / "run"
    receipt = runner.freeze_reconstruction(recipe, out)
    def missing(*args, **kwargs):
        result = emit_scene(*args, **kwargs)
        graph = read_json(out / "scene/scene_graph.json")
        graph["nodes"].pop()
        save(out / "scene/scene_graph.json", graph)
        return result
    monkeypatch.setattr(runner, "reconstruct", missing)
    result = runner.run_reconstruction(out, expected_lock_sha256=receipt["lock_sha256"])
    assert result["status"] == "rejected"
    assert result["comparison"]["decisions"][1]["decision"] == "reject"


def test_candidate_pose_and_observation_files_are_locked(recipe, tmp_path, monkeypatch):
    import trimesh

    trimesh.creation.box(extents=[.02, .02, .04]).export(tmp_path / "mesh.ply")
    save(tmp_path / "pose.json", dict(camera_from_object=np.eye(4).tolist()))
    Image.fromarray(np.ones((4, 4), np.uint8)).save(tmp_path / "observed.png")
    save(tmp_path / "meshes.json", dict(schema="roboskills-chess-meshes-v1", objects={
        "a1": dict(mesh="mesh.ply", pose="pose.json", frame=0,
                   observations=[dict(frame=1, mask="observed.png", label=1)]),
    }))
    config = read_json(recipe)
    config["mesh_manifest"] = "meshes.json"
    save(recipe, config)
    out = tmp_path / "run"
    receipt = runner.freeze_reconstruction(recipe, out)
    lock = read_json(out / "lock.json")
    for name in ("mesh.ply", "pose.json", "observed.png"):
        assert str(tmp_path / name) in lock["inputs"]
    (tmp_path / "observed.png").unlink()
    with pytest.raises(ValueError, match="missing"):
        runner.run_reconstruction(out, expected_lock_sha256=receipt["lock_sha256"])


def test_cli_freeze_returns_pinnable_receipt(recipe, tmp_path, monkeypatch, capsys):
    import sys

    from lidar_video.cli import main

    monkeypatch.setattr(sys, "argv", ["lidar-video", "reconstruction-freeze", str(recipe),
                                     "--out", str(tmp_path / "run")])
    main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "frozen" and len(receipt["lock_sha256"]) == 64


def test_cli_missing_input_is_command_error(tmp_path, monkeypatch, capsys):
    import sys

    from lidar_video.cli import main

    monkeypatch.setattr(sys, "argv", ["lidar-video", "reconstruction-freeze", str(tmp_path / "absent.json"),
                                     "--out", str(tmp_path / "run")])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    assert "unavailable reconstruction input" in capsys.readouterr().err


def test_cli_rejected_run_has_nonzero_exit(recipe, tmp_path, monkeypatch, capsys):
    import sys

    from lidar_video.cli import main

    out = tmp_path / "run"
    receipt = runner.freeze_reconstruction(recipe, out)
    def invalid(*args, **kwargs):
        result = emit_scene(*args, **kwargs)
        (out / "scene/scenario.scenic").write_text("invalid Scenic program")
        return result
    monkeypatch.setattr(runner, "reconstruct", invalid)
    monkeypatch.setattr(sys, "argv", ["lidar-video", "reconstruction-run", str(out),
                                     "--lock-sha256", receipt["lock_sha256"]])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 3
    assert json.loads(capsys.readouterr().out)["status"] == "rejected"
