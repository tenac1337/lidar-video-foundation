import json

import numpy as np
import pytest
from PIL import Image
from test_general_scene import fixture

from lidar_video.capture import write_json
from lidar_video.general_run import execute, freeze, load_detections, verify


def recipe_fixture(root, **changes):
    root.mkdir()
    data, detections, priors = fixture()
    np.savez_compressed(root / "capture.npz", **data)
    frames = {}
    for frame, items in detections.items():
        labels = np.zeros(data["depth"].shape[1:], np.uint16)
        for label, item in enumerate(items, 1):
            labels[item["mask"]] = label
        filename = f"{frame}.png"
        Image.fromarray(labels).save(root / filename)
        frames[str(data["source_frame"][frame])] = dict(labels=filename)
    write_json(root / "detections.json", dict(schema="generic-instance-masks-v1", frames=frames))
    entries = {}
    for name, prior in priors.items():
        prior["visual"].export(root / f"{name}.ply")
        entries[name] = dict(visual=f"{name}.ply", collision=[f"{name}.ply"], provenance=prior["provenance"])
    write_json(
        root / "meshes.json", dict(schema="generic-world-mesh-priors-v1", units="metres", objects=entries)
    )
    config = dict(
        schema="general-mesh-recipe-v1",
        scene_id="analytic-development",
        evaluation_scope="development",
        static_scene=True,
        capture="capture.npz",
        detections="detections.json",
        mesh_manifest="meshes.json",
        policy={},
        jitter_m=0.03,
        samples=4,
        seed=7,
        simulate=False,
    )
    config.update(changes)
    write_json(root / "recipe.json", config)
    return root / "recipe.json", data


def test_locked_run_and_no_repeat(tmp_path):
    recipe, _ = recipe_fixture(tmp_path / "inputs")
    run = tmp_path / "run"
    receipt = freeze(recipe, run)
    result = execute(run, receipt["lock_sha256"])
    assert result["status"] == "completed"
    assert result["associated_objects"] == 2
    assert result["replay_exact"]
    assert result["variation_attempts"] == [1] * 4
    with pytest.raises(FileExistsError):
        execute(run, receipt["lock_sha256"])


@pytest.mark.parametrize("filename", ["recipe.json", "0.png", "object_000.ply"])
def test_changed_inputs_block_before_attempt(tmp_path, filename):
    recipe, _ = recipe_fixture(tmp_path / "inputs")
    run = tmp_path / "run"
    receipt = freeze(recipe, run)
    path = recipe.parent / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="inputs changed"):
        execute(run, receipt["lock_sha256"])
    assert not (run / "execution.json").exists()


def test_bad_lock_and_evaluation_scope_blocked(tmp_path):
    recipe, _ = recipe_fixture(tmp_path / "inputs")
    run = tmp_path / "run"
    freeze(recipe, run)
    with pytest.raises(ValueError, match="lock hash"):
        verify(run, "0" * 64)
    config = json.loads(recipe.read_text())
    config["evaluation_scope"] = "evaluation"
    write_json(recipe, config)
    with pytest.raises(ValueError, match="development scope"):
        freeze(recipe, tmp_path / "eval-run")


def test_wrong_source_frame_is_not_silently_reindexed(tmp_path):
    recipe, data = recipe_fixture(tmp_path / "inputs")
    path = recipe.parent / "detections.json"
    record = json.loads(path.read_text())
    record["frames"]["0"] = record["frames"].pop("11")
    write_json(path, record)
    with pytest.raises(ValueError, match="source frame"):
        load_detections(dict(detections=str(path)), data)


def test_failed_attempt_keeps_failure_and_cannot_retry(tmp_path):
    recipe, _ = recipe_fixture(tmp_path / "inputs")
    path = recipe.parent / "meshes.json"
    record = json.loads(path.read_text())
    record["objects"].pop("object_001")
    write_json(path, record)
    run = tmp_path / "run"
    receipt = freeze(recipe, run)
    with pytest.raises(ValueError, match="inventory"):
        execute(run, receipt["lock_sha256"])
    assert json.loads((run / "result.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        execute(run, receipt["lock_sha256"])
