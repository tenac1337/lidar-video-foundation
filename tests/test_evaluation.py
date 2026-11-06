import json

import numpy as np
import pytest
from PIL import Image
from test_object_evidence import aligned_fixture

from lidar_video.creation_audit import digest
from lidar_video.evaluation import freeze_evaluation, score_evaluation, split_evaluation_capture


def case_fixture(tmp_path):
    mesh, _, data, cal = aligned_fixture(tmp_path)
    data["source_frame"] = np.array([100, 200, 300])
    np.savez_compressed(tmp_path / "capture.npz", **data)
    (tmp_path / "calibration.json").write_text(json.dumps(cal))
    mesh.export(tmp_path / "prediction.ply")
    prediction = dict(
        schema="frozen-scene-prediction-v1", scene_id="fixture", method_id="fixture-method",
        method_decision="accept", units="metres", construction_source_frames=[100],
        calibration_source_frames=[100],
        objects=[dict(id="object", mesh="prediction.ply", board_from_mesh=np.eye(4).tolist())],
    )
    (tmp_path / "prediction.json").write_text(json.dumps(prediction))
    case = dict(
        schema="scene-evaluation-case-v1", scene_id="fixture", evaluation_scope="development_replay",
        expected_objects=["object"], evaluation_source_frames=[300], prediction="prediction.json",
        capture="capture.npz", calibration="calibration.json",
        observations=[dict(object_id="object", source_frame=300, label=1, mask="2.png")],
    )
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    return path, case, prediction


def evaluate(path, out):
    frozen = freeze_evaluation(path, out)
    return score_evaluation(out, expected_lock_sha256=frozen["lock_sha256"])


def test_scoring_does_not_fit_or_change_prediction_and_detects_shift(tmp_path):
    path, _, prediction = case_fixture(tmp_path)
    original = digest(tmp_path / "prediction.ply")
    first = evaluate(path, tmp_path / "first")
    assert first["views"][0]["native_samples"] == 400
    assert digest(tmp_path / "prediction.ply") == original
    prediction["objects"][0]["board_from_mesh"][0][3] = 0.1
    (tmp_path / "prediction.json").write_text(json.dumps(prediction))
    shifted = evaluate(path, tmp_path / "shifted")
    assert shifted["views"][0]["median_m"] > first["views"][0]["median_m"] + .05
    assert shifted["views"][0]["silhouette_iou"] < first["views"][0]["silhouette_iou"]
    with pytest.raises(FileExistsError):
        score_evaluation(tmp_path / "first", expected_lock_sha256=first["lock_sha256"])


@pytest.mark.parametrize("field", ["construction_source_frames", "calibration_source_frames"])
def test_source_frame_leakage_is_rejected_before_scoring(tmp_path, field):
    path, _, prediction = case_fixture(tmp_path)
    prediction[field] = [300]
    (tmp_path / "prediction.json").write_text(json.dumps(prediction))
    with pytest.raises(ValueError, match="overlap"):
        freeze_evaluation(path, tmp_path / "frozen")


@pytest.mark.parametrize("changed", ["prediction.ply", "2.png", "prediction.json", "calibration.json"])
def test_changed_inputs_cannot_reuse_freeze(tmp_path, changed):
    path, _, _ = case_fixture(tmp_path)
    frozen = freeze_evaluation(path, tmp_path / "frozen")
    with (tmp_path / changed).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="changed"):
        score_evaluation(tmp_path / "frozen", expected_lock_sha256=frozen["lock_sha256"])


def test_lock_tampering_and_accidental_confirmatory_label_fail(tmp_path):
    path, case, _ = case_fixture(tmp_path)
    case["evaluation_scope"] = "confirmatory"
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="attestation"):
        freeze_evaluation(path, tmp_path / "frozen")
    case.update(previously_inspected=False, custodian_attestation="Synthetic test only")
    path.write_text(json.dumps(case))
    frozen = freeze_evaluation(path, tmp_path / "frozen")
    lock = tmp_path / "frozen/lock.json"
    lock.write_text(lock.read_text() + " ")
    with pytest.raises(ValueError, match="precommitted"):
        score_evaluation(tmp_path / "frozen", expected_lock_sha256=frozen["lock_sha256"])


def test_missing_predictions_and_missing_reference_are_not_silently_dropped(tmp_path):
    path, _, prediction = case_fixture(tmp_path)
    prediction["objects"] = []
    (tmp_path / "prediction.json").write_text(json.dumps(prediction))
    result = evaluate(path, tmp_path / "missing")
    assert result["missing_objects"] == ["object"]
    assert result["views"][0]["status"] == "missing_prediction"
    assert result["objects"][0]["mean_view_median_m"] is None
    Image.fromarray(np.zeros((80, 80), dtype=np.uint16)).save(tmp_path / "2.png")
    result = evaluate(path, tmp_path / "no-reference")
    assert result["views"][0]["status"] == "unscorable_reference"
    assert "silhouette_iou" not in result["views"][0]


def test_no_depth_does_not_turn_into_zero_error(tmp_path):
    path, _, _ = case_fixture(tmp_path)
    p = tmp_path / "capture.npz"
    with np.load(p) as archive:
        data = dict(archive)
    mask = np.asarray(Image.open(tmp_path / "2.png")) == 1
    data["confidence"][2][mask] = 0
    np.savez_compressed(p, **data)
    result = evaluate(path, tmp_path / "frozen")
    assert result["views"][0]["status"] == "no_usable_depth"
    assert result["views"][0]["native_samples"] == 0
    assert "median_m" not in result["views"][0]


def test_repeated_rgb_pixels_count_native_samples_once(tmp_path):
    path, _, _ = case_fixture(tmp_path)
    p = tmp_path / "capture.npz"
    with np.load(p) as archive:
        data = dict(archive)
    data["native_depth_shape"] = np.array([20, 20])
    np.savez_compressed(p, **data)
    result = evaluate(path, tmp_path / "frozen")
    assert 0 < result["views"][0]["native_samples"] < 400


def test_duplicate_source_frame_ids_and_reference_labels_rejected(tmp_path):
    path, case, _ = case_fixture(tmp_path)
    case["expected_objects"].append("other")
    case["observations"].append(dict(object_id="other", source_frame=300, label=1, mask="2.png"))
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="share an instance label"):
        freeze_evaluation(path, tmp_path / "bad-label")
    case["observations"].pop()
    case["expected_objects"].pop()
    path.write_text(json.dumps(case))
    p = tmp_path / "capture.npz"
    with np.load(p) as archive:
        data = dict(archive)
    data["source_frame"] = np.array([100, 300, 300])
    np.savez_compressed(p, **data)
    with pytest.raises(ValueError, match="unique original"):
        evaluate(path, tmp_path / "bad-source")


def test_split_materializes_disjoint_pixels_and_preserves_native_metadata(tmp_path):
    case_fixture(tmp_path)
    capture = tmp_path / "capture.npz"
    with np.load(capture) as archive:
        data = dict(archive)
    data["native_K"] = np.repeat(data["K"][None], 3, axis=0)
    data["native_depth_shape"] = np.array([80, 80])
    data["depth_sample_id"] = np.repeat(np.arange(6400).reshape(1, 80, 80), 3, axis=0)
    np.savez_compressed(capture, **data)
    plan = tmp_path / "split-plan.json"
    plan.write_text(json.dumps(dict(
        schema="scene-evaluation-split-v1", construction_source_frames=[200, 100],
        evaluation_source_frames=[300],
    )))
    result = split_evaluation_capture(capture, plan, tmp_path / "split")
    assert result["outputs"]["construction"]["source_frames"] == [100, 200]
    with np.load(result["outputs"]["evaluation"]["path"]) as scored:
        np.testing.assert_array_equal(scored["depth"], data["depth"][[2]])
        np.testing.assert_array_equal(scored["native_K"], data["native_K"][[2]])
        np.testing.assert_array_equal(scored["depth_sample_id"], data["depth_sample_id"][[2]])
        np.testing.assert_array_equal(scored["rgb_time"], data["rgb_time"][[2]])


@pytest.mark.parametrize("frames,match", [([100], "overlap"), ([999], "absent")])
def test_bad_split_rejected(tmp_path, frames, match):
    case_fixture(tmp_path)
    plan = tmp_path / "split-plan.json"
    plan.write_text(json.dumps(dict(
        schema="scene-evaluation-split-v1", construction_source_frames=[100],
        evaluation_source_frames=frames,
    )))
    with pytest.raises(ValueError, match=match):
        split_evaluation_capture(tmp_path / "capture.npz", plan, tmp_path / "split")
