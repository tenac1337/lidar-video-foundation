import csv
import json
import subprocess
import sys

import cv2
import numpy as np
import pytest
from test_stray import fixture_directory

from lidar_video import intake
from lidar_video.creation_audit import digest
from lidar_video.intake import inspect_stray


def test_intake_is_read_only_and_preserves_source_association(tmp_path):
    source = fixture_directory(tmp_path / "source")
    before = {str(p): digest(p) for p in source.rglob("*") if p.is_file()}
    report = inspect_stray(source, tmp_path / "report.json", fps=30)
    assert report["status"] == "ready"
    assert report["source_frames"] == report["video"]["nb_read_frames"] == 6
    assert report["selected_source_frames"] == [0, 2, 4]
    assert report["source_time_s"] == pytest.approx([100, 100 + 2 / 60, 100 + 4 / 60])
    assert report["native_depth_shape"] == [6, 8]
    assert report["depth_frames_inspected"] == 6
    assert len(report["source_files_sha256"]) == 14
    assert not report["independent_depth_timestamps"]
    assert before == {str(p): digest(p) for p in source.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        inspect_stray(source, tmp_path / "report.json")
    with pytest.raises(ValueError, match="outside"):
        inspect_stray(source, source / "report.json")


@pytest.mark.parametrize("folder,dtype,value", [
    ("depth", np.uint8, 50), ("confidence", np.uint8, 255), ("confidence", np.uint16, 2),
])
def test_invalid_unsampled_frame_is_not_hidden_by_sampling(tmp_path, folder, dtype, value):
    source = fixture_directory(tmp_path / "source")
    cv2.imwrite(str(source / folder / "000001.png"), np.full((6, 8), value, dtype))
    report = inspect_stray(source, tmp_path / "report.json", fps=30)
    assert report["status"] == "invalid"
    assert 1 not in report["selected_source_frames"]
    assert any(e.get("source_frame") == 1 and e["stage"] == folder for e in report["errors"])


def test_missing_files_and_frame_inventory_are_reported(tmp_path):
    source = fixture_directory(tmp_path / "source")
    (source / "rgb.mp4").unlink()
    (source / "depth/000003.png").unlink()
    cv2.imwrite(str(source / "confidence/000099.png"), np.full((6, 8), 2, np.uint8))
    report = inspect_stray(source, tmp_path / "report.json")
    assert report["status"] == "invalid"
    assert any(e.get("missing") == ["000003.png"] for e in report["errors"])
    assert any(e.get("extra") == ["000099.png"] for e in report["errors"])
    assert any("rgb.mp4" in e["reason"] for e in report["errors"])


def test_zero_depth_is_explicit_and_selected_empty_frames_block_import(tmp_path):
    source = fixture_directory(tmp_path / "source")
    cv2.imwrite(str(source / "depth/000001.png"), np.zeros((6, 8), np.uint16))
    report = inspect_stray(source, tmp_path / "warning.json", fps=30)
    assert report["status"] == "ready"
    assert report["warnings"][0]["source_frames"] == [1]
    cv2.imwrite(str(source / "depth/000002.png"), np.zeros((6, 8), np.uint16))
    report = inspect_stray(source, tmp_path / "invalid.json", fps=30)
    assert report["status"] == "invalid"
    assert any(e.get("source_frames") == [2] for e in report["errors"])


@pytest.mark.parametrize("evaluation,valid", [([4], True), ([0], False), ([1], False), ([99], False)])
def test_split_plan_requires_disjoint_retained_source_ids(tmp_path, evaluation, valid):
    source = fixture_directory(tmp_path / "source")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(schema="scene-evaluation-split-v1",
                                   construction_source_frames=[0, 2], evaluation_source_frames=evaluation)))
    report = inspect_stray(source, tmp_path / "report.json", fps=30, plan=path)
    assert report["status"] == ("ready" if valid else "invalid")
    if valid:
        assert report["split_plan_sha256"] == digest(path)
    else:
        assert any(e["stage"] == "split" for e in report["errors"])


def test_absent_ffprobe_is_blocked_and_corrupt_video_is_invalid(tmp_path, monkeypatch):
    source = fixture_directory(tmp_path / "source")
    (source / "rgb.mp4").write_bytes(b"not an mp4")
    report = inspect_stray(source, tmp_path / "invalid.json")
    assert report["status"] == "invalid"
    def missing(*args):
        raise FileNotFoundError("ffprobe not installed")
    monkeypatch.setattr(intake, "_video_info", missing)
    report = inspect_stray(source, tmp_path / "blocked.json")
    assert report["status"] == "blocked"
    assert report["blocked"][0]["stage"] == "video"


def test_edit_list_does_not_shift_intake_frame_count(tmp_path):
    source = fixture_directory(tmp_path / "source")
    shifted = tmp_path / "shifted.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-itsoffset", "-0.016667", "-i", str(source / "rgb.mp4"),
                    "-map", "0:v:0", "-c:v", "copy", str(shifted)], check=True)
    shifted.replace(source / "rgb.mp4")
    report = inspect_stray(source, tmp_path / "report.json", fps=30)
    assert report["status"] == "ready"
    assert report["video"]["nb_read_frames"] == 6


def test_invalid_focal_length_is_not_deferred_until_import(tmp_path):
    source = fixture_directory(tmp_path / "source")
    path = source / "odometry.csv"
    rows = list(csv.DictReader(path.read_text().splitlines()))
    rows[1]["fx"] = "-1"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    report = inspect_stray(source, tmp_path / "report.json", fps=30)
    assert report["status"] == "invalid"
    assert any("Focal lengths" in e["reason"] for e in report["errors"])


def test_input_changed_during_inspection_is_not_ready(tmp_path, monkeypatch):
    source = fixture_directory(tmp_path / "source")
    original = intake._video_info
    def change(path):
        result = original(path)
        (source / "depth/000001.png").unlink()
        return result
    monkeypatch.setattr(intake, "_video_info", change)
    report = inspect_stray(source, tmp_path / "report.json", fps=30)
    assert report["status"] == "invalid"
    assert any(e["stage"] == "integrity" for e in report["errors"])


def test_invalid_cli_input_still_writes_machine_readable_report(tmp_path):
    out = tmp_path / "missing.json"
    run = subprocess.run([sys.executable, "-m", "lidar_video.cli", "inspect-stray",
                          str(tmp_path / "absent"), "--out", str(out)], capture_output=True, text=True)
    assert run.returncode == 3
    assert json.loads(out.read_text())["status"] == "invalid"


def test_nonfinite_fps_and_nonobject_plan_report_errors(tmp_path):
    source = fixture_directory(tmp_path / "source")
    report = inspect_stray(source, tmp_path / "nan.json", fps=float("nan"))
    assert report["status"] == "invalid" and report["requested_fps"] is None
    path = tmp_path / "plan.json"
    path.write_text("[]")
    report = inspect_stray(source, tmp_path / "bad-plan.json", plan=path)
    assert report["status"] == "invalid"
    assert any(e["stage"] == "split" for e in report["errors"])
