import json

import numpy as np
import pytest
from test_general_scene import fixture

from lidar_video.general_discovery import discover, support_depth_diagnostics
from lidar_video.run_checkpoint import pack, restore


def test_discovery_preserves_raw_masks_and_source_ids_without_mesh_generation(tmp_path):
    data, detections, _ = fixture()
    path = tmp_path / "capture.npz"
    np.savez_compressed(path, **data)
    checkpoints = []

    class Detector:
        frame = 0
        closed = False

        def detect(self, rgb, prompts, threshold):
            assert threshold == 0.3
            items = [dict(d, score=0.9) for d in detections[self.frame]]
            self.frame += 1
            return {p: items for p in prompts}

        def close(self):
            self.closed = True

    detector = Detector()

    def checkpoint(root, stage):
        checkpoints.append(
            pack(root, [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()], stage=stage)
        )

    result = discover(
        path, tmp_path / "run", frames=[0, 1, 2], detector_factory=lambda: detector, checkpoint=checkpoint
    )
    assert detector.closed and result["status"] == "diagnostics_completed"
    assert result["source_frames"] == [11, 43, 72]
    assert all(r["accepted_tracks"] == 2 for r in result["conditions"].values())
    assert not result["inventory_complete"] and not result["frozen_evaluation_ready"]
    assert len(checkpoints) == 4
    payload = checkpoints[-1]
    restored = restore(payload, tmp_path / "restored", expected_sha256=payload["sha256"])
    tracks = json.loads((restored / "p00-nms-0.8/tracks.json").read_text())
    assert [o["source_frame"] for o in tracks["object_000"]["observations"]] == [11, 43, 72]
    assert not list(restored.rglob("*.ply"))
    assert (restored / "p00-nms-0.8/support-depth-diagnostic.json").exists()


def test_support_depth_diagnostic_separates_raised_and_plane_returns_without_mutation():
    data, detections, _ = fixture()
    raised = support_depth_diagnostics(data, detections)
    assert all(r["fraction_above_15mm"] == 1 for r in raised["rows"])
    # Same RGB masks and confident depth, but sensor returns the support behind them.
    # This could also represent a legitimate flat object; never infer absence.
    data["depth"][:] = 0.8
    original = data["depth"].copy()
    flat = support_depth_diagnostics(data, detections)
    assert len(flat["rows"]) == 6
    assert all(r["fraction_within_6mm_of_plane"] == 1 for r in flat["rows"])
    assert all(not r["object_surface_confirmed"] for r in flat["rows"])
    np.testing.assert_array_equal(original, data["depth"])


def test_support_depth_diagnostic_does_not_fill_missing_background():
    data, detections, _ = fixture()
    for frame, items in detections.items():
        occupied = np.logical_or.reduce([item["mask"] for item in items])
        data["confidence"][frame][~occupied] = 0
    result = support_depth_diagnostics(data, detections)
    assert len(result["rows"]) == 6
    assert all("unresolved" in row and "median_height_m" not in row for row in result["rows"])


def test_detector_failure_retains_completed_view_and_closes_worker(tmp_path):
    data, detections, _ = fixture()
    path = tmp_path / "capture.npz"
    np.savez_compressed(path, **data)

    class Detector:
        calls = 0
        closed = False

        def detect(self, rgb, prompts, threshold):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("injected detector failure")
            return {p: [dict(d, score=0.9) for d in detections[0]] for p in prompts}

        def close(self):
            self.closed = True

    detector = Detector()
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="injected"):
        discover(path, out, frames=[0, 1], detector_factory=lambda: detector)
    assert detector.closed
    assert (out / "raw-p00-f00000.npz").exists()
    assert json.loads((out / "status.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        discover(path, out, frames=[0, 1], detector_factory=lambda: detector)
