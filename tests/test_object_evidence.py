"""Regression cases for unreliable foreground depth and cross-view evidence."""

import copy

import cv2
import numpy as np
import pytest
import trimesh
from PIL import Image

from lidar_video.capture import normalize
from lidar_video.gpu_reconstruction import plan_instances
from lidar_video.mesh_alignment import validate_and_align
from lidar_video.object_evidence import assess_depth
from lidar_video.reconstruction import MeshQualityError


def fixture():
    n, h, w = 3, 80, 80
    poses = np.repeat(np.eye(4)[None], n, axis=0)
    poses[:, :3, :3] = np.diag([1, -1, -1])
    poses[:, 2, 3] = 1
    poses[:, 0, 3] = [0, 0.008, -0.008]
    k = np.array([[500.0, 0, 40], [0, 500.0, 40], [0, 0, 1]])
    data = dict(
        rgb=np.zeros((n, h, w, 3), np.uint8),
        depth=np.ones((n, h, w), np.float32),
        confidence=np.full((n, h, w), 2, np.uint8),
        K=k,
        world_from_camera=poses,
        rgb_time=np.arange(n, dtype=float),
        depth_time=np.arange(n, dtype=float),
    )
    masks = []
    for frame, shift in enumerate([0, -4, 4]):
        mask = np.zeros((h, w), bool)
        mask[17:37, 43 + shift : 63 + shift] = True
        data["depth"][frame][mask] = 0.94
        masks.append(mask)
    cal = dict(square_size_m=0.05, board_from_world=np.eye(4).tolist())
    return data, cal, masks


def test_board_like_core_fails_even_at_high_sensor_confidence():
    data, cal, masks = fixture()
    data["depth"][0][masks[0]] = 0.997
    quality, trusted, _ = assess_depth(data, cal, 0, masks[0])
    assert not quality["passed"]
    assert "board-like or implausible depth inside object" in quality["reasons"]
    assert not trusted.any()


def test_repeated_rgb_pixels_do_not_invent_native_samples():
    data, cal, masks = fixture()
    data["native_depth_shape"] = np.array([20, 20])
    quality, _, _ = assess_depth(data, cal, 0, masks[0])
    assert masks[0].sum() == 400
    assert quality["native_core_samples"] < 24
    assert not quality["passed"]


def test_missing_depth_is_not_rescued_by_a_few_high_pixels():
    data, cal, masks = fixture()
    data["confidence"][0, :, :53] = 0
    quality, _, _ = assess_depth(data, cal, 0, masks[0])
    assert not quality["passed"]
    assert "incomplete registered depth" in quality["reasons"]


def test_normalization_preserves_native_ids_through_intrinsic_change():
    data, _, _ = fixture()
    data["K"] = np.repeat(data["K"][None], 3, axis=0)
    data["K"][1, 0, 2] += 3
    data["native_depth_shape"] = np.array([20, 20])
    normalized = normalize(data)
    assert normalized["depth_sample_id"].shape == data["depth"].shape
    assert len(np.unique(normalized["depth_sample_id"][1])) <= 401
    np.testing.assert_array_equal(
        normalized["depth_sample_id"][1, :, :-3], normalized["depth_sample_id"][0, :, 3:]
    )
    assert (normalized["depth_sample_id"][1, :, -3:] == -1).all()


def test_selects_supported_alternative_and_requires_corroboration():
    data, cal, masks = fixture()
    data["depth"][0][masks[0]] = 0.997
    raw = {f: [dict(mask=m, score=0.9)] for f, m in enumerate(masks)}
    _, selected, audit = plan_instances(data, cal, raw, "skip")
    assert list(selected) == ["e5"]
    assert selected["e5"]["reference"]["frame"] in [1, 2]
    assert {o["frame"] for o in selected["e5"]["observations"]} == {1, 2}
    assert audit["frames"]["0"]["excluded"]
    _, single, _ = plan_instances(data, cal, {1: raw[1]}, "skip")
    assert not single


def test_same_square_without_matching_silhouette_is_not_associated():
    data, cal, masks = fixture()
    # Move one small object within the same square; distinct masks must not fuse.
    for f, m in enumerate(masks):
        data["depth"][f][m] = 1
        masks[f] = cv2.erode(m.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    masks[2] = np.roll(masks[2], 8, axis=0)
    for f, m in enumerate(masks):
        data["depth"][f][m] = 0.94
    raw = {f: [dict(mask=m, score=0.9)] for f, m in enumerate(masks)}
    _, selected, audit = plan_instances(data, cal, raw, "skip")
    assert not selected
    assert "e5" in audit["rejected_associations"]


def test_duplicate_source_views_do_not_supply_holdout_evidence():
    data, cal, masks = fixture()
    data["source_frame"] = np.array([9, 9, 9])
    _, selected, audit = plan_instances(
        data, cal, {f: [dict(mask=m, score=0.9)] for f, m in enumerate(masks)}, "skip"
    )
    assert not selected
    assert "distinct camera" in audit["rejected_associations"]["e5"]


def aligned_fixture(tmp_path):
    data, cal, masks = fixture()
    observations = []
    for f, m in enumerate(masks):
        Image.fromarray(m.astype(np.uint16)).save(tmp_path / f"{f}.png")
        observations.append(dict(frame=f, label=1, mask=f"{f}.png"))
    mesh = trimesh.creation.box(extents=[0.04, 0.04, 0.06])
    mesh.apply_translation([0.024, 0.026, 0.03])
    entry = dict(frame=0, static_scene=True, observations=observations)
    return mesh, entry, data, cal


def test_supported_geometry_is_evaluated_on_each_instance_view(tmp_path):
    mesh, entry, data, cal = aligned_fixture(tmp_path)
    original = mesh.extents.copy()
    aligned, evidence = validate_and_align(mesh, entry, tmp_path, data, cal)
    np.testing.assert_allclose(aligned.extents, original, atol=1e-10)
    assert abs(aligned.bounds[0, 2]) < 1e-9
    assert len(evidence["views"]) == 3
    assert sum(v["role"] == "held_out" for v in evidence["views"]) == 2
    assert all(v["passed"] for v in evidence["views"])


def test_missing_heldout_or_unreliable_view_never_passes_on_reference(tmp_path):
    mesh, entry, data, cal = aligned_fixture(tmp_path)
    lone = dict(entry, observations=entry["observations"][:1])
    with pytest.raises(MeshQualityError, match="held-out"):
        validate_and_align(mesh.copy(), lone, tmp_path, data, cal)
    bad = copy.deepcopy(data)
    bad["depth"][2] = 1
    with pytest.raises(MeshQualityError, match="Unreliable object depth"):
        validate_and_align(mesh.copy(), entry, tmp_path, bad, cal)


def test_support_fit_is_rechecked_against_heldout_depth(tmp_path):
    mesh, entry, data, cal = aligned_fixture(tmp_path)
    mesh.apply_translation([0, 0, -0.008])
    aligned, evidence = validate_and_align(mesh, entry, tmp_path, data, cal)
    assert abs(aligned.bounds[0, 2]) < 1e-9
    assert evidence["method"] == "support_snap"
    assert all(v["passed"] for v in evidence["views"])
    # A different observed object height in another view cannot pass by fitting frame 0.
    for observation in entry["observations"][1:]:
        mask = np.array(Image.open(tmp_path / observation["mask"])) > 0
        data["depth"][observation["frame"]][mask] = 0.88
    with pytest.raises(MeshQualityError):
        validate_and_align(aligned, entry, tmp_path, data, cal)


def test_invalid_observation_indices_fail_before_pose_lookup(tmp_path):
    mesh, entry, data, cal = aligned_fixture(tmp_path)
    entry["observations"][1]["frame"] = 999
    with pytest.raises(ValueError, match="Invalid observation frame"):
        validate_and_align(mesh, entry, tmp_path, data, cal)


def test_depth_sample_ids_cannot_invent_sensor_indices():
    data, cal, masks = fixture()
    data["depth_sample_id"] = np.full(data["depth"].shape, 999999, dtype=np.int64)
    with pytest.raises(ValueError, match="native-grid index"):
        assess_depth(data, cal, 0, masks[0])


def test_reference_fit_preserves_scale_and_validates_holdout(tmp_path):
    mesh, entry, data, cal = aligned_fixture(tmp_path)
    mesh.apply_translation([0.025, 0, 0.02])
    original_volume = mesh.volume
    aligned, evidence = validate_and_align(mesh, entry, tmp_path, data, cal)
    assert evidence["method"] == "reference_only_translation_with_held_out_validation"
    assert aligned.volume == pytest.approx(original_volume)
    assert all(v["passed"] for v in evidence["views"])
    assert abs(aligned.bounds[0, 2]) < 1e-9


def test_new_manifest_reaches_shared_graph_and_scenic(tmp_path):
    import json

    from lidar_video.reconstruction import reconstruct

    mesh, entry, data, cal = aligned_fixture(tmp_path)
    # Export camera-frame object pose with metric mesh vertices, exercising the
    # complete import -> validation -> CoACD -> scene graph -> Scenic path.
    mesh.export(tmp_path / "mesh.ply")
    pose = np.linalg.inv(data["world_from_camera"][0])
    (tmp_path / "pose.json").write_text(json.dumps(dict(camera_from_object=pose.tolist())))
    entry.update(mesh="mesh.ply", pose="pose.json", validation="instance-multiview-v1")
    (tmp_path / "meshes.json").write_text(
        json.dumps(dict(schema="roboskills-chess-meshes-v1", objects={"e5": entry}))
    )
    np.savez_compressed(tmp_path / "capture.npz", **data)
    (tmp_path / "calibration.json").write_text(json.dumps(cal))
    report = reconstruct(
        tmp_path / "capture.npz",
        tmp_path / "calibration.json",
        tmp_path / "result",
        mesh_manifest=tmp_path / "meshes.json",
        static_scene=True,
        samples=2,
        simulation=False,
    )
    assert report["imported_learned_meshes"] == 1
    assert report["geometric_completions"] == 0
    assert report["scenic_samples"] == 2
    graph = json.loads((tmp_path / "result/scene_graph.json").read_text())
    obj = next(n for n in graph["nodes"] if n["kind"] == "object")
    evidence = obj["evidence"]["learned_mesh"]
    assert evidence["validation"] == "instance-multiview-v1"
    assert len(evidence["views"]) == 3
    assert evidence["scale_reapplied"] is False


def test_gpu_handoff_keeps_scene_context_and_only_trusted_target_depth(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import lidar_video.gpu_reconstruction as gpu

    data, cal, masks = fixture()
    data["depth"][0][masks[0]] = 0.997
    np.savez_compressed(tmp_path / "capture.npz", **data)
    (tmp_path / "calibration.json").write_text(json.dumps(cal))
    monkeypatch.setenv("HF_TOKEN", "fixture-only-placeholder")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "fixture A100")
        ),
    )
    exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: str(path) == "/opt/venv/bin/python" or exists(path))

    class Detector:
        index = 0

        def detect(self, image, prompts):
            mask = masks[self.index]
            self.index += 1
            return {"chess piece": [dict(mask=mask, score=0.9)]}

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "roboskills.tools.sam3.sam3", SimpleNamespace(Sam=Detector))
    calls = []

    class Process:
        def __init__(self, argv, **kwargs):
            calls.append(argv)
            frame = int(argv[argv.index("--frame") + 1])
            assert frame in (1, 2)
            with np.load(argv[argv.index("--depth") + 1]) as archive:
                valid = archive["valid"][frame]
                assert np.all(archive["depth"][frame][~valid] == 0)
                assert 24 <= (valid & masks[frame]).sum() < masks[frame].sum()
                trusted = np.asarray(Image.open(tmp_path / "run/trusted_masks" / f"{frame:05d}.png")) > 0
                np.testing.assert_array_equal(valid & masks[frame], trusted)
                with np.load(tmp_path / "run/upstream/depth/depth.npz") as source:
                    np.testing.assert_array_equal(valid[~masks[frame]], source["valid"][frame][~masks[frame]])
                    np.testing.assert_array_equal(
                        archive["depth"][frame][valid], source["depth"][frame][valid]
                    )
                assert (valid & ~masks[frame]).any()

        def wait(self, **kwargs):
            return 0

    # Replace the GPU process boundary only; real preparation writes the measured
    # depth/video and the real planner selects and records the corroborating views.
    original_popen = gpu.subprocess.Popen
    monkeypatch.setattr(
        gpu.subprocess,
        "Popen",
        lambda argv, **kw: Process(argv, **kw) if "infer" in argv else original_popen(argv, **kw),
    )
    monkeypatch.setattr(gpu, "reconstruct", lambda *args, **kw: dict(args=args, kwargs=kw))
    out = tmp_path / "run"
    result = gpu.run_gpu(
        tmp_path / "capture.npz",
        tmp_path / "calibration.json",
        out,
        tmp_path / "native.py",
        static_scene=True,
        candidate_frames=[0, 1, 2],
    )
    assert len(calls) == 1
    manifest = json.loads((out / "meshes.json").read_text())["objects"]["e5"]
    assert manifest["frame"] in (1, 2)
    assert {o["frame"] for o in manifest["observations"]} == {1, 2}
    assert manifest["validation"] == "instance-multiview-v1"
    assert result["kwargs"]["masks"] == out / "trusted_masks"
    assert not np.asarray(Image.open(out / "trusted_masks/00000.png")).any()
    assignments = json.loads((out / "trusted_masks/instances.json").read_text())
    assert assignments["frames"]["0"] == {}
    assert assignments["frames"]["1"] == {"1": "e5"}

    data["depth"][:] = 1  # Board-level depth must stop before any additional model job.
    np.savez_compressed(tmp_path / "capture.npz", **data)
    with pytest.raises(ValueError, match="No objects have reliable native depth"):
        gpu.run_gpu(
            tmp_path / "capture.npz",
            tmp_path / "calibration.json",
            tmp_path / "abstained",
            tmp_path / "native.py",
            static_scene=True,
            candidate_frames=[0, 1, 2],
        )
    assert len(calls) == 1
    status = json.loads((tmp_path / "abstained/status.json").read_text())
    assert status["status"] == "abstained"
    assert not (tmp_path / "abstained/meshes.json").exists()


def test_integration_preserves_evidence_and_remaps_normalized_masks(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "integration", Path(__file__).parents[1] / "scripts/integrate_learned_chess.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data, cal, masks = fixture()
    data["source_frame"] = np.array([10, 20, 30])
    data["K"] = np.repeat(data["K"][None], 3, axis=0)
    data["K"][1, 0, 2] += 3
    cal["reference_frame"] = 0
    local, learned, out = (tmp_path / name for name in ("local", "learned", "out"))
    local.mkdir()
    learned.mkdir()
    np.savez_compressed(local / "refined_capture.npz", **data)
    np.savez_compressed(learned / "capture.npz", **normalize(data))
    for path in (local / "board_calibration.json", learned / "calibration.json"):
        path.write_text(json.dumps(cal))
    (local / "fusion_frames.json").write_text("[0,1,2]")
    observations = []
    for frame, mask in enumerate(masks):
        # Normalized principal point is 3 px left of frame 1's native K.
        stored = np.roll(mask, -3, axis=1) if frame == 1 else mask
        Image.fromarray(stored.astype(np.uint16)).save(learned / f"{frame}.png")
        observations.append(dict(frame=frame, label=1, mask=f"{frame}.png"))
    entry = dict(
        mesh="mesh.ply",
        pose="pose.json",
        frame=1,
        static_scene=True,
        validation="instance-multiview-v1",
        observations=observations,
    )
    (learned / "meshes.json").write_text(
        json.dumps(dict(schema="roboskills-chess-meshes-v1", objects={"e5": entry}))
    )
    for name in ("status.json", "segmentation_audit.json"):
        (learned / name).write_text("{}")
    monkeypatch.setattr(module, "reconstruct", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        sys, "argv", ["integrate", str(local), str(learned), "--out", str(out), "--no-simulation"]
    )
    module.main()
    mapped = json.loads((out / "mapped_mesh_manifest.json").read_text())["objects"]["e5"]
    assert mapped["validation"] == "instance-multiview-v1"
    assert mapped["static_scene"] is True
    assert mapped["frame"] == 1
    for observation in mapped["observations"]:
        np.testing.assert_array_equal(
            np.asarray(Image.open(observation["mask"])) > 0, masks[observation["frame"]]
        )


def test_cross_square_native_depth_fails_before_inference():
    data, cal, masks = fixture()
    data["K"][0, 2] += 12
    quality, _, _ = assess_depth(data, cal, 0, masks[0], square="e5")
    assert not quality["passed"]
    assert quality["native_square_fraction"] < 0.9
    assert "foreground depth spans multiple squares" in quality["reasons"]
    _, selected, _ = plan_instances(data, cal, {0: [dict(mask=masks[0], score=0.9)]}, "skip")
    assert not selected


def test_associated_instance_is_not_split_into_phantom_neighbor(tmp_path):
    import json

    from lidar_video.reconstruction import check_mesh_bounds, collect_squares, completed_hull

    data, cal, masks = fixture()
    data["K"][0, 2] += 12
    Image.fromarray(masks[0].astype(np.uint16)).save(tmp_path / "00000.png")
    (tmp_path / "instances.json").write_text(
        json.dumps(dict(schema="chess-instance-assignments-v1", frames={"0": {"1": "e5"}}))
    )
    points, _, _ = collect_squares(data, cal, [0], tmp_path)
    assert len(points[36]) == 400  # All depth remains associated with the same e5 instance.
    assert len(points[35]) == 0  # No invented d5 object from the cross-square fragment.
    with pytest.raises(MeshQualityError, match="crosses square boundary"):
        check_mesh_bounds(completed_hull(points[36]), np.array([0.0, 0.0]), np.array([0.05, 0.05]))
    (tmp_path / "instances.json").write_text(
        json.dumps(dict(schema="chess-instance-assignments-v1", frames={"0": {}}))
    )
    with pytest.raises(ValueError, match="requires an instance assignment"):
        collect_squares(data, cal, [0], tmp_path)


def test_large_base_offset_does_not_force_rotation_after_valid_support_snap(tmp_path):
    mesh, entry, data, cal = aligned_fixture(tmp_path)
    mesh.apply_translation([0, 0, -0.02])
    _, evidence = validate_and_align(mesh, entry, tmp_path, data, cal)
    assert evidence["method"] == "support_snap"
    assert len(evidence["attempts"]) == 1
    np.testing.assert_array_equal(np.array(evidence["board_refinement"])[:3, :3], np.eye(3))


def test_translation_candidate_is_fit_without_heldout_points(tmp_path):
    mesh, entry, data, cal = aligned_fixture(tmp_path)
    mesh.apply_translation([0.025, 0, 0.02])
    _, evidence = validate_and_align(mesh.copy(), entry, tmp_path, data, cal)
    candidate = evidence["attempts"][-1]
    assert candidate["method"] == "reference_only_translation_with_held_out_validation"
    altered = copy.deepcopy(data)
    for observation in entry["observations"][1:]:
        mask = np.asarray(Image.open(tmp_path / observation["mask"])) > 0
        altered["depth"][observation["frame"]][mask] = 0.954
    _, changed_evidence = validate_and_align(mesh.copy(), entry, tmp_path, altered, cal)
    attempts = changed_evidence["attempts"]
    assert changed_evidence["views"][1]["median_m"] != evidence["views"][1]["median_m"]
    translation = next(a for a in attempts if a["method"] == candidate["method"])
    np.testing.assert_allclose(translation["matrix"], candidate["matrix"], atol=1e-12)
