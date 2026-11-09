import copy
import json

import numpy as np
import pytest
import trimesh
from PIL import Image
from test_object_evidence import aligned_fixture

from lidar_video.creation_audit import digest
from lidar_video.mesh_alignment import construction_split, validate_and_align
from lidar_video.multiview_completion import complete_case, constrained_carve
from lidar_video.reconstruction import MeshQualityError


def test_v2_roles_are_explicit_and_support_fit_keeps_validation_out(tmp_path):
    mesh, entry, data, cal = aligned_fixture(tmp_path)
    entry.update(validation="instance-multiview-v2", construction_frames=[0, 1], validation_frames=[2])
    _, evidence = validate_and_align(mesh, entry, tmp_path, data, cal)
    assert evidence["validation"] == "instance-multiview-v2"
    assert [v["role"] for v in evidence["views"]] == ["training", "training", "held_out"]
    assert evidence["method"] == "support_snap"


@pytest.mark.parametrize(
    "training,held_out",
    [([0, 1, 2], []), ([0, 1], [1, 2]), ([0], [1, 2]), ([1, 2], [0]), ([0, True], [2]), ([0, 1], [5])],
)
def test_invalid_split_is_rejected(tmp_path, training, held_out):
    _, entry, data, _ = aligned_fixture(tmp_path)
    entry.update(validation="instance-multiview-v2", construction_frames=training, validation_frames=held_out)
    with pytest.raises(ValueError, match="partition"):
        construction_split(entry, data)


def test_v1_cannot_hide_construction_metadata_and_v2_cannot_reuse_source(tmp_path):
    _, entry, data, _ = aligned_fixture(tmp_path)
    entry.update(construction_frames=[0, 1], validation_frames=[2])
    with pytest.raises(ValueError, match="require instance-multiview-v2"):
        construction_split(entry, data)
    entry["validation"] = "instance-multiview-v2"
    data["source_frame"] = np.array([10, 11, 10])
    with pytest.raises(ValueError, match="distinct from every"):
        construction_split(entry, data)


def test_board_and_footprint_are_shape_constraints_not_translation_or_rescaling():
    mesh = trimesh.creation.box(extents=[0.08, 0.08, 0.08])
    mesh.apply_translation([0, 0, 0.02])
    original = mesh.vertices.copy()
    labels = np.ones((128, 128), np.uint16)
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 0.6
    k = np.array([[300.0, 0, 64], [0, 300.0, 64], [0, 0, 1]])
    views = [dict(labels=labels, label=1, K=k, board_from_camera=pose)] * 2
    candidate, evidence = constrained_carve(mesh, views, [-0.025, -0.025], [0.025, 0.025])
    assert candidate.is_watertight
    assert candidate.bounds[0, 2] >= 0
    assert candidate.bounds[1, 2] == pytest.approx(0.06, abs=0.001)
    assert np.all(candidate.bounds[0, :2] >= -0.025)
    assert np.all(candidate.bounds[1, :2] <= 0.025)
    assert evidence["below_board_voxels"] > 0 and evidence["outside_footprint_voxels"] > 0
    assert not evidence["accepted"] and not evidence["scale_reapplied"]
    np.testing.assert_array_equal(mesh.vertices, original)
    with pytest.raises(ValueError, match="cell budget"):
        constrained_carve(mesh, views, [-0.025, -0.025], [0.025, 0.025], max_cells=1)


def test_reserved_mask_cannot_change_construction_but_can_reject_candidate(tmp_path):
    mesh, entry, data, cal = aligned_fixture(tmp_path)
    mesh.export(tmp_path / "prior.ply")
    pose = np.linalg.inv(data["world_from_camera"][0])
    (tmp_path / "pose.json").write_text(json.dumps({"camera_from_object": pose.tolist()}))
    np.savez_compressed(tmp_path / "capture.npz", **data)
    (tmp_path / "cal.json").write_text(json.dumps(cal))
    entry.update(mesh="prior.ply", pose="pose.json", validation="instance-multiview-v1")
    (tmp_path / "meshes.json").write_text(
        json.dumps({"schema": "roboskills-chess-meshes-v1", "objects": {"e5": entry}})
    )
    case = dict(
        schema="creation-audit-case-v1",
        evaluation_scope="development",
        geometry_scope="measured",
        expected_objects=["e5"],
        capture="capture.npz",
        calibration="cal.json",
        mesh_manifest="meshes.json",
    )
    (tmp_path / "case.json").write_text(json.dumps(case))
    plan = dict(
        schema="multiview-completion-plan-v1",
        objects={
            "e5": dict(
                construction_frames=[0, 1],
                validation_frames=[2],
                physical_priors=["board_plane", "assigned_square"],
            )
        },
    )
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    complete_case(tmp_path / "case.json", tmp_path / "plan.json", tmp_path / "first", save_normal_field=True)
    Image.fromarray(np.zeros_like(data["depth"][2], dtype=np.uint16)).save(tmp_path / "2.png")
    complete_case(tmp_path / "case.json", tmp_path / "plan.json", tmp_path / "second", save_normal_field=True)
    assert digest(tmp_path / "first/objects/e5/mesh.ply") == digest(tmp_path / "second/objects/e5/mesh.ply")
    with np.load(tmp_path / "first/objects/e5/normal_field.npz") as first:
        with np.load(tmp_path / "second/objects/e5/normal_field.npz") as second:
            for key in first.files:
                np.testing.assert_array_equal(first[key], second[key])
    updated = json.loads((tmp_path / "second/meshes.json").read_text())["objects"]["e5"]
    with pytest.raises((ValueError, MeshQualityError), match="nonempty|Unreliable"):
        validate_and_align(mesh.copy(), updated, tmp_path, data, cal)
    bad = copy.deepcopy(plan)
    bad["objects"]["e5"]["physical_priors"] = []
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Explicitly declare"):
        complete_case(tmp_path / "case.json", tmp_path / "bad.json", tmp_path / "bad-out")
