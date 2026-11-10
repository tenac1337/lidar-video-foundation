import numpy as np
import pytest
import trimesh

from lidar_video.silhouette_refinement import carve_reference


@pytest.fixture
def carve_input():
    mesh = trimesh.creation.box(extents=[0.04, 0.04, 0.06])
    mesh.apply_translation([0, 0, 0.03])
    labels = np.zeros((128, 128), dtype=np.uint8)
    labels[48:81, 48:65] = 1
    k = np.array([[300.0, 0, 64], [0, 300.0, 64], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1, -1, -1])
    pose[2, 3] = 0.6
    return mesh, labels, k, pose


def test_known_background_is_removed_without_scaling_or_mutating_input(carve_input):
    mesh, labels, k, pose = carve_input
    original = mesh.vertices.copy()
    candidate, evidence = carve_reference(mesh, labels, 1, k, pose)
    assert evidence["removed_voxels"] > 0
    assert evidence["accepted"] is False
    assert candidate.is_watertight
    assert candidate.bounds[1, 0] < 0.006
    assert candidate.extents[2] == pytest.approx(mesh.extents[2], abs=0.0011)
    np.testing.assert_array_equal(mesh.vertices, original)


def test_other_instances_are_unknown_not_background(carve_input):
    mesh, labels, k, pose = carve_input
    labels[:, 65:] = 2
    candidate, evidence = carve_reference(mesh, labels, 1, k, pose)
    assert evidence["removed_voxels"] == 0
    np.testing.assert_array_equal(candidate.vertices, mesh.vertices)
    np.testing.assert_array_equal(candidate.faces, mesh.faces)


def test_out_of_frame_geometry_is_not_erased(carve_input):
    mesh, labels, k, pose = carve_input
    k[0, 2] = 500
    candidate, evidence = carve_reference(mesh, labels, 1, k, pose)
    assert evidence["removed_voxels"] == 0
    np.testing.assert_array_equal(candidate.vertices, mesh.vertices)


def test_complete_removal_is_failure_not_an_empty_candidate(carve_input):
    mesh, labels, k, pose = carve_input
    labels[:] = 0
    labels[:3, :3] = 1
    with pytest.raises(ValueError, match="entire candidate"):
        carve_reference(mesh, labels, 1, k, pose)


@pytest.mark.parametrize("pitch", [0, -0.001, float("nan"), 0.1])
def test_invalid_voxel_resolution_rejected(carve_input, pitch):
    mesh, labels, k, pose = carve_input
    with pytest.raises(ValueError, match="Voxel pitch"):
        carve_reference(mesh, labels, 1, k, pose, pitch=pitch)


def test_cell_budget_checked_before_voxelization(carve_input, monkeypatch):
    mesh, labels, k, pose = carve_input
    monkeypatch.setattr(mesh, "voxelized", lambda **kwargs: pytest.fail("Allocated before budget check"))
    with pytest.raises(ValueError, match="cell budget"):
        carve_reference(mesh, labels, 1, k, pose, max_cells=10)


def test_nonwatertight_prior_and_missing_instance_rejected(carve_input):
    mesh, labels, k, pose = carve_input
    with pytest.raises(ValueError, match="nonempty positive label"):
        carve_reference(mesh, labels, 3, k, pose)
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    with pytest.raises(ValueError, match="watertight"):
        carve_reference(mesh, labels, 1, k, pose)


def test_saving_normal_field_never_changes_candidate(carve_input, tmp_path):
    mesh, labels, k, pose = carve_input
    expected, _ = carve_reference(mesh, labels, 1, k, pose)
    path = tmp_path / "field.npz"
    actual, evidence = carve_reference(mesh, labels, 1, k, pose, normal_field_path=path)
    np.testing.assert_array_equal(actual.vertices, expected.vertices)
    np.testing.assert_array_equal(actual.faces, expected.faces)
    with np.load(path) as field:
        assert field["matrix"].dtype == np.bool_
        assert field["matrix"].sum() == evidence["occupied_voxels_after"]
        assert field["pitch_m"] == 0.001
    assert evidence["normal_field"]["sha256"]
