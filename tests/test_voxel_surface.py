import numpy as np
import pytest
import trimesh

from lidar_video.multiview_shape import support_connected_volume
from lidar_video.voxel_surface import cancel_opposed_faces


def test_opposed_internal_membrane_removed_without_changing_boundary_or_volume():
    cube = trimesh.creation.box()
    vertices = np.vstack([cube.vertices, [[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0]]])
    faces = np.vstack([cube.faces, [[8, 9, 10], [10, 9, 8]]])
    source = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    repaired, record = cancel_opposed_faces(source)
    assert record["removed_opposed_pairs"] == 1
    np.testing.assert_array_equal(repaired.vertices, vertices)
    np.testing.assert_array_equal(repaired.faces, cube.faces)
    np.testing.assert_array_equal(source.faces, faces)
    assert repaired.is_watertight and repaired.is_winding_consistent
    assert repaired.volume == pytest.approx(source.volume)


def test_clean_surface_is_unchanged():
    source = trimesh.creation.icosphere(subdivisions=1)
    result, record = cancel_opposed_faces(source)
    np.testing.assert_array_equal(result.vertices, source.vertices)
    np.testing.assert_array_equal(result.faces, source.faces)
    assert record["removed_opposed_pairs"] == 0


def test_sparse_voxel_saddle_extracts_cancelling_membrane():
    # A minimal ambiguous voxel configuration, independent of object category.
    grid = np.zeros((4, 4, 4), bool)
    indices = np.array(
        [[0, 1, 1], [0, 2, 1], [0, 2, 2], [1, 1, 2], [1, 2, 1], [2, 1, 1], [2, 1, 2], [2, 2, 2]]
    )
    grid[tuple(indices.T)] = True
    source = trimesh.voxel.ops.matrix_to_marching_cubes(grid, pitch=0.001)
    result, record = cancel_opposed_faces(source)
    assert record["removed_opposed_pairs"] > 0
    assert not source.is_watertight
    assert result.is_watertight and result.is_winding_consistent
    np.testing.assert_array_equal(source.vertices, result.vertices)
    assert result.volume == pytest.approx(source.volume, abs=1e-18)


@pytest.mark.parametrize("extra", [[[0, 1, 2], [1, 2, 0]], [[0, 1, 2], [2, 1, 0], [0, 1, 2]]])
def test_ambiguous_duplicates_are_not_silently_repaired(extra):
    cube = trimesh.creation.box()
    v = np.vstack([cube.vertices, [[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0]]])
    source = trimesh.Trimesh(vertices=v, faces=np.vstack([cube.faces, np.array(extra) + 8]), process=False)
    with pytest.raises(ValueError):
        cancel_opposed_faces(source)


def test_nearby_nonidentical_faces_do_not_cancel():
    cube = trimesh.creation.box()
    v = np.vstack([cube.vertices, [[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 1e-12]]])
    source = trimesh.Trimesh(
        vertices=v, faces=np.vstack([cube.faces, [[8, 9, 10], [10, 9, 11]]]), process=False
    )
    result, record = cancel_opposed_faces(source)
    assert record["removed_opposed_pairs"] == 0
    assert len(result.faces) == len(source.faces)
    assert not result.is_watertight


def test_support_component_is_not_selected_by_size():
    grid = np.zeros((8, 8, 8), bool)
    grid[1, 1, 0:2] = True
    grid[3:7, 3:7, 4:7] = True  # Larger detached volume must not win.
    selected, record = support_connected_volume(grid)
    assert selected.sum() == 2 and record["discarded_cells"] == 48
    assert record["selection_by_size"] is False
    assert grid.sum() == 50


@pytest.mark.parametrize("touching", [0, 2])
def test_missing_or_ambiguous_support_components_abstain(touching):
    grid = np.zeros((6, 6, 6), bool)
    grid[2, 2, 3] = True
    if touching:
        grid[0, 0, 0] = True
        grid[5, 5, 0] = True
    with pytest.raises(ValueError, match="absent or ambiguous"):
        support_connected_volume(grid)
