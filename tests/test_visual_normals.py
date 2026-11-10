import io
import json

import numpy as np
import pytest
from trimesh.exchange.obj import load_obj

from lidar_video.visual_normals import _obj_geometry, export_visual_normals, field_normals, save_normal_field


def grid():
    matrix = np.ones((8, 8, 8), dtype=bool)
    return matrix, np.zeros(3), 0.001


def test_outward_unit_normals_and_rigid_frame_mapping():
    matrix, origin, pitch = grid()
    points = np.array([[-0.0005, 0.0035, 0.0035], [0.0075, 0.0035, 0.0035]])
    expected = np.array([[-1, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(field_normals(points, matrix, origin, pitch, np.eye(4)), expected, atol=1e-8)
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [0.1, -0.2, 0.3]
    local = (points - transform[:3, 3]) @ transform[:3, :3]
    result = field_normals(local, matrix, origin, pitch, transform)
    np.testing.assert_allclose(result, expected @ transform[:3, :3], atol=1e-8)


def test_obj_export_preserves_vertices_topology_uvs_and_inputs(tmp_path):
    matrix, origin, pitch = grid()
    source = tmp_path / "source.obj"
    text = (
        "v -0.0005 0.002 0.002\nv -0.0005 0.004 0.002\nv -0.0005 0.002 0.004\n"
        "vt 0 0\nvt 1 0\nvt 0 1\nvn 0 0 1\nf -3/1/1 -2/2/1 -1/3/1\n"
    )
    source.write_text(text)
    field = tmp_path / "grid.npz"
    save_normal_field(field, matrix, origin, pitch)
    before = field.read_bytes()
    out = tmp_path / "output.obj"
    report = export_visual_normals(source, field, np.eye(4), out)
    assert source.read_text() == text and field.read_bytes() == before
    a, b = _obj_geometry(text), _obj_geometry(out.read_text())
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert "f 1/1/1 2/2/2 3/3/3" in out.read_text()
    loaded = load_obj(io.StringIO(out.read_text()))["geometry"]["geometry"]
    np.testing.assert_allclose(loaded["vertex_normals"], np.tile([-1, 0, 0], (3, 1)), atol=0.01)
    assert report["geometry_identical"]
    assert json.loads(out.with_suffix(".normals.json").read_text())["output_obj_sha256"]
    with pytest.raises(FileExistsError):
        export_visual_normals(source, field, np.eye(4), out)


@pytest.mark.parametrize("problem", ["outside", "undefined", "nonrigid", "budget", "sigma", "nonbool"])
def test_invalid_fields_do_not_guess_normals(problem):
    matrix, origin, pitch = grid()
    points = np.array([[-0.0005, 0.003, 0.003]])
    transform = np.eye(4)
    kwargs = {}
    if problem == "outside":
        points += 100
    if problem == "undefined":
        matrix = np.ones((30, 30, 30), dtype=bool)
        points = np.array([[0.015] * 3])
    if problem == "nonrigid":
        transform[0, 0] = 2
    if problem == "budget":
        kwargs["max_cells"] = 1
    if problem == "sigma":
        kwargs["sigma"] = float("nan")
    if problem == "nonbool":
        matrix = matrix.astype(float)
    with pytest.raises(ValueError):
        field_normals(points, matrix, origin, pitch, transform, **kwargs)
