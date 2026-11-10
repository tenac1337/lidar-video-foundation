import copy
import json

import numpy as np
import pytest
import trimesh
from scipy.spatial import ConvexHull

from lidar_video.collision_padding import clip_convex_padding, trim_collision_padding


def test_convex_cut_removes_only_below_plane_and_does_not_mutate():
    mesh = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
    mesh.apply_translation([0, 0, 0.0097])
    before = mesh.vertices.copy()
    clipped, report = clip_convex_padding(mesh, 0)
    assert clipped.volume == pytest.approx(0.02 * 0.02 * 0.0197)
    assert clipped.bounds[0, 2] == pytest.approx(0)
    hull = ConvexHull(mesh.vertices)
    assert np.max(clipped.vertices @ hull.equations[:, :3].T + hull.equations[:, 3]) < 1e-10
    assert clipped.is_watertight and clipped.is_convex
    assert report["depth_m"] == pytest.approx(0.0003)
    np.testing.assert_array_equal(mesh.vertices, before)


def test_noop_and_budgets_fail_closed():
    mesh = trimesh.creation.box(extents=[0.01] * 3)
    result, report = clip_convex_padding(mesh, -0.006)
    np.testing.assert_array_equal(mesh.vertices, result.vertices)
    assert not report["changed"]
    with pytest.raises(ValueError, match="budget"):
        clip_convex_padding(mesh, 0)
    mesh = trimesh.creation.box(extents=[0.01, 0.01, 0.0005])
    with pytest.raises(ValueError, match="remove or flatten"):
        clip_convex_padding(mesh, mesh.bounds[1, 2], max_depth=0.001)


def source_scene(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    visual = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
    collision = visual.copy()
    collision.vertices[:, 2] -= 0.0003
    visual.export(folder / "visual.obj")
    collision.export(folder / "part.obj")
    graph = dict(
        units="metres",
        nodes=[
            dict(id="table", kind="support", position_m=[0, 0, -0.005], size_m=[0.2, 0.2, 0.01]),
            dict(
                id="piece",
                kind="object",
                position_m=[0, 0, 0.01],
                size_m=[0.02, 0.02, 0.0206],
                mesh_path="visual.obj",
                collision_paths=["part.obj"],
            ),
        ],
        edges=[dict(source="piece", target="table", relation="on")],
    )
    path = folder / "scene_graph.json"
    path.write_text(json.dumps(graph))
    return path, graph


def test_scene_variant_preserves_visual_pose_and_source_assets(tmp_path):
    path, graph = source_scene(tmp_path)
    original = path.read_bytes()
    collider = (path.parent / "part.obj").read_bytes()
    out = tmp_path / "variant"
    report = trim_collision_padding(path, out, ["piece"], support_tolerance=0.001)
    result = json.loads((out / "scene_graph.json").read_text())
    assert result["nodes"] == graph["nodes"] and result["edges"] == graph["edges"]
    assert report["visual_bytes_unchanged"] and report["poses_dimensions_edges_unchanged"]
    assert (out / "visual.obj").read_bytes() == (path.parent / "visual.obj").read_bytes()
    assert (out / "part.obj").read_bytes() != collider
    assert path.read_bytes() == original and (path.parent / "part.obj").read_bytes() == collider
    assert trimesh.load(out / "part.obj", force="mesh").bounds[0, 2] == pytest.approx(-0.01)
    with pytest.raises(FileExistsError):
        trim_collision_padding(path, out, ["piece"], support_tolerance=0.001)


@pytest.mark.parametrize(
    "problem",
    [
        "unknown",
        "shared_visual",
        "aliased_visual",
        "shared_collision",
        "duplicate",
        "too_deep",
        "volume_loss",
    ],
)
def test_invalid_scene_corrections_fail_before_output(tmp_path, problem):
    path, graph = source_scene(tmp_path)
    names = ["piece"]
    kwargs = {"support_tolerance": 0.001}
    if problem == "unknown":
        names = ["missing"]
    if problem == "duplicate":
        names = ["piece", "piece"]
    if problem == "shared_visual":
        graph["nodes"][1]["mesh_path"] = "part.obj"
    if problem == "aliased_visual":
        graph["nodes"][1]["mesh_path"] = "./part.obj"
    if problem == "shared_collision":
        other = copy.deepcopy(graph["nodes"][1])
        other.update(id="other", position_m=[0.05, 0, 0.01])
        graph["nodes"].append(other)
        graph["edges"].append(dict(source="other", target="table", relation="on"))
    if problem == "too_deep":
        kwargs["max_depth"] = 0.0001
    if problem == "volume_loss":
        # Thin collision part mostly below visual minimum: reject without deleting it.
        thin = trimesh.creation.box(extents=[0.01, 0.01, 0.0004])
        thin.apply_translation([0, 0, -0.0101])
        thin.export(path.parent / "part.obj")
    path.write_text(json.dumps(graph))
    out = tmp_path / "invalid"
    with pytest.raises(ValueError):
        trim_collision_padding(path, out, names, **kwargs)
    assert not out.exists()
