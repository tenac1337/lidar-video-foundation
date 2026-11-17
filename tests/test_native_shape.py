import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from lidar_video.native_shape import affine_obj, checked_affine, refine_native_assets


def scene(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    visual = trimesh.creation.box([0.02, 0.02, 0.04])
    for name in ["visual", "collision"]:
        visual.export(root / f"{name}.obj")
    graph = dict(
        units="metres",
        nodes=[
            dict(id="table", kind="support", position_m=[0, 0, -0.005], size_m=[0.3, 0.3, 0.01]),
            dict(
                id="piece",
                kind="object",
                position_m=[0, 0, 0.02],
                size_m=[0.02, 0.02, 0.04],
                mass_kg=0.08,
                mesh_path="visual.obj",
                collision_paths=["collision.obj"],
            ),
        ],
        edges=[dict(source="piece", target="table", relation="on")],
        native_template=dict(scene_path="scene.xml", body_bindings={"piece": "piece", "table": "table"}),
    )
    (root / "scene_graph.json").write_text(json.dumps(graph))
    (root / "scene.xml").write_text(
        """<mujoco><asset><mesh name="visual" file="visual.obj"/><mesh name="collision" file="collision.obj"/></asset><worldbody><body name="table" pos="0 0 -.005"><geom type="box" size=".15 .15 .005"/></body><body name="piece" pos="0 0 .02"><freejoint/><geom name="visual" type="mesh" mesh="visual" mass="0" contype="0" conaffinity="0"/><geom name="collision" type="mesh" mesh="collision" mass=".08"/></body></worldbody></mujoco>"""
    )
    return root / "scene_graph.json", graph


def test_export_preserves_edges_and_native_world_geometry(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    path, graph = scene(tmp_path)
    original = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    affine = np.diag([1.1, 1.1, 0.9, 1.0])
    out = tmp_path / "variant"
    result = refine_native_assets(path, out, {"piece": affine})
    assert result["semantic_edges_preserved"] and not result["physics_validated"]
    new = json.loads((out / "scene_graph.json").read_text())
    np.testing.assert_allclose(new["nodes"][1]["position_m"], [0, 0, 0.018])
    np.testing.assert_allclose(new["nodes"][1]["size_m"], [0.022, 0.022, 0.036], atol=1e-9)
    assert new["edges"] == graph["edges"]
    for p in path.parent.iterdir():
        assert p.read_bytes() == original[p.name]
    native = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
    data = mujoco.MjData(native)
    mujoco.mj_forward(native, data)
    assert native.body_mass[native.body("piece").id] == pytest.approx(0.08)
    for name in ["visual", "collision"]:
        mesh = trimesh.load(out / f"{name}.obj", force="mesh", process=False)
        np.testing.assert_allclose(
            mesh.bounds + new["nodes"][1]["position_m"],
            [[-0.011, -0.011, 0], [0.011, 0.011, 0.036]],
            atol=1e-9,
        )
    with pytest.raises(FileExistsError):
        refine_native_assets(path, out, {"piece": affine})


def test_obj_faces_uvs_and_materials_survive_affine():
    text = "mtllib surface.mtl\nv 1 2 3\nvt .2 .3\nvn 1 1 0\nusemtl paint\nf 1/1/1 1/1/1 1/1/1\n"
    result = affine_obj(text, np.diag([2, 1, 1, 1]))
    for line in ["mtllib surface.mtl", "vt .2 .3", "usemtl paint", "f 1/1/1 1/1/1 1/1/1"]:
        assert line in result
    normal = np.array(result.split("vn ")[1].splitlines()[0].split(), float)
    np.testing.assert_allclose(normal, np.array([0.5, 1, 0]) / np.linalg.norm([0.5, 1, 0]))
    assert "v 2 2 3\n" in result


@pytest.mark.parametrize("bad", ["height", "shear", "scale", "binding", "shared"])
def test_invalid_changes_never_publish_an_output(tmp_path, bad):
    path, graph = scene(tmp_path)
    a = np.eye(4)
    if bad == "height":
        a[2, 3] = 0.001
    if bad == "shear":
        a[0, 1] = 0.1
    if bad == "scale":
        a[0, 0] = a[1, 1] = 1.5
    if bad == "binding":
        graph["native_template"]["body_bindings"]["piece"] = "missing"
        path.write_text(json.dumps(graph))
    if bad == "shared":
        tree = ET.parse(path.parent / "scene.xml")
        ET.SubElement(tree.find("worldbody"), "geom", name="duplicate", type="mesh", mesh="visual")
        tree.write(path.parent / "scene.xml")
    out = tmp_path / "rejected"
    with pytest.raises(ValueError):
        refine_native_assets(path, out, {"piece": a})
    assert not out.exists()


def test_reflection_and_nonfinite_matrices_rejected():
    for a in [np.diag([-1, 1, 1, 1]), np.full((4, 4), np.nan)]:
        with pytest.raises(ValueError):
            checked_affine(a)


def test_same_file_under_another_mesh_name_is_rejected(tmp_path):
    path, _ = scene(tmp_path)
    tree = ET.parse(path.parent / "scene.xml")
    ET.SubElement(tree.find("asset"), "mesh", name="alias", file="visual.obj")
    ET.SubElement(tree.find("worldbody"), "geom", name="alias", type="mesh", mesh="alias")
    tree.write(path.parent / "scene.xml")
    with pytest.raises(ValueError, match="Shared native asset"):
        refine_native_assets(path, tmp_path / "rejected", {"piece": np.eye(4)})
    assert not (tmp_path / "rejected").exists()
