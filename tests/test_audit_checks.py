"""Audit verdicts must be grounded in executable artifacts, never self-reports."""

import copy
import json
from pathlib import Path

import pytest

from lidar_video.audit_checks import _link_aligned_mesh, run_checks, validate_graph
from lidar_video.scenic_export import export_scenic


@pytest.fixture
def scene(tmp_path):
    graph = dict(
        units="metres",
        nodes=[
            dict(id="board", kind="support", position_m=[0, 0, -0.01], size_m=[0.4, 0.4, 0.02]),
            dict(id="left", kind="object", position_m=[-0.06, 0, 0.021], size_m=[0.03, 0.03, 0.04]),
            dict(id="right", kind="object", position_m=[0.06, 0, 0.021], size_m=[0.03, 0.03, 0.04]),
        ],
        edges=[dict(source=x, relation="on", target="board") for x in ("left", "right")],
    )
    folder = tmp_path / "scene"
    folder.mkdir()
    (folder / "scene_graph.json").write_text(json.dumps(graph))
    (folder / "scenic_samples.json").write_text(json.dumps(dict(samples=[graph])))
    export_scenic(graph, folder / "scenario.scenic", jitter=0.001)
    (folder / "simulation").mkdir()
    (folder / "simulation" / "scene.xml").write_text(
        '<mujoco><default><geom friction="0.8 0.005 0.0001"/></default>'
        '<option timestep="0.002" gravity="0 0 -9.81"/>'
        '<worldbody><body name="board" pos="0 0 -0.01">'
        '<geom type="box" size="0.2 0.2 0.01"/></body>'
        '<body name="left" pos="-0.06 0 0.021"><freejoint/>'
        '<geom type="box" size="0.015 0.015 0.02" mass="0.08"/></body>'
        '<body name="right" pos="0.06 0 0.021"><freejoint/>'
        '<geom type="box" size="0.015 0.015 0.02" mass="0.08"/></body>'
        "</worldbody></mujoco>"
    )
    case = dict(
        schema="creation-audit-case-v1",
        geometry_scope="synthetic_fixture",
        scene_dir="scene",
        expected_objects=["left", "right"],
    )
    return folder, graph, case


def test_positive_fixture_runs_actual_physics_and_fresh_scenic(scene, tmp_path):
    _, _, case = scene
    result = run_checks(case, tmp_path, tmp_path / "audit")
    assert {role: record["status"] for role, record in result.items()} == dict.fromkeys(
        ("geometry", "physics", "exports", "adversarial"), "pass"
    )
    assert result["physics"]["findings"][0]["evidence"]["steps"] == 500
    assert (tmp_path / "audit/fresh_scenic_samples.json").is_file()
    probes = result["adversarial"]["findings"][0]["evidence"]
    assert len(probes) == 4 and all(p["rejected"] for p in probes.values())
    assert "overlap" in probes["overlapping_objects"]["reason"]


def test_missing_scene_and_manifest_are_blocked(tmp_path):
    result = run_checks(
        dict(schema="creation-audit-case-v1", expected_objects=["f8"]), tmp_path, tmp_path / "audit"
    )
    assert all(r["status"] == "blocked" for r in result.values())


def test_forged_success_and_empty_graph_cannot_pass(scene, tmp_path):
    folder, _, case = scene
    (folder / "scene_graph.json").write_text(json.dumps(dict(units="metres", nodes=[], edges=[])))
    (folder / "verification.json").write_text(json.dumps(dict(passed=True)))
    result = run_checks(case, tmp_path, tmp_path / "audit")
    assert all(r["status"] == "fail" for r in result.values())


def test_pinned_objects_cannot_pass_physics(scene, tmp_path):
    folder, _, case = scene
    path = folder / "simulation/scene.xml"
    path.write_text(path.read_text().replace("<freejoint/>", ""))
    result = run_checks(case, tmp_path, tmp_path / "audit")
    assert result["physics"]["status"] == "fail"
    assert "free joint" in result["physics"]["findings"][0]["summary"]


def test_broken_export_sizes_cannot_pass(scene, tmp_path):
    folder, graph, case = scene
    graph["nodes"][1]["size_m"][0] = 0.02
    (folder / "scenic_samples.json").write_text(json.dumps(dict(samples=[graph])))
    result = run_checks(case, tmp_path, tmp_path / "audit")
    assert result["exports"]["status"] == "fail"
    assert "dimensions" in result["exports"]["findings"][0]["summary"]


def test_bad_scenic_program_is_recompiled_and_rejected(scene, tmp_path):
    folder, _, case = scene
    (folder / "scenario.scenic").write_text("this cannot compile @@@")
    result = run_checks(case, tmp_path, tmp_path / "audit")
    assert result["exports"]["status"] == "fail"
    assert (tmp_path / "audit/scenic_compile.log").stat().st_size > 0


def test_duplicate_raw_scenic_object_cannot_disappear_from_export(scene, tmp_path):
    from lidar_video.scenic_export import sample_scenic

    folder, graph, case = scene
    program = folder / "scenario.scenic"
    # The duplicate comes first, so a lossy dict comprehension would hide it
    # behind the valid original object and still produce a plausible graph.
    ghost = (
        "ghost = new Object at (-0.15, -0.15, 0.03), with width 0.01, "
        "with length 0.01, with height 0.02, with requireVisible False, with objectId 'left'\n"
    )
    program.write_text(ghost + program.read_text())
    target = tmp_path / "duplicate_samples.json"
    with pytest.raises(ValueError, match="IDs must be unique"):
        sample_scenic(program, graph, target, count=1)
    assert not target.exists()
    result = run_checks(case, tmp_path, tmp_path / "audit")
    assert result["exports"]["status"] == "fail"
    assert "IDs must be unique" in (tmp_path / "audit/scenic_compile.log").read_text()


@pytest.mark.parametrize("entries", [{}, {"f8": {}}, {"f8": {"validation": "legacy_square_cloud"}}])
def test_weak_or_incomplete_mesh_contract_is_rejected(entries, tmp_path):
    (tmp_path / "meshes.json").write_text(
        json.dumps(dict(schema="roboskills-chess-meshes-v1", objects=entries))
    )
    case = dict(schema="creation-audit-case-v1", mesh_manifest="meshes.json", expected_objects=["f8"])
    result = run_checks(case, tmp_path, tmp_path / "audit")
    assert result["geometry"]["status"] == "fail"
    assert result["physics"]["status"] == "blocked"


def test_failed_fixed_mesh_gate_is_not_replaced_by_fallback(tmp_path, monkeypatch):
    from lidar_video import reconstruction

    fixtures = Path(__file__).parent / "fixtures/chess-demo"
    entries = {"f8": dict(validation="instance-multiview-v1"), "b8": dict(validation="instance-multiview-v1")}
    (tmp_path / "meshes.json").write_text(
        json.dumps(dict(schema="roboskills-chess-meshes-v1", objects=entries))
    )
    visited = []

    def reject(entry, root, data, calibration, observed):
        visited.append(entry["square"])
        raise reconstruction.MeshQualityError("silhouette rejected", evidence=dict(silhouette_iou=0.2))

    monkeypatch.setattr(reconstruction, "import_learned_mesh", reject)
    case = dict(
        schema="creation-audit-case-v1",
        mesh_manifest="meshes.json",
        capture=str((fixtures / "capture.npz").resolve()),
        calibration=str((fixtures / "board_calibration.json").resolve()),
        expected_objects=["f8", "b8"],
    )
    result = run_checks(case, tmp_path, tmp_path / "audit")
    assert visited == ["f8", "b8"]
    assert result["geometry"]["status"] == "fail"
    assert all(f["evidence"]["silhouette_iou"] == 0.2 for f in result["geometry"]["findings"])


def test_graph_validator_rejects_wrong_support_relation_and_duplicate_id(scene):
    _, graph, _ = scene
    bad = copy.deepcopy(graph)
    bad["edges"][0]["target"] = "right"
    with pytest.raises(ValueError, match="support relation"):
        validate_graph(bad, ["left", "right"])
    bad = copy.deepcopy(graph)
    bad["nodes"][2]["id"] = "left"
    with pytest.raises(ValueError, match="unique stable IDs"):
        validate_graph(bad, ["left", "right"])


def test_synthetic_scope_cannot_launder_a_capture_claim(scene, tmp_path):
    _, _, case = scene
    case["capture"] = "capture.npz"
    result = run_checks(case, tmp_path, tmp_path / "audit")
    assert result["geometry"]["status"] == "fail"


@pytest.mark.parametrize("change", ["mass", "friction", "equality", "gravity"])
def test_physics_rejects_changed_priors_or_hidden_pinning(scene, tmp_path, change):
    folder, _, case = scene
    path = folder / "simulation/scene.xml"
    text = path.read_text()
    if change == "mass":
        text = text.replace('mass="0.08"', 'mass="0.16"')
    elif change == "friction":
        text = text.replace('friction="0.8 0.005 0.0001"', 'friction="8 0.005 0.0001"')
    elif change == "equality":
        text = text.replace("</mujoco>", '<equality><weld body1="left"/></equality></mujoco>')
    else:
        text = text.replace('name="left" pos=', 'name="left" gravcomp="1" pos=')
    path.write_text(text)
    result = run_checks(case, tmp_path, tmp_path / "audit")
    assert result["physics"]["status"] == "fail"


def test_export_link_binds_exact_aligned_mesh_source_and_graph_position(tmp_path):
    import trimesh

    folder = tmp_path / "assets/piece_f8"
    folder.mkdir(parents=True)
    mesh = trimesh.creation.box(extents=[0.02, 0.03, 0.04])
    mesh.export(folder / "mesh.ply")
    aligned = mesh.copy()
    aligned.apply_translation([0.1, 0, 0.02])
    source = str(tmp_path / "learned.ply")
    evidence = dict(source=source)
    node = dict(
        mesh_path="assets/piece_f8/visual.obj",
        position_m=[0.1, 0, 0.02],
        evidence=dict(learned_mesh=evidence),
    )
    assert _link_aligned_mesh(aligned, evidence, node, tmp_path)["max_vertex_difference_m"] < 1e-6
    bad = copy.deepcopy(node)
    bad["position_m"][0] += 0.001
    with pytest.raises(ValueError, match="differs from"):
        _link_aligned_mesh(aligned, evidence, bad, tmp_path)
    bad = copy.deepcopy(node)
    bad["evidence"]["learned_mesh"]["source"] = "unrelated.ply"
    with pytest.raises(ValueError, match="source"):
        _link_aligned_mesh(aligned, evidence, bad, tmp_path)
    unrelated = mesh.copy()
    unrelated.vertices[0, 0] += 0.002
    unrelated.export(folder / "mesh.ply")
    with pytest.raises(ValueError, match="differs from"):
        _link_aligned_mesh(aligned, evidence, node, tmp_path)


@pytest.mark.parametrize("mutation", ["translation", "rotation", "extra_collider", "extra_world_geom"])
def test_physics_rejects_primitive_collision_transform_and_inventory_bypass(scene, tmp_path, mutation):
    import xml.etree.ElementTree as ET

    folder, _, case = scene
    path = folder / "simulation/scene.xml"
    tree = ET.parse(path)
    support = tree.getroot().find("worldbody/body[@name='board']")
    geom = support.find("geom")
    if mutation == "translation":
        geom.set("pos", "0.05 0 0")
    elif mutation == "rotation":
        geom.set("euler", "0 0 10")
    elif mutation == "extra_collider":
        ET.SubElement(support, "geom", type="sphere", size="0.01", pos="0.15 0.15 0")
    else:
        ET.SubElement(tree.getroot().find("worldbody"), "geom", type="sphere", size="0.01", pos="1 1 0")
    tree.write(path)
    result = run_checks(case, tmp_path, tmp_path / "audit")
    assert result["physics"]["status"] == "fail"
    summary = result["physics"]["findings"][0]["summary"]
    assert any(word in summary for word in ("position", "orientation", "additional", "absent"))


@pytest.mark.parametrize("mutation", [None, "visual_shift", "graph_surface_change"])
def test_compiled_visual_and_collision_meshes_match_graph_assets(scene, tmp_path, mutation):
    import xml.etree.ElementTree as ET

    import trimesh

    folder, graph, case = scene
    (folder / "assets").mkdir()
    (folder / "simulation/assets").mkdir()
    mesh = trimesh.creation.box(extents=[0.03, 0.03, 0.04])
    mesh.export(folder / "assets/left.obj")
    mesh.export(folder / "simulation/assets/left.obj")
    graph["nodes"][1].update(mesh_path="assets/left.obj", collision_paths=["assets/left.obj"])
    (folder / "scene_graph.json").write_text(json.dumps(graph))
    path = folder / "simulation/scene.xml"
    tree = ET.parse(path)
    asset = ET.SubElement(tree.getroot(), "asset")
    ET.SubElement(asset, "mesh", name="left_mesh", file="assets/left.obj")
    left = tree.getroot().find("worldbody/body[@name='left']")
    left.remove(left.find("geom"))
    visual = ET.SubElement(
        left, "geom", type="mesh", mesh="left_mesh", mass="0", contype="0", conaffinity="0"
    )
    ET.SubElement(left, "geom", type="mesh", mesh="left_mesh", mass="0.08")
    if mutation == "visual_shift":
        visual.set("pos", "0.002 0 0")
    elif mutation == "graph_surface_change":
        mesh.vertices[0, 0] += 0.002
        mesh.export(folder / "assets/left.obj")
    tree.write(path)
    result = run_checks(case, tmp_path, tmp_path / "audit")
    assert result["physics"]["status"] == ("pass" if mutation is None else "fail")
    if mutation:
        assert "surface disagrees" in result["physics"]["findings"][0]["summary"]


@pytest.mark.parametrize(
    "mutation", ["swap_square", "asset_path", "symbol", "evidence", "mass", "extra_field"]
)
def test_saved_scenic_samples_bind_metadata_to_each_stable_id(scene, tmp_path, mutation):
    folder, graph, case = scene
    graph["nodes"][1].update(square="f3", symbol="P", evidence={"source": "fixture"}, mass_kg=0.08)
    graph["nodes"][2].update(square="d5", symbol="K", evidence={"source": "fixture"}, mass_kg=0.08)
    case["expected_objects"] = ["f3", "d5"]
    (folder / "scene_graph.json").write_text(json.dumps(graph))
    sample = copy.deepcopy(graph)
    if mutation == "swap_square":
        sample["nodes"][1]["square"], sample["nodes"][2]["square"] = "d5", "f3"
    elif mutation == "asset_path":
        sample["nodes"][1]["mesh_path"] = "unrelated.obj"
    elif mutation == "symbol":
        sample["nodes"][1]["symbol"] = "K"
    elif mutation == "evidence":
        sample["nodes"][1]["evidence"] = {"source": "fabricated"}
    elif mutation == "mass":
        sample["nodes"][1]["mass_kg"] = 0.16
    else:
        sample["nodes"][1]["new_semantic_claim"] = "recognized"
    (folder / "scenic_samples.json").write_text(json.dumps(dict(samples=[sample])))
    result = run_checks(case, tmp_path, tmp_path / "audit")
    assert result["geometry"]["status"] == "pass"
    assert result["physics"]["status"] == "pass"
    assert result["exports"]["status"] == "fail"
    assert "per-ID metadata" in result["exports"]["findings"][0]["summary"]


def test_rich_node_metadata_survives_fresh_scenic_sampling_with_position_variation(scene, tmp_path):
    folder, graph, case = scene
    graph["nodes"][1].update(square="f3", symbol="P", evidence={"source": "fixture"}, mass_kg=0.08)
    graph["nodes"][2].update(square="d5", symbol="K", evidence={"source": "fixture"}, mass_kg=0.08)
    case["expected_objects"] = ["f3", "d5"]
    (folder / "scene_graph.json").write_text(json.dumps(graph))
    (folder / "scenic_samples.json").write_text(json.dumps(dict(samples=[graph])))
    result = run_checks(case, tmp_path, tmp_path / "audit")
    assert result["exports"]["status"] == "pass"
    fresh = json.loads((tmp_path / "audit/fresh_scenic_samples.json").read_text())["samples"][0]
    assert fresh["nodes"][1]["position_m"] != graph["nodes"][1]["position_m"]
    assert fresh["nodes"][1]["evidence"] == graph["nodes"][1]["evidence"]
    assert fresh["nodes"][1]["square"] == "f3"


@pytest.mark.parametrize(
    "mutation", ["occupancy", "missing_relation", "duplicate", "edge_evidence", "frame", None]
)
def test_samples_preserve_relationships_and_coordinate_metadata(scene, tmp_path, mutation):
    folder, graph, case = scene
    graph["frame"] = "fixture_frame"
    graph["nodes"].extend(
        dict(id=name, kind="square", position_m=[0, 0, 0]) for name in ("f3", "a1")
    )
    graph["edges"].extend(
        [
            dict(source="left", relation="occupies", target="f3", evidence="fixture"),
            dict(source="f3", relation="part_of", target="board"),
            dict(source="a1", relation="part_of", target="board"),
        ]
    )
    (folder / "scene_graph.json").write_text(json.dumps(graph))
    sample = copy.deepcopy(graph)
    if mutation == "occupancy":
        sample["edges"][2]["target"] = "a1"
    elif mutation == "missing_relation":
        sample["edges"] = [e for e in sample["edges"] if e["relation"] != "part_of"]
    elif mutation == "duplicate":
        sample["edges"].append(copy.deepcopy(sample["edges"][2]))
    elif mutation == "edge_evidence":
        sample["edges"][2]["evidence"] = "fabricated"
    elif mutation == "frame":
        sample["frame"] = "unrelated_frame"
    else:
        sample["edges"].reverse()
        for edge in sample["edges"]:
            if edge["relation"] == "on":
                edge["evidence"] = "sampled_constraint"
    (folder / "scenic_samples.json").write_text(json.dumps(dict(samples=[sample])))
    result = run_checks(case, tmp_path, tmp_path / "audit")
    assert result["geometry"]["status"] == "pass"
    assert result["physics"]["status"] == "pass"
    assert result["exports"]["status"] == ("pass" if mutation is None else "fail")
    if mutation:
        assert any(
            word in result["exports"]["findings"][0]["summary"]
            for word in ("relationships", "graph metadata")
        )
