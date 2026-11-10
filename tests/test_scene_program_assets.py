import copy
import json

import numpy as np
import pytest
import trimesh

from lidar_video.scene_program import compile_scene_program
from lidar_video.scene_program_assets import prepare_asset_graph, run_asset_scene_program


def scene(tmp_path, penetration=0):
    assets = tmp_path / "source" / "assets"
    assets.mkdir(parents=True)
    mesh = trimesh.creation.box(extents=[0.08, 0.08, 0.1])
    mesh.export(assets / "piece.obj")
    graph = dict(
        units="metres",
        time_s=0,
        nodes=[
            dict(
                id="board",
                kind="support",
                position_m=[0, 0, -0.05],
                size_m=[1, 1, 0.1],
                square_size_m=0.2,
                color_rgb=[0.5] * 3,
            ),
            dict(id="s1", kind="square", position_m=[0, 0, 0]),
            dict(
                id="piece",
                kind="object",
                position_m=[0, 0, 0.05 - penetration],
                size_m=[0.08, 0.08, 0.1],
                square="s1",
                square_bounds_m=[[-0.1, -0.1], [0.1, 0.1]],
                color_rgb=[0.5] * 3,
                mesh_path="assets/piece.obj",
                collision_paths=["assets/piece.obj"],
            ),
        ],
        edges=[
            dict(source="s1", target="board", relation="part_of"),
            dict(source="piece", target="s1", relation="occupies"),
            dict(source="piece", target="board", relation="on"),
        ],
    )
    path = assets.parent / "scene_graph.json"
    path.write_text(json.dumps(graph))
    return path, graph


def test_exact_replay_preserves_full_graph_and_asset_bytes(tmp_path):
    path, graph = scene(tmp_path)
    result = run_asset_scene_program(path, tmp_path / "replay", count=1)
    replay = json.loads((tmp_path / "replay/scene_graph_000.json").read_text())
    assert replay["nodes"] == graph["nodes"]
    assert replay["edges"] == graph["edges"]
    assert result["replay_nodes_edges_exact"]
    assert (tmp_path / "replay/assets/piece.obj").read_bytes() == (
        path.parent / "assets/piece.obj"
    ).read_bytes()
    text = (tmp_path / "replay/program/scene.scenic").read_text()
    assert "meshPath" in text and "collisionPaths" in text
    assert result["attempts"] == [1]


def test_randomized_samples_preserve_semantics_and_collision_envelope(tmp_path):
    path, graph = scene(tmp_path)
    result = run_asset_scene_program(path, tmp_path / "random", count=15, randomize=True, jitter=0.2)
    samples = json.loads((tmp_path / "random/samples.json").read_text())["samples"]
    for sample in samples:
        assert len(sample["nodes"]) == 3 and len(sample["edges"]) == 3
        assert sample["nodes"][0] == graph["nodes"][0]
        assert sample["nodes"][1] == graph["nodes"][1]
        item = sample["nodes"][2]
        assert np.max(np.abs(item["position_m"][:2])) <= 0.06 + 1e-8
        assert item["position_m"][2] == pytest.approx(0.05)
        assert item["mesh_path"] == graph["nodes"][2]["mesh_path"]
        assert item["size_m"] == graph["nodes"][2]["size_m"]
    assert result["attempts"] == [1] * 15
    assert np.ptp([s["nodes"][2]["position_m"][0] for s in samples]) > 0.02


def test_existing_support_overlap_requires_explicit_budget_no_pose_repair(tmp_path):
    path, graph = scene(tmp_path, penetration=0.0004)
    with pytest.raises(ValueError, match="support tolerance"):
        prepare_asset_graph(graph, path.parent)
    proxy, report = prepare_asset_graph(graph, path.parent, support_tolerance=0.001)
    assert proxy["nodes"][1]["position_m"] == graph["nodes"][2]["position_m"]
    assert proxy["edges"][0]["support_penetration_m"] == pytest.approx(0.0004)
    assert report["support_contacts"][0]["penetration_m"] == pytest.approx(0.0004)
    program = compile_scene_program(proxy)
    for seed in range(10):
        sampled = program.sample(seed=seed)
        assert sampled["nodes"][1]["position_m"][2] == pytest.approx(0.0496)
    with pytest.raises(ValueError):
        prepare_asset_graph(graph, path.parent, support_tolerance=0.0001)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_relation",
        "missing_part_of",
        "wrong_square",
        "undersized_proxy",
        "escaped_asset",
        "missing_collision",
        "missing_occupies",
        "moving_region",
    ],
)
def test_invalid_reconstruction_is_rejected(tmp_path, mutation):
    path, original = scene(tmp_path)
    graph = copy.deepcopy(original)
    item = graph["nodes"][2]
    if mutation == "unknown_relation":
        graph["edges"].append(dict(source="piece", target="board", relation="near"))
    elif mutation == "missing_part_of":
        graph["edges"] = graph["edges"][1:]
    elif mutation == "wrong_square":
        item["square"] = "s2"
    elif mutation == "undersized_proxy":
        item["size_m"] = [0.04, 0.04, 0.1]
    elif mutation == "escaped_asset":
        item["mesh_path"] = "../outside.obj"
    elif mutation == "missing_collision":
        item["collision_paths"] = []
    elif mutation == "missing_occupies":
        graph["edges"] = [e for e in graph["edges"] if e["relation"] != "occupies"]
    elif mutation == "moving_region":
        graph["nodes"][0]["fixed"] = False
    with pytest.raises(ValueError):
        prepare_asset_graph(graph, path.parent)


def test_penetration_cannot_disable_unrelated_nonoverlap(tmp_path):
    path, graph = scene(tmp_path, penetration=0.0004)
    graph["nodes"].append(
        dict(id="other", kind="object", position_m=[0.01, 0, 0.05], size_m=[0.08, 0.08, 0.1])
    )
    graph["edges"].append(dict(source="other", target="board", relation="on"))
    with pytest.raises(ValueError, match="overlap"):
        prepare_asset_graph(graph, path.parent, support_tolerance=0.001)


def test_rejects_invalid_regions_and_bound_arrays(tmp_path):
    path, graph = scene(tmp_path)
    graph["nodes"][2]["placement_bounds_m"] = [[0, 0], [1, 1]]
    with pytest.raises(ValueError, match="placement bounds"):
        prepare_asset_graph(graph, path.parent)


def test_nested_support_is_not_implicitly_fixed_for_square_regions(tmp_path):
    path, graph = scene(tmp_path)
    graph["nodes"].append(dict(id="table", kind="support", position_m=[0, 0, -0.2], size_m=[2, 2, 0.2]))
    graph["edges"].append(dict(source="board", target="table", relation="on"))
    with pytest.raises(ValueError, match="fixed support"):
        prepare_asset_graph(graph, path.parent)
    graph["nodes"][0]["fixed"] = True
    prepare_asset_graph(graph, path.parent)


def test_physical_ids_cannot_escape_native_asset_directory(tmp_path):
    path, graph = scene(tmp_path)
    graph["nodes"][2]["id"] = "../piece"
    for edge in graph["edges"]:
        if edge["source"] == "piece":
            edge["source"] = "../piece"
    with pytest.raises(ValueError, match="filename"):
        prepare_asset_graph(graph, path.parent)


@pytest.mark.parametrize("penetration", [-0.001, float("nan"), 0.2])
def test_invalid_explicit_support_penetration_rejected(tmp_path, penetration):
    path, graph = scene(tmp_path)
    graph["edges"][-1]["support_penetration_m"] = penetration
    with pytest.raises(ValueError):
        compile_scene_program(dict(nodes=[graph["nodes"][0], graph["nodes"][2]], edges=[graph["edges"][-1]]))
