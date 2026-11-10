import copy

import numpy as np
import pytest

from lidar_video.scene_program import as_networkx, compile_scene_program, export_constructive_scenic


def node(name, pos, size, **kwargs):
    return dict(id=name, kind="object", position_m=pos, size_m=size, **kwargs)


def stacked():
    return dict(
        nodes=[
            node(
                "table",
                [0, 0, -0.05],
                [1, 1, 0.1],
                fixed=False,
                placement_bounds_m=[[-0.2, -0.2, -0.05], [0.2, 0.2, -0.05]],
            ),
            node(
                "tray",
                [-0.15, 0, 0.05],
                [0.4, 0.4, 0.1],
                placement_bounds_m=[[-0.4, -0.4, 0.05], [0.4, 0.4, 0.05]],
            ),
            node(
                "cup",
                [-0.15, 0, 0.15],
                [0.1, 0.1, 0.1],
                placement_bounds_m=[[-0.5, -0.5, 0.15], [0.5, 0.5, 0.15]],
            ),
            node("arm_base", [0.3, 0, 0.1], [0.1, 0.1, 0.2], fixed=True),
        ],
        edges=[
            dict(source="cup", target="tray", relation="on"),
            dict(source="tray", target="table", relation="on"),
            dict(source="arm_base", target="table", relation="on"),
            dict(source="tray", target="arm_base", relation="left_of", gap_m=0.01),
        ],
    )


def test_graph_preserves_parallel_relations_and_evidence():
    graph = stacked()
    graph["edges"].append(dict(source="tray", target="table", relation="above", evidence="annotation"))
    for edge in graph["edges"]:
        edge["key"] = "shared-metadata-key"
    net = as_networkx(graph)
    assert net.number_of_edges("tray", "table") == 2
    net.nodes["tray"]["position_m"][0] = 99
    assert graph["nodes"][1]["position_m"][0] == -0.15


def test_replay_is_exact_with_no_clearance_or_pose_repair():
    graph = stacked()
    program = compile_scene_program(graph)
    replay = program.sample(randomize=False)
    assert replay["nodes"] == graph["nodes"]
    assert replay["edges"] == graph["edges"]
    assert program.order.index("table") < program.order.index("tray") < program.order.index("cup")


def test_randomized_hierarchy_preserves_relations_and_is_reproducible():
    program = compile_scene_program(stacked())
    values = []
    for seed in range(30):
        graph = program.sample(seed=seed)
        nodes = {n["id"]: n for n in graph["nodes"]}
        for a, b in [("cup", "tray"), ("tray", "table"), ("arm_base", "table")]:
            child, parent = nodes[a], nodes[b]
            p, c = np.array(parent["position_m"]), np.array(child["position_m"])
            ps, cs = np.array(parent["size_m"]), np.array(child["size_m"])
            assert np.all(np.abs(c[:2] - p[:2]) + cs[:2] / 2 <= ps[:2] / 2 + 1e-8)
            assert abs((c[2] - cs[2] / 2) - (p[2] + ps[2] / 2)) < 1e-8
        assert nodes["tray"]["position_m"][0] + 0.2 + 0.01 <= nodes["arm_base"]["position_m"][0] - 0.05 + 1e-8
        assert graph["scene_program"]["sampling_attempts"] == 1
        values.append(nodes["cup"]["position_m"])
    assert np.ptp(values, axis=0)[:2].min() > 0.05
    assert program.sample(seed=27) == program.sample(seed=27)


def test_future_feasibility_prevents_greedy_dead_end():
    graph = dict(
        nodes=[
            node("a", [0, 0, 0], [0.4, 0.2, 0.2], placement_bounds_m=[[0, 0, 0], [0.8, 0, 0]]),
            node("b", [0.4, 0, 0], [0.4, 0.2, 0.2], fixed=True),
        ],
        edges=[dict(source="a", target="b", relation="left_of")],
    )
    program = compile_scene_program(graph)
    for seed in range(10):
        assert abs(program.sample(seed=seed)["nodes"][0]["position_m"][0]) < 1e-8


def test_unsupported_root_and_inside_container():
    graph = dict(
        nodes=[
            node("container", [0, 0, 1], [2, 2, 2], fixed=False),
            node("part", [0, 0, 1], [0.1, 0.1, 0.1]),
        ],
        edges=[dict(source="part", target="container", relation="inside")],
    )
    program = compile_scene_program(graph, jitter=0.5)
    assert program.order == ("container", "part")
    assert program.sample(seed=4)["nodes"][0]["position_m"] != [0, 0, 1]


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda g: g["edges"].append(dict(source="table", target="cup", relation="on")), "cycle"),
        (lambda g: g["edges"].append(dict(source="cup", target="missing", relation="on")), "endpoint"),
        (lambda g: g["nodes"].append(copy.deepcopy(g["nodes"][0])), "unique"),
        (lambda g: g["edges"].append(dict(source="cup", target="tray", relation="near")), "Unsupported"),
        (lambda g: g["nodes"][1].update(heading=0.2), "Rotation"),
        (lambda g: g["edges"][0].update(clearance_m=-0.1), "nonnegative"),
        (lambda g: g["edges"].append(dict(source="cup", target="table", relation="on")), "Multiple"),
        (lambda g: g["edges"][-1].update(gap_m=10), "infeasible"),
        (lambda g: g["nodes"][2].update(position_m=[-0.15, 0, 0.17]), "declared domain"),
    ],
)
def test_invalid_contracts_fail_before_sampling(change, match):
    graph = stacked()
    change(graph)
    with pytest.raises(ValueError, match=match):
        compile_scene_program(graph)


def test_scenic_generation_uses_one_attempt_and_correlated_positions(tmp_path):
    import scenic

    for randomize in (False, True):
        path = tmp_path / f"scene-{randomize}.scenic"
        graph = stacked()
        program = export_constructive_scenic(graph, path, randomize=randomize)
        assert "\nrequire " not in path.read_text()
        scenario = scenic.scenarioFromFile(str(path))
        for _ in range(5):
            scene, attempts = scenario.generate(maxIterations=1)
            assert attempts == 1
            by_id = {obj.objectId: obj for obj in scene.objects}
            values = [by_id[name].position for name in program.order]
            program.validate_positions(values)
            if not randomize:
                expected = {n["id"]: n["position_m"] for n in graph["nodes"]}
                assert all(list(by_id[name].position) == expected[name] for name in program.order)


def test_command_executes_emitted_program_and_preserves_rng(tmp_path, monkeypatch, capsys):
    import json
    import random
    import sys

    from lidar_video.cli import main
    from lidar_video.scene_program import run_scene_program

    graph = stacked()
    path = tmp_path / "input.json"
    path.write_text(json.dumps(graph))
    monkeypatch.setattr(
        sys,
        "argv",
        ["lidar-video", "scene-program", str(path), "--out", str(tmp_path / "replay"), "--samples", "2"],
    )
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["randomized"] is False and report["attempts"] == [1, 1]
    original_rng = random.getstate()
    run_scene_program(graph, tmp_path / "random", randomize=True, count=3)
    assert random.getstate() == original_rng
    saved = json.loads((tmp_path / "replay/samples.json").read_text())
    assert saved["samples"][0]["nodes"] == graph["nodes"]


@pytest.mark.parametrize(
    "relation,position,axis,sign",
    [
        ("left_of", [-1, 0, 0], 0, -1),
        ("right_of", [1, 0, 0], 0, 1),
        ("behind", [0, -1, 0], 1, -1),
        ("in_front_of", [0, 1, 0], 1, 1),
        ("below", [0, 0, -1], 2, -1),
        ("above", [0, 0, 1], 2, 1),
    ],
)
def test_directional_constraints_use_faces(relation, position, axis, sign):
    graph = dict(
        nodes=[node("a", position, [0.2] * 3), node("b", [0, 0, 0], [0.2] * 3)],
        edges=[dict(source="a", target="b", relation=relation, gap_m=0.3)],
    )
    program = compile_scene_program(graph, jitter=1)
    for seed in range(5):
        a, b = program.sample(seed=seed)["nodes"]
        assert sign * (a["position_m"][axis] - b["position_m"][axis]) >= 0.5 - 1e-8


def test_inside_margin_and_invalid_recorded_support():
    graph = dict(
        nodes=[node("box", [0, 0, 0], [1] * 3, fixed=True), node("part", [0, 0, 0], [0.2] * 3)],
        edges=[dict(source="part", target="box", relation="inside", gap_m=0.1)],
    )
    program = compile_scene_program(graph, jitter=1)
    for seed in range(5):
        assert np.max(np.abs(program.sample(seed=seed)["nodes"][1]["position_m"])) <= 0.3 + 1e-8
    graph = stacked()
    graph["nodes"][2]["position_m"][2] = 0.17
    graph["nodes"][2]["placement_bounds_m"][1][2] = 0.2
    with pytest.raises(ValueError, match="support constraint"):
        compile_scene_program(graph)
