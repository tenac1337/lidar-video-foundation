"""Small generated export-contract fixtures, not measured reconstruction data."""

import copy
import shutil
from pathlib import Path

from lidar_video.comparison import seal_exports
from lidar_video.creation_audit import read_json, save
from lidar_video.scenic_export import export_scenic, sample_scenic


def generate_comparison_fixtures(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    template = out / "template"
    scene = template / "scene"
    scene.mkdir(parents=True)
    graph = dict(
        units="metres",
        nodes=[
            dict(id="board", kind="support", position_m=[0, 0, -.01], size_m=[.4, .4, .02]),
            dict(id="left", kind="object", position_m=[-.06, 0, .021], size_m=[.03, .03, .04]),
            dict(id="right", kind="object", position_m=[.06, 0, .021], size_m=[.03, .03, .04]),
        ],
        edges=[dict(source=name, relation="on", target="board") for name in ("left", "right")],
    )
    save(scene / "scene_graph.json", graph)
    export_scenic(graph, scene / "scenario.scenic", jitter=.001)
    sample_scenic(scene / "scenario.scenic", graph, scene / "scenic_samples.json", count=1, seed=7)
    seal_exports(scene, template / "seal.json")
    descriptions = {
        "clean": ("valid", "Generated disjoint supported boxes with unchanged exports and seal"),
        "overlap": ("invalid", "Both object centers are deliberately set equal with positive extents"),
        "missing_object": ("invalid", "The required right object is deliberately removed"),
        "sample_outside_program": (
            "invalid", "Saved x=-0.04 lies outside the program's left-object range [-0.061,-0.059]"
        ),
        "sample_metadata": ("invalid", "A saved object receives a semantic field absent from its source"),
        "stale_seal": ("invalid", "Graph bytes change after sealing; integrity fault, not geometry error"),
    }
    cases = []
    for name, (label, reason) in descriptions.items():
        folder = out / name
        shutil.copytree(template, folder)
        candidate = folder / "scene"
        changed = copy.deepcopy(graph)
        if name == "overlap":
            changed["nodes"][1]["position_m"] = changed["nodes"][2]["position_m"].copy()
            save(candidate / "scene_graph.json", changed)
        elif name == "missing_object":
            changed["nodes"].pop()
            save(candidate / "scene_graph.json", changed)
        elif name in ("sample_outside_program", "sample_metadata"):
            record = read_json(candidate / "scenic_samples.json")
            node = next(n for n in record["samples"][0]["nodes"] if n["id"] == "left")
            if name == "sample_outside_program":
                node["position_m"][0] = -.04
            else:
                node["semantic_type"] = "unexpected-field"
            save(candidate / "scenic_samples.json", record)
        elif name == "stale_seal":
            path = candidate / "scene_graph.json"
            path.write_text(path.read_text() + "\n")
        # Non-integrity faults get a matching seal, isolating their intended contract violation.
        if name not in ("clean", "stale_seal"):
            (folder / "seal.json").unlink()
            seal_exports(candidate, folder / "seal.json")
        save(folder / "case.json", dict(
            schema="export-comparison-v1", scene_dir="scene", expected_objects=["left", "right"],
            case_id=name, scene_id="generated-two-box-scene",
            evidence_seal="seal.json",
        ))
        cases.append(dict(
            case_id=name, scene_id="generated-two-box-scene", label=label, label_reason=reason,
            manifest=str(folder / "case.json"), result=str(folder / "comparison/comparison.json"),
        ))
    save(out / "labels.json", dict(
        schema="labeled-export-comparisons-v1", label_scope="generated_export_contract",
        cases=cases,
        limitation="Known software fixtures; not unseen faults or evidence of real-scene performance.",
    ))
    return dict(status="generated", labels=str(out / "labels.json"), cases=cases)
