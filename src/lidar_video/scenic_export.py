"""Graph -> executable Scenic -> sampled graphs, preserving stable object IDs."""

import copy
import random
from pathlib import Path

import numpy as np

from lidar_video.capture import write_json


def export_scenic(graph, path, jitter=0.004):
    if not np.isfinite(jitter) or jitter < 0:
        raise ValueError("jitter must be a finite, nonnegative distance")
    nodes = [n for n in graph["nodes"] if n["kind"] in ("support", "object")]
    support = next(n for n in nodes if n["kind"] == "support")
    center, size = list(map(float, support["position_m"])), list(map(float, support["size_m"]))
    top = center[2] + size[2] / 2
    lines = [
        "# Generated from a reconstructed scene graph; metres, z up.",
        "# Jitter is deliberate scenario variation, NOT estimated sensor uncertainty.",
        "# Box proxies and fixed support; no chess legality or dynamic behaviors.",
        "# 1 mm initial clearance allows the physics solver to establish contact.",
    ]
    for i, node in enumerate(nodes):
        name = f"obj_{i}"
        x, y, z = map(float, node["position_m"])
        w, length, height = map(float, node["size_m"])
        if min(w, length, height) <= 0:
            raise ValueError("Object dimensions must be positive")
        if node["kind"] == "support":
            position = repr(tuple(float(v) for v in (x, y, z)))
        else:
            lower = np.array(center[:2]) - np.array(size[:2]) / 2
            upper = np.array(center[:2]) + np.array(size[:2]) / 2
            if "square_bounds_m" in node:
                lower = np.maximum(lower, node["square_bounds_m"][0])
                upper = np.minimum(upper, node["square_bounds_m"][1])
            lower = np.maximum(np.array([x, y]) - jitter, lower + np.array([w, length]) / 2)
            upper = np.minimum(np.array([x, y]) + jitter, upper - np.array([w, length]) / 2)
            if np.any(lower > upper):
                raise ValueError(f"No contained placement exists for {node['id']}")
            # Condition each independent range on containment directly. Global
            # rejection becomes impractical when many measured meshes fill a square.
            rx, ry = [f"Range({float(a)!r}, {float(b)!r})" for a, b in zip(lower, upper)]
            position = f"({rx}, {ry}, {top + height / 2 + 0.001!r})"
        lines += [
            f"{name} = new Object at {position},",
            f"    with width {w!r}, with length {length!r}, with height {height!r},",
            f"    with requireVisible False, with objectId {node['id']!r}",
        ]
        if node["kind"] == "support":
            lines.append(f"ego = {name}")
        else:
            lines += [
                f"require {name}.position.x - {w / 2!r} >= {center[0] - size[0] / 2!r}",
                f"require {name}.position.x + {w / 2!r} <= {center[0] + size[0] / 2!r}",
                f"require {name}.position.y - {length / 2!r} >= {center[1] - size[1] / 2!r}",
                f"require {name}.position.y + {length / 2!r} <= {center[1] + size[1] / 2!r}",
            ]
            if "square_bounds_m" in node:
                for axis, lower, upper, extent in zip("xy", *node["square_bounds_m"], (w, length)):
                    lines += [
                        f"require {name}.position.{axis} - {extent / 2!r} >= {float(lower)!r}",
                        f"require {name}.position.{axis} + {extent / 2!r} <= {float(upper)!r}",
                    ]
        lines.append("")
    Path(path).write_text("\n".join(lines))


def sample_scenic(path, graph, out, count=20, seed=7):
    import scenic

    if count < 1:
        raise ValueError("sample count must be positive")
    random.seed(seed)
    np.random.seed(seed)
    scenario = scenic.scenarioFromFile(str(Path(path).resolve()))
    samples, iterations = [], []
    for i in range(count):
        scene, attempts = scenario.generate(maxIterations=2000)
        by_id = {obj.objectId: obj for obj in scene.objects}
        expected = {n["id"] for n in graph["nodes"] if n["kind"] in ("support", "object")}
        if set(by_id) != expected:
            raise ValueError("Scenic object IDs do not match graph")
        sampled = copy.deepcopy(graph)
        sampled.update(
            source="Scenic sample",
            sample_index=i,
            sample_seed=seed,
            variation="bounded XY jitter; 1 mm initial clearance",
            tracks=[],
        )
        sampled["edges"] = [
            dict(e, evidence="sampled_constraint") if e["relation"] == "on" else e for e in sampled["edges"]
        ]
        for node in sampled["nodes"]:
            if node["id"] not in by_id:
                continue
            obj = by_id[node["id"]]
            node["position_m"] = [float(v) for v in obj.position]
            node["size_m"] = [float(obj.width), float(obj.length), float(obj.height)]
        validate_sample(sampled)
        samples.append(sampled)
        iterations.append(attempts)
    write_json(out, dict(seed=seed, count=count, attempts=iterations, samples=samples))
    return samples


def validate_sample(graph):
    support = next(n for n in graph["nodes"] if n["kind"] == "support")
    c, d = np.array(support["position_m"]), np.array(support["size_m"])
    objects = [n for n in graph["nodes"] if n["kind"] == "object"]
    for node in objects:
        p, s = np.array(node["position_m"]), np.array(node["size_m"])
        if np.any(p[:2] - s[:2] / 2 < c[:2] - d[:2] / 2 - 1e-8) or np.any(
            p[:2] + s[:2] / 2 > c[:2] + d[:2] / 2 + 1e-8
        ):
            raise ValueError("Scenic sample extends beyond the support")
        if abs(p[2] - s[2] / 2 - (c[2] + d[2] / 2) - 0.001) > 1e-7:
            raise ValueError("Sample has incorrect support clearance")
        if "square_bounds_m" in node:
            lower, upper = np.array(node["square_bounds_m"])
            if np.any(p[:2] - s[:2] / 2 < lower - 1e-8) or np.any(p[:2] + s[:2] / 2 > upper + 1e-8):
                raise ValueError("Sample extends beyond its assigned square")
    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            separation = np.abs(np.array(a["position_m"]) - b["position_m"])
            extent = (np.array(a["size_m"]) + b["size_m"]) / 2
            if np.all(separation < extent - 1e-8):
                raise ValueError("Sample contains overlapping boxes")
