"""Constructive Scenic placements bound to an existing reconstruction's assets.

The full semantic graph is retained. Its physical constraints are compiled into
box envelopes containing the supplied visual and collision meshes. Small known
support penetrations require an explicit budget; replay never moves a node.
"""

import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh

from lidar_video.scene_program import RELATIONS, TOL, as_networkx, compile_scene_program, run_scene_program


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_asset_graph(graph, asset_root, *, jitter=0.004, support_tolerance=0.0):
    """Validate assets and compile known semantic edges without dropping records."""
    if not np.isfinite(support_tolerance) or not 0 <= support_tolerance <= 0.005:
        raise ValueError("support_tolerance must be between zero and 5 mm")
    if not np.isfinite(jitter) or jitter < 0:
        raise ValueError("jitter must be finite and nonnegative")
    root = Path(asset_root).resolve()
    net = as_networkx(graph)
    nodes = {n["id"]: n for n in graph["nodes"]}
    if any(n.get("kind") not in {"support", "object", "square"} for n in nodes.values()):
        raise ValueError("Asset adapter supports physical nodes and calibrated square regions only")
    physical = {name: copy.deepcopy(n) for name, n in nodes.items() if n["kind"] != "square"}
    if any(name in {".", ".."} or "/" in name or "\\" in name for name in physical):
        raise ValueError("Physical IDs must be safe asset filename components")
    if not physical or not any(n["kind"] == "support" for n in physical.values()):
        raise ValueError("Native asset export requires a physical support")
    assets, contacts, semantic = {}, [], []
    for name, node in physical.items():
        position, size = np.asarray(node["position_m"], float), np.asarray(node["size_m"], float)
        if (
            position.shape != (3,)
            or size.shape != (3,)
            or not np.isfinite([position, size]).all()
            or np.any(size <= 0)
        ):
            raise ValueError("Physical nodes require finite XYZ centers and positive dimensions")
        if "mesh_path" in node:
            if not isinstance(node.get("collision_paths"), list) or not node["collision_paths"]:
                raise ValueError(f"Mesh node {name} requires collision assets")
            for relative in [node["mesh_path"], *node["collision_paths"]]:
                path = (root / relative).resolve()
                if (
                    Path(relative).is_absolute()
                    or not path.is_relative_to(root)
                    or path.suffix.lower() != ".obj"
                ):
                    raise ValueError("Native mesh assets must be relative OBJ files within asset_root")
                mesh = trimesh.load(path, force="mesh", process=False)
                if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
                    raise ValueError(f"Invalid mesh asset for {name}")
                if np.any(mesh.bounds[0] < -size / 2 - TOL) or np.any(mesh.bounds[1] > size / 2 + TOL):
                    raise ValueError(f"Declared envelope does not contain asset for {name}: {relative}")
                assets[relative] = _digest(path)
        elif node.get("collision_paths"):
            raise ValueError("Collision assets without a visual mesh")

    region_parents = {}
    supported = {e["source"] for e in graph["edges"] if e["relation"] in {"on", "inside"}}
    for edge in graph["edges"]:
        a, b, relation = edge["source"], edge["target"], edge["relation"]
        if relation == "part_of":
            if (
                nodes[a]["kind"] != "square"
                or b not in physical
                or physical[b]["kind"] != "support"
                or a in region_parents
            ):
                raise ValueError("part_of requires one physical support per calibrated square")
            if not physical[b].get("fixed", b not in supported):
                raise ValueError("Calibrated square regions currently require a fixed support")
            region_parents[a] = b
    if set(region_parents) != {name for name, n in nodes.items() if n["kind"] == "square"}:
        raise ValueError("Each calibrated square requires a part_of support")
    constraints, occupants = [], set()
    for index, edge in enumerate(graph["edges"]):
        a, b, relation = edge["source"], edge["target"], edge["relation"]
        if relation == "part_of":
            semantic.append(dict(edge_index=index, handling="retained fixed region membership"))
        elif relation == "occupies":
            if (
                a not in physical
                or physical[a]["kind"] != "object"
                or b not in region_parents
                or a in occupants
            ):
                raise ValueError("occupies requires one calibrated square per physical object")
            occupants.add(a)
            node, support = physical[a], physical[region_parents[b]]
            if node.get("square", b) != b:
                raise ValueError("Object square metadata contradicts occupies")
            square_size = float(support.get("square_size_m", 0))
            center = np.asarray(nodes[b]["position_m"], float)
            if (
                center.shape != (3,)
                or not np.isfinite(center).all()
                or not np.isfinite(square_size)
                or square_size <= 0
            ):
                raise ValueError("Calibrated squares require finite centers and support square_size_m")
            region = np.array([center[:2] - square_size / 2, center[:2] + square_size / 2])
            if "square_bounds_m" in node and not np.allclose(
                node["square_bounds_m"], region, atol=TOL, rtol=0
            ):
                raise ValueError("Declared square bounds disagree with calibrated region")
            position, size = np.array(node["position_m"]), np.array(node["size_m"])
            bounds = np.array(node.get("placement_bounds_m", [position - jitter, position + jitter]), float)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1]):
                raise ValueError("Invalid placement bounds")
            bounds[0, :2] = np.maximum(bounds[0, :2], region[0] + size[:2] / 2)
            bounds[1, :2] = np.minimum(bounds[1, :2], region[1] - size[:2] / 2)
            # Even fixed nodes must satisfy the semantic region constraint.
            if np.any(position[:2] < bounds[0, :2] - TOL) or np.any(position[:2] > bounds[1, :2] + TOL):
                raise ValueError("Reference envelope violates occupies or its placement bounds")
            node["placement_bounds_m"] = bounds.tolist()
            if not any(
                e["source"] == a and e["target"] == region_parents[b] and e["relation"] == "on"
                for e in graph["edges"]
            ):
                raise ValueError("Occupant must be on its square's support")
            semantic.append(
                dict(edge_index=index, handling="full envelope footprint inside calibrated square")
            )
        elif relation in RELATIONS:
            if a not in physical or b not in physical:
                raise ValueError("Physical relation references a semantic-only region")
            converted = copy.deepcopy(edge)
            if relation == "on":
                child, parent = physical[a], physical[b]
                gap = (
                    child["position_m"][2]
                    - child["size_m"][2] / 2
                    - parent["position_m"][2]
                    - parent["size_m"][2] / 2
                )
                declared = float(edge.get("clearance_m", 0)) - float(edge.get("support_penetration_m", 0))
                if abs(gap - declared) > TOL:
                    if gap < 0 and -gap <= support_tolerance and not declared:
                        converted["support_penetration_m"] = -gap
                    else:
                        raise ValueError(
                            f"Reference support gap {gap:.9g} m for {a}; explicit support tolerance required for existing overlap"
                        )
                penetration = float(converted.get("support_penetration_m", 0))
                if penetration > support_tolerance + TOL:
                    raise ValueError("Declared support penetration exceeds explicit tolerance budget")
                contacts.append(dict(source=a, target=b, reference_gap_m=gap, penetration_m=penetration))
            constraints.append(converted)
            semantic.append(dict(edge_index=index, handling="compiled physical constraint"))
        else:
            raise ValueError(f"Unsupported relation: {relation}; no records are silently discarded")
    for name, node in physical.items():
        if "square_bounds_m" in node and name not in occupants:
            raise ValueError("square_bounds_m requires an explicit occupies relation")
    proxy = {k: copy.deepcopy(v) for k, v in graph.items() if k not in {"nodes", "edges"}}
    proxy.update(nodes=list(physical.values()), edges=constraints)
    compile_scene_program(proxy, jitter=jitter)
    return proxy, dict(
        assets=assets,
        support_contacts=contacts,
        semantic_edges=semantic,
        support_tolerance_m=support_tolerance,
        source_nodes=len(net),
        physical_nodes=len(physical),
    )


def run_asset_scene_program(
    graph_file,
    out,
    *,
    randomize=False,
    jitter=0.004,
    count=20,
    seed=7,
    support_tolerance=0.0,
    simulate=False,
    frames=30,
):
    """Emit Scenic, complete sampled graphs, copied assets and optional native rollouts."""
    path, out = Path(graph_file).resolve(), Path(out).resolve()
    if out == path.parent or out.is_relative_to(path.parent):
        raise ValueError("Output must be separate from the source scene")
    graph = json.loads(path.read_text())
    graph_hash = _digest(path)
    proxy, audit = prepare_asset_graph(graph, path.parent, jitter=jitter, support_tolerance=support_tolerance)
    if simulate and (not isinstance(frames, int) or isinstance(frames, bool) or frames < 2):
        raise ValueError("Simulation requires at least two frames")
    if out.exists():
        raise FileExistsError(out)
    # run_scene_program validates counts and constraints before creating output.
    result = run_scene_program(
        proxy, out / "program", randomize=randomize, jitter=jitter, count=count, seed=seed
    )
    samples = json.loads((out / "program/samples.json").read_text())["samples"]
    complete = []
    for sample in samples:
        restored = copy.deepcopy(graph)
        if randomize:
            positions = {n["id"]: n["position_m"] for n in sample["nodes"]}
            for node in restored["nodes"]:
                if node["id"] in positions:
                    node["position_m"] = positions[node["id"]]
            restored.update(source="constructive asset scene program", tracks=[])
            for edge in restored["edges"]:
                if edge["relation"] != "part_of":
                    edge["evidence"] = "sampled_constraint"
        restored["scene_program"] = dict(
            sample["scene_program"],
            support_tolerance_m=support_tolerance,
            constraints="program/constraint_graph.json",
        )
        complete.append(restored)
    # Detect source edits before exporting any asset copy or native simulation.
    if _digest(path) != graph_hash or any(_digest(path.parent / p) != h for p, h in audit["assets"].items()):
        raise ValueError("Source graph/assets changed while sampling")
    for relative in audit["assets"]:
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path.parent / relative, target)
    (out / "source_graph.json").write_text(json.dumps(graph, indent=2) + "\n")
    (out / "program/constraint_graph.json").write_text(json.dumps(proxy, indent=2) + "\n")
    (out / "samples.json").write_text(
        json.dumps(dict(samples=complete, attempts=result["attempts"]), indent=2) + "\n"
    )
    physics = []
    for i, sample in enumerate(complete):
        folder = out / "samples" / f"{i:03d}"
        folder.mkdir(parents=True)
        # Assets remain relative to the output root; each sample has its own copy-free graph at root.
        (out / f"scene_graph_{i:03d}.json").write_text(json.dumps(sample, indent=2) + "\n")
        if simulate:
            from lidar_video.simulation import simulate as simulate_native

            report, replay = simulate_native(sample, folder / "simulation", frames=frames, asset_root=out)
            physics.append(
                dict(
                    sample_index=i,
                    warnings=report["warnings"],
                    max_penetration_m=report["max_penetration_m"],
                    replay=replay,
                )
            )
    result.update(
        audit,
        source_graph_sha256=graph_hash,
        physics=physics,
        replay_nodes_edges_exact=not randomize
        and all(s["nodes"] == graph["nodes"] and s["edges"] == graph["edges"] for s in complete),
        scope="Initial placements of fixed-orientation asset envelopes; explicit support overlap; no dynamic stability guarantee",
    )
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
