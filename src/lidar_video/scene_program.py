"""Constructive scene programs for fixed-orientation, axis-aligned box proxies.

Sequential conditional intervals are projections of a *joint* feasible polytope.
Every draw therefore admits a completion (up to solver tolerances), without scene
rejection, clipping or repair. This distribution is order-dependent, not uniform
over the polytope. Non-overlap keeps one separating axis witnessed in the input;
it does not explore every possible collision-free arrangement.
"""

import copy
import json
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import linprog

TOL = 1e-8
RELATIONS = {"on", "inside", "left_of", "right_of", "in_front_of", "behind", "above", "below"}


def as_networkx(graph):
    """Lossless attributed multigraph view; reject dangling edges and duplicate IDs."""
    result = nx.MultiDiGraph()
    result.graph.update(copy.deepcopy({k: v for k, v in graph.items() if k not in ("nodes", "edges")}))
    for node in graph["nodes"]:
        name = node["id"]
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("Scene node IDs must be unique nonempty strings")
        result.add_node(name, **copy.deepcopy(node))
    for edge in graph["edges"]:
        if edge["source"] not in result or edge["target"] not in result:
            raise ValueError("Scene relation has a missing endpoint")
        key = result.add_edge(edge["source"], edge["target"])
        result.edges[edge["source"], edge["target"], key].update(copy.deepcopy(edge))
    return result


def _vector(value, label):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f"{label} must be a finite XYZ vector")
    return a


@dataclass
class SceneProgram:
    graph: dict
    order: tuple
    bounds: np.ndarray
    inequalities: np.ndarray
    limits: np.ndarray
    equalities: np.ndarray
    targets: np.ndarray
    witnesses: list

    def _solve(self, objective, bounds):
        result = linprog(
            objective,
            A_ub=self.inequalities if len(self.inequalities) else None,
            b_ub=self.limits if len(self.limits) else None,
            A_eq=self.equalities if len(self.equalities) else None,
            b_eq=self.targets if len(self.targets) else None,
            bounds=bounds,
            method="highs",
            options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
        )
        if not result.success:
            raise ValueError(f"Scene constraints infeasible or solver failed: {result.message}")
        return result.x

    def validate_positions(self, values):
        x = np.asarray(values, dtype=float).reshape(-1)
        if x.shape != (len(self.bounds),) or not np.isfinite(x).all():
            raise ValueError("Invalid sampled positions")
        if np.any(x < self.bounds[:, 0] - TOL) or np.any(x > self.bounds[:, 1] + TOL):
            raise ValueError("Placement outside declared domain")
        if len(self.limits) and np.any(self.inequalities @ x > self.limits + TOL):
            raise ValueError("Placement violates a spatial or non-overlap constraint")
        if len(self.targets) and np.any(np.abs(self.equalities @ x - self.targets) > TOL):
            raise ValueError("Placement violates a support constraint")

    def positions(self, *, randomize=True, rng=None):
        if not isinstance(randomize, bool):
            raise ValueError("randomize must be boolean")
        nodes = {n["id"]: n for n in self.graph["nodes"]}
        if not randomize:
            values = [nodes[name]["position_m"] for name in self.order]
            self.validate_positions(values)
            return tuple(tuple(p) for p in values)
        rng = random if rng is None else rng
        bounds = self.bounds.copy()
        result = self._solve(np.zeros(len(bounds)), bounds)
        for index in range(len(bounds)):
            if bounds[index, 0] == bounds[index, 1]:
                continue
            objective = np.zeros(len(bounds))
            objective[index] = 1
            low = self._solve(objective, bounds)[index]
            high = self._solve(-objective, bounds)[index]
            if low > high + TOL:
                raise ValueError("Inconsistent feasible projection; refusing to retry")
            value = low if high <= low else rng.uniform(low, high)
            bounds[index] = value, value
        result = self._solve(np.zeros(len(bounds)), bounds)
        self.validate_positions(result)
        return tuple(tuple(p) for p in result.reshape(-1, 3))

    def sample(self, *, randomize=True, seed=None):
        positions = dict(zip(self.order, self.positions(randomize=randomize, rng=random.Random(seed))))
        graph = copy.deepcopy(self.graph)
        for node in graph["nodes"]:
            node["position_m"] = list(positions[node["id"]])
        if randomize:
            graph.update(source="constructive scene program", tracks=[])
            for edge in graph["edges"]:
                edge["evidence"] = "sampled_constraint"
        graph["scene_program"] = {
            "randomized": randomize,
            "seed": seed,
            "placement_order": list(self.order),
            "sampling_attempts": 1,
            "distribution": "sequential uniform feasible-coordinate intervals; not joint uniform",
            "geometry": "fixed-orientation axis-aligned box proxies",
            "nonoverlap_witnesses": copy.deepcopy(self.witnesses),
        }
        return graph


def compile_scene_program(graph, *, jitter=0.004):
    """Compile explicit box relations. Unknown relations/rotations fail closed.

    `on` means full XY containment and top-face contact (optional nonnegative
    clearance_m, or an explicit support_penetration_m); `inside` means full XYZ
    containment. Directional edges compare box faces, with optional nonnegative
    gap_m. +Y is `in_front_of`.
    `placement_bounds_m` bounds the object's center in world XYZ. Root supports
    default to fixed; other nodes default to +/- jitter around recorded centers.
    Set fixed=False and explicit bounds to randomize a root support.
    """
    if not np.isfinite(jitter) or jitter < 0:
        raise ValueError("jitter must be finite and nonnegative")
    if graph.get("units", "metres") != "metres":
        raise ValueError("Scene program coordinates must use metres")
    network = as_networkx(graph)
    if not network:
        raise ValueError("Scene must contain at least one physical node")
    dependency = nx.DiGraph()
    dependency.add_nodes_from(network)
    parents = {}
    for a, b, edge in network.edges(data=True):
        relation = edge["relation"]
        if relation not in RELATIONS or a == b:
            raise ValueError(f"Unsupported spatial relation: {relation} ({a}, {b})")
        if relation in ("on", "inside"):
            if a in parents:
                raise ValueError(f"Multiple placement parents for {a} are unsupported")
            parents[a] = b
            dependency.add_edge(b, a)
    if not nx.is_directed_acyclic_graph(dependency):
        raise ValueError("Support/containment dependency cycle")
    sizes, positions = {}, {}
    for name, node in network.nodes(data=True):
        if node.get("kind") not in ("object", "support"):
            raise ValueError("Compiler accepts physical object/support nodes only")
        sizes[name] = _vector(node["size_m"], "size_m")
        positions[name] = _vector(node["position_m"], "position_m")
        if np.any(sizes[name] <= 0):
            raise ValueError("Box dimensions must be positive")
        for field in (
            "rotation",
            "rotation_matrix",
            "orientation",
            "orientation_quat",
            "quaternion",
            "heading",
        ):
            if field in node:
                raise ValueError("Rotation fields are unsupported; supply fixed world-aligned box proxies")
    order = tuple(nx.lexicographical_topological_sort(dependency, key=lambda n: (-np.prod(sizes[n]), n)))
    indices = {name: 3 * i for i, name in enumerate(order)}
    bounds = []
    for name in order:
        node, pos = network.nodes[name], positions[name]
        fixed = node.get("fixed", node["kind"] == "support" and name not in parents)
        if not isinstance(fixed, bool):
            raise ValueError("fixed must be a boolean")
        if fixed:
            lo = hi = pos
        elif "placement_bounds_m" in node:
            lo, hi = (_vector(v, "placement_bounds_m") for v in node["placement_bounds_m"])
        else:
            lo, hi = pos - jitter, pos + jitter
        if np.any(lo > hi):
            raise ValueError("Placement bounds are reversed")
        bounds.extend(zip(lo, hi))
    width = 3 * len(order)
    inequalities, limits, equalities, targets = [], [], [], []
    support_contacts = {}

    def constraint(a, b, axis, sign, limit, equality=False):
        row = np.zeros(width)
        row[indices[a] + axis], row[indices[b] + axis] = sign, -sign
        (equalities if equality else inequalities).append(row)
        (targets if equality else limits).append(limit)

    directional = {
        "left_of": (0, 1),
        "right_of": (0, -1),
        "behind": (1, 1),
        "in_front_of": (1, -1),
        "below": (2, 1),
        "above": (2, -1),
    }
    for a, b, edge in network.edges(data=True):
        relation = edge["relation"]
        gap = float(edge.get("clearance_m" if relation == "on" else "gap_m", 0))
        if not np.isfinite(gap) or gap < 0:
            raise ValueError("Relation clearance/gap must be finite and nonnegative")
        penetration = float(edge.get("support_penetration_m", 0))
        if not np.isfinite(penetration) or penetration < 0:
            raise ValueError("Support penetration must be finite and nonnegative")
        if penetration and (relation != "on" or gap or penetration >= min(sizes[a][2], sizes[b][2])):
            raise ValueError("Penetration allowance requires on, zero clearance and partial overlap")
        if relation == "on":
            support_contacts[frozenset((a, b))] = (a, b, penetration)
        if relation in ("on", "inside"):
            for axis in range(2 if relation == "on" else 3):
                margin = (sizes[b][axis] - sizes[a][axis]) / 2 - (gap if relation == "inside" else 0)
                for sign in (-1, 1):
                    constraint(a, b, axis, sign, margin)
            if relation == "on":
                constraint(a, b, 2, 1, (sizes[a][2] + sizes[b][2]) / 2 + gap - penetration, equality=True)
        else:
            axis, sign = directional[relation]
            constraint(a, b, axis, sign, -(sizes[a][axis] + sizes[b][axis]) / 2 - gap)
    # The interior of a container is not a solid obstacle for its descendants.
    inside = nx.DiGraph()
    inside.add_nodes_from(order)
    inside.add_edges_from((b, a) for a, b, e in network.edges(data=True) if e["relation"] == "inside")
    witnesses = []
    for a, b in combinations(order, 2):
        if nx.has_path(inside, a, b) or nx.has_path(inside, b, a):
            continue
        contact = support_contacts.get(frozenset((a, b)))
        if contact:
            child, parent, penetration = contact
            witnesses.append(
                dict(source=parent, target=child, axis="z", sign=1, support_penetration_m=penetration)
            )
            # The on equality already enforces separation/contact with the
            # explicitly declared collision-envelope penetration. No extra gap.
            continue
        delta = positions[b] - positions[a]
        extents = (sizes[a] + sizes[b]) / 2
        axes = [i for i in range(3) if abs(delta[i]) >= extents[i] - TOL]
        if not axes:
            raise ValueError(f"Reference boxes overlap: {a}, {b}; cannot infer a separation branch")
        axis = max(axes, key=lambda i: abs(delta[i]) - extents[i])
        sign = 1 if delta[axis] >= 0 else -1
        constraint(a, b, axis, sign, -extents[axis])
        witnesses.append(dict(source=a, target=b, axis="xyz"[axis], sign=sign))
    program = SceneProgram(
        copy.deepcopy(graph),
        order,
        np.array(bounds),
        np.array(inequalities).reshape(-1, width),
        np.array(limits),
        np.array(equalities).reshape(-1, width),
        np.array(targets),
        witnesses,
    )
    program._solve(np.zeros(width), program.bounds)
    # Replay must be exact and valid, never snapped into an acceptable scene.
    program.positions(randomize=False)
    return program


def export_constructive_scenic(graph, path, *, randomize=True, jitter=0.004):
    """Emit executable Scenic backed by one joint constructive distribution."""
    program = compile_scene_program(graph, jitter=jitter)
    nodes = {n["id"]: n for n in graph["nodes"]}
    lines = [
        "# Constructive box-proxy program; rotations and articulated dynamics are not modeled.",
        "# No scene rejection or require clauses. Geometric constraints live in the joint distribution.",
        "from lidar_video.scenic_constructive import ConstructivePlacement",
        f"placements = ConstructivePlacement({json.dumps(graph)!r}, {randomize!r}, {jitter!r})",
    ]
    for i, name in enumerate(program.order):
        w, length, height = map(float, nodes[name]["size_m"])
        lines += [
            f"obj_{i} = new Object at (placements[{i}][0], placements[{i}][1], placements[{i}][2]),",
            f"    with width {w!r}, with length {length!r}, with height {height!r},",
            f"    with objectId {name!r}, with requireVisible False, with allowCollisions True",
        ]
        if "mesh_path" in nodes[name]:
            # Metadata binds the proxy placement to the external native assets.
            lines[-1] += ","
            lines += [
                f"    with meshPath {nodes[name]['mesh_path']!r},",
                f"    with collisionPaths {tuple(nodes[name]['collision_paths'])!r}",
            ]
    lines.append("ego = obj_0")
    Path(path).write_text("\n".join(lines) + "\n")
    return program


def run_scene_program(graph, out, *, randomize=False, jitter=0.004, count=20, seed=7):
    """Execute the emitted Scenic program with a hard one-attempt budget per scene."""
    import scenic

    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(randomize, bool):
        raise ValueError("randomize must be boolean")
    compile_scene_program(graph, jitter=jitter)  # Fail before creating output on invalid input.
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    path = out / "scene.scenic"
    program = export_constructive_scenic(graph, path, randomize=randomize, jitter=jitter)
    previous_python, previous_numpy = random.getstate(), np.random.get_state()
    samples, attempts = [], []
    try:
        random.seed(seed)
        np.random.seed(seed)
        scenario = scenic.scenarioFromFile(str(path.resolve()))
        for i in range(count):
            scene, used = scenario.generate(maxIterations=1)
            ids = [obj.objectId for obj in scene.objects]
            if len(ids) != len(set(ids)) or set(ids) != set(program.order):
                raise ValueError("Scenic object inventory differs from compiled graph")
            by_id = dict(zip(ids, scene.objects))
            positions = [list(by_id[name].position) for name in program.order]
            program.validate_positions(positions)
            sample = copy.deepcopy(graph)
            for node in sample["nodes"]:
                obj = by_id[node["id"]]
                if not np.allclose([obj.width, obj.length, obj.height], node["size_m"], rtol=0, atol=TOL):
                    raise ValueError("Scenic dimensions differ from graph")
                if "mesh_path" in node and (
                    obj.meshPath != node["mesh_path"]
                    or tuple(obj.collisionPaths) != tuple(node["collision_paths"])
                ):
                    raise ValueError("Scenic asset bindings differ from graph")
                node["position_m"] = list(obj.position)
            if randomize:
                sample.update(source="constructive Scenic sample", tracks=[])
                for edge in sample["edges"]:
                    edge["evidence"] = "sampled_constraint"
            sample["scene_program"] = dict(sample_index=i, seed=seed, randomized=randomize)
            samples.append(sample)
            attempts.append(used)
    finally:
        random.setstate(previous_python)
        np.random.set_state(previous_numpy)
    result = dict(
        count=count,
        seed=seed,
        randomized=randomize,
        attempts=attempts,
        placement_order=list(program.order),
        nonoverlap_witnesses=program.witnesses,
        distribution="sequential uniform feasible-coordinate intervals; not joint uniform",
        geometry="fixed-orientation axis-aligned box proxies",
        samples=samples,
    )
    (out / "samples.json").write_text(json.dumps(result, indent=2) + "\n")
    return {k: v for k, v in result.items() if k != "samples"} | {"program": str(path)}
