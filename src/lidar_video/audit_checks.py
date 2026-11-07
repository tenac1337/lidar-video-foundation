"""Deterministic evidence for a bounded creation/audit loop.

These checks validate declared geometry and passive tabletop simulation only.
They do not establish grasp feasibility, chess semantics, or robot task success.
An agent cannot override the existing measured-geometry acceptance gates.
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _result(status, identifier, summary, evidence=None, **extra):
    return dict(
        status=status, findings=[dict(id=identifier, summary=summary, evidence=evidence or {})], **extra
    )


def _path(root, value):
    return (Path(root) / value).resolve()


def validate_graph(graph, expected_objects):
    """Reject empty, incomplete, nonfinite, overlapping or disconnected scene data."""
    if graph.get("units") != "metres":
        raise ValueError("Graph units must be metres")
    nodes = graph.get("nodes", [])
    ids = [n.get("id") for n in nodes]
    if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Graph requires nonempty unique stable IDs")
    physical = [n for n in nodes if n.get("kind") in ("support", "object")]
    supports = [n for n in physical if n["kind"] == "support"]
    objects = [n for n in physical if n["kind"] == "object"]
    if len(supports) != 1 or not objects:
        raise ValueError("Require exactly one support and at least one object")
    actual = {n.get("square", n["id"]) for n in objects}
    if len(actual) != len(objects) or actual != set(expected_objects):
        raise ValueError("Object identities do not exactly match expected_objects")
    for node in nodes:
        p = np.asarray(node.get("position_m"), dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError(f"Nonfinite or invalid position: {node['id']}")
        if node in physical:
            size = np.asarray(node.get("size_m"), dtype=float)
            if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
                raise ValueError(f"Invalid dimensions: {node['id']}")
    edges = graph.get("edges", [])
    if any(e.get("source") not in ids or e.get("target") not in ids for e in edges):
        raise ValueError("Edge references a missing stable ID")
    support = supports[0]
    center, size = np.array(support["position_m"]), np.array(support["size_m"])
    top = center[2] + size[2] / 2
    for node in objects:
        p, s = np.array(node["position_m"]), np.array(node["size_m"])
        if np.any(p[:2] - s[:2] / 2 < center[:2] - size[:2] / 2 - 1e-7) or np.any(
            p[:2] + s[:2] / 2 > center[:2] + size[:2] / 2 + 1e-7
        ):
            raise ValueError(f"Object extends outside support: {node['id']}")
        if not -0.005 <= p[2] - s[2] / 2 - top <= 0.01:
            raise ValueError(f"Object base is unsupported: {node['id']}")
        if not any(
            e.get("source") == node["id"] and e.get("relation") == "on" and e.get("target") == support["id"]
            for e in edges
        ):
            raise ValueError(f"Missing support relation: {node['id']}")
    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            separation = np.abs(np.array(a["position_m"]) - b["position_m"])
            extent = (np.array(a["size_m"]) + b["size_m"]) / 2
            if np.all(separation < extent - 1e-7):
                raise ValueError("Object envelopes overlap")
    return physical


def _geometry(case, root, graph):
    if case.get("geometry_scope") == "synthetic_fixture":
        if case.get("capture") or case.get("calibration") or case.get("mesh_manifest"):
            raise ValueError("Synthetic fixture cannot claim capture, calibration or learned mesh validation")
        if graph is None:
            return _result("blocked", "geometry.scene_missing", "Synthetic scene graph is missing")
        validate_graph(graph, case["expected_objects"])
        return _result("pass", "geometry.synthetic", "Synthetic structure only; no real capture claim")
    if not case.get("mesh_manifest"):
        return _result(
            "blocked", "geometry.manifest_missing", "Measured learned geometry needs a mesh manifest"
        )
    if graph is not None:
        validate_graph(graph, case["expected_objects"])
    from lidar_video.capture import load_capture
    from lidar_video.reconstruction import check_mesh_bounds, import_learned_mesh

    manifest_path = _path(root, case["mesh_manifest"])
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("objects")
    if manifest.get("schema") != "roboskills-chess-meshes-v1" or not isinstance(entries, dict):
        raise ValueError("Unsupported learned-mesh manifest")
    if set(entries) != set(case["expected_objects"]) or not entries:
        raise ValueError("Manifest must contain every expected object exactly once")
    if any(
        e.get("validation") not in ("instance-multiview-v1", "instance-multiview-v2")
        for e in entries.values()
    ):
        raise ValueError(
            "Every learned mesh requires instance-multiview-v1 or v2; legacy evidence is rejected"
        )
    if not case.get("capture") or not case.get("calibration"):
        return _result("blocked", "geometry.inputs_missing", "Capture and saved calibration are required")
    import chess

    data = load_capture(_path(root, case["capture"]))
    calibration = json.loads(_path(root, case["calibration"]).read_text())
    size = float(calibration["square_size_m"])
    gap = float(calibration.get("center_gap_m", 0))
    if not np.isfinite([size, gap]).all() or not 0.01 <= size <= 0.2 or not 0 <= gap <= size:
        raise ValueError("Invalid calibrated square dimensions")
    findings = []
    for name, entry in entries.items():
        try:
            square = chess.parse_square(name)
            mesh, evidence = import_learned_mesh(
                dict(entry, square=name), manifest_path.parent, data, calibration, []
            )
            center = (np.array([chess.square_file(square), chess.square_rank(square)]) - 3.5) * size
            center[1] += np.sign(center[1]) * gap / 2
            check_mesh_bounds(mesh, center - size / 2, center + size / 2)
            if graph is not None:
                scene = _path(root, case["scene_dir"])
                node = next(
                    n for n in graph["nodes"] if n.get("square", n["id"]) == name and n["kind"] == "object"
                )
                evidence["export_link"] = _link_aligned_mesh(mesh, evidence, node, scene)
            findings.append(
                dict(
                    id=f"geometry.{name}",
                    summary="Fixed measured geometry gates passed",
                    evidence=evidence,
                    status="pass",
                )
            )
        except (ValueError, OSError, KeyError) as error:
            findings.append(
                dict(
                    id=f"geometry.{name}",
                    summary=str(error),
                    evidence=getattr(error, "evidence", None) or {},
                    status="fail",
                )
            )
    return dict(
        status="pass" if all(f["status"] == "pass" for f in findings) else "fail",
        findings=findings,
        scope="unchanged per-instance multiview import gates; no fallback",
    )


def _link_aligned_mesh(mesh, evidence, node, scene):
    """Bind the graph's saved full-resolution asset to this exact accepted candidate."""
    import trimesh
    from scipy.spatial import cKDTree

    claimed = node.get("evidence", {}).get("learned_mesh", {})
    if not claimed.get("source") or Path(claimed["source"]).resolve() != Path(evidence["source"]).resolve():
        raise ValueError("Export does not identify the validated learned mesh source")
    relative = node.get("mesh_path")
    if not relative:
        raise ValueError("Validated learned object has no exported mesh")
    full_path = (scene / relative).with_name("mesh.ply").resolve()
    if not full_path.is_relative_to(scene.resolve()):
        raise ValueError("Full learned mesh export escapes scene directory")
    exported = trimesh.load(full_path, force="mesh", process=True)
    if len(exported.vertices) != len(mesh.vertices) or len(exported.faces) != len(mesh.faces):
        raise ValueError("Full exported learned mesh has different topology counts")
    exported.apply_translation(node["position_m"])
    if not np.isfinite(exported.vertices).all():
        raise ValueError("Nonfinite full exported learned mesh")
    distances = [
        cKDTree(a).query(b)[0].max()
        for a, b in ((mesh.vertices, exported.vertices), (exported.vertices, mesh.vertices))
    ]
    face_distances = [
        cKDTree(a).query(b)[0].max()
        for a, b in (
            (mesh.triangles_center, exported.triangles_center),
            (exported.triangles_center, mesh.triangles_center),
        )
    ]
    # PLY uses float32. This is serialization tolerance, not a new geometric acceptance threshold.
    if max(*distances, *face_distances) > 1e-6 or not np.isclose(
        exported.area, mesh.area, rtol=1e-5, atol=1e-10
    ):
        raise ValueError("Full exported learned mesh differs from the validated aligned candidate")
    return dict(
        full_mesh=str(full_path),
        max_vertex_difference_m=float(max(distances)),
        max_face_center_difference_m=float(max(face_distances)),
        serialization_tolerance_m=1e-6,
    )


def _physics(graph, scene, expected):
    import mujoco

    physical = validate_graph(graph, expected)
    path = scene / "simulation" / "scene.xml"
    if not path.is_file():
        return _result("blocked", "physics.model_missing", "simulation/scene.xml is missing")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    if model.nu or model.neq or np.any(model.body_gravcomp):
        raise ValueError("Passive audit forbids actuators, equality pinning and gravity compensation")
    if model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY):
        raise ValueError("Gravity cannot be disabled")
    if not 0.0001 <= model.opt.timestep <= 0.01 or model.opt.gravity[2] >= -1:
        raise ValueError("Physics requires gravity and a bounded physical timestep")
    mujoco.mj_forward(model, data)
    if model.nbody != len(physical) + 1:
        raise ValueError("MuJoCo body inventory disagrees with graph")
    if model.body_geomnum[0]:
        raise ValueError("MuJoCo world contains geometry absent from the scene graph")
    ids = {}
    for node in physical:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, node["id"])
        if bid < 1 or not np.allclose(data.xpos[bid], node["position_m"], atol=1e-6):
            raise ValueError(f"MuJoCo body is missing or displaced: {node['id']}")
        if not np.allclose(data.xmat[bid].reshape(3, 3), np.eye(3), atol=1e-7, rtol=0):
            raise ValueError("Initial MuJoCo body orientation disagrees with axis-aligned graph export")
        ids[node["id"]] = bid
        joints = range(model.body_jntadr[bid], model.body_jntadr[bid] + model.body_jntnum[bid])
        free = [j for j in joints if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if node["kind"] == "object" and (len(free) != 1 or model.body_mass[bid] <= 0):
            raise ValueError(f"Object must have positive mass and a free joint: {node['id']}")
        if node["kind"] == "object" and not np.isclose(
            model.body_mass[bid], node.get("mass_kg", 0.08), rtol=1e-5, atol=1e-8
        ):
            raise ValueError("MuJoCo mass differs from declared object prior")
        if node["kind"] == "support" and model.body_jntnum[bid]:
            raise ValueError("Support must be fixed")
        geoms = range(model.body_geomadr[bid], model.body_geomadr[bid] + model.body_geomnum[bid])
        colliders = [g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g]]
        if not colliders:
            raise ValueError(f"No collision geometry for {node['id']}")
        if any(
            not np.allclose(model.geom_friction[g], [0.8, 0.005, 0.0001], atol=1e-8, rtol=0)
            for g in colliders
        ):
            raise ValueError("MuJoCo friction differs from declared pipeline prior")
        if "mesh_path" not in node:
            boxes = [g for g in colliders if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
            if len(geoms) != 1 or len(colliders) != 1 or len(boxes) != 1:
                raise ValueError("Primitive object requires exactly one box geom; no additional shapes")
            if not np.allclose(model.geom_size[boxes[0]] * 2, node["size_m"], atol=1e-6):
                raise ValueError("MuJoCo collision dimensions disagree with graph")
            if not np.allclose(data.geom_xpos[boxes[0]], node["position_m"], atol=1e-6, rtol=0):
                raise ValueError("MuJoCo primitive collision position disagrees with graph")
            if not np.allclose(data.geom_xmat[boxes[0]].reshape(3, 3), np.eye(3), atol=1e-7, rtol=0):
                raise ValueError("MuJoCo primitive collision orientation disagrees with graph")
        else:
            _validate_mujoco_mesh_assets(model, data, geoms, node, scene)
            vertices = []
            for g in colliders:
                if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                    raise ValueError("Graph mesh requires mesh collision geometry")
                mid = model.geom_dataid[g]
                start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                local = model.mesh_vert[start : start + count]
                vertices.extend(local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
            vertices = np.array(vertices)
            bounds = np.array([vertices.min(0), vertices.max(0)])
            size = np.array(node["size_m"])
            if np.any(np.abs(bounds - node["position_m"]) > size / 2 + 1e-5) or np.any(
                np.ptp(bounds, axis=0) < size * 0.9
            ):
                raise ValueError("MuJoCo mesh collision envelope disagrees with graph")
    penetration = 0.0
    # One physical second is enough for this bounded smoke test, not manipulation validation.
    steps = min(10000, int(np.ceil(1.0 / model.opt.timestep)))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Nonfinite MuJoCo state")
        penetration = max(penetration, max((-float(c.dist) for c in data.contact), default=0))
    warnings = [int(w.number) for w in data.warning]
    if any(warnings) or penetration > 0.005:
        raise ValueError(f"Physics warnings or excessive penetration: {warnings}, {penetration}")
    support = next(n for n in physical if n["kind"] == "support")
    top = support["position_m"][2] + support["size_m"][2] / 2
    touching = set()
    for contact in data.contact:
        bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        if ids[support["id"]] in bodies:
            touching.update(bodies)
    positions = {}
    for node in physical:
        if node["kind"] != "object":
            continue
        pos = data.xpos[ids[node["id"]]]
        positions[node["id"]] = pos.tolist()
        if ids[node["id"]] not in touching:
            raise ValueError(f"Object has no support contact after rollout: {node['id']}")
        if np.linalg.norm(pos[:2] - node["position_m"][:2]) > 0.02 or pos[2] < top:
            raise ValueError(f"Object escaped its passive placement: {node['id']}")
    if np.linalg.norm(data.qvel) > 0.1:
        raise ValueError("Scene has not settled after bounded rollout")
    return _result(
        "pass",
        "physics.headless",
        "Fresh headless passive-stability check passed",
        dict(
            steps=steps,
            seconds=float(data.time),
            max_penetration_m=penetration,
            warning_counts=warnings,
            final_positions_m=positions,
        ),
        scope="passive stability only; no grasp or robot manipulation claim",
    )


def _validate_mujoco_mesh_assets(model, data, geoms, node, scene):
    """Check the compiled visual/collision geometry against the graph's actual asset files."""
    import mujoco
    import trimesh
    from scipy.spatial import cKDTree

    paths = [node["mesh_path"], *node.get("collision_paths", [])]
    if len(geoms) != len(paths) or len(paths) < 2:
        raise ValueError("MuJoCo visual/collision asset inventory disagrees with graph")
    for index, (gid, relative) in enumerate(zip(geoms, paths)):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("MuJoCo graph asset must remain a mesh")
        collision = bool(model.geom_contype[gid] or model.geom_conaffinity[gid])
        if collision != (index > 0):
            raise ValueError("MuJoCo visual/collision roles disagree with graph")
        path = (scene / relative).resolve()
        if not path.is_relative_to(scene.resolve()):
            raise ValueError("MuJoCo graph asset escapes scene directory")
        source = trimesh.load(path, force="mesh", process=True)
        if not len(source.vertices) or not np.isfinite(source.vertices).all():
            raise ValueError("Invalid graph mesh asset")
        expected = source.vertices + node["position_m"]
        mid = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        compiled = model.mesh_vert[start : start + count] @ data.geom_xmat[gid].reshape(3, 3).T
        compiled += data.geom_xpos[gid]
        if (
            not len(compiled)
            or max(cKDTree(expected).query(compiled)[0].max(), cKDTree(compiled).query(expected)[0].max())
            > 1e-6
        ):
            raise ValueError("Compiled MuJoCo visual/collision surface disagrees with graph asset")


def _exports(graph, scene, expected, out):
    physical = validate_graph(graph, expected)
    program = scene / "scenario.scenic"
    saved = scene / "scenic_samples.json"
    if not program.is_file() or not saved.is_file():
        raise ValueError("Required Scenic program or saved samples are missing")
    import trimesh

    for node in physical:
        if "mesh_path" in node:
            paths = [node["mesh_path"], *node.get("collision_paths", [])]
            if len(paths) < 2:
                raise ValueError("Mesh object requires collision assets")
            for relative in paths:
                path = (scene / relative).resolve()
                if not path.is_relative_to(scene.resolve()):
                    raise ValueError("Export asset escapes scene directory")
                mesh = trimesh.load(path, force="mesh")
                if (
                    not isinstance(mesh, trimesh.Trimesh)
                    or not len(mesh.vertices)
                    or not len(mesh.faces)
                    or not np.isfinite(mesh.vertices).all()
                    or not np.isfinite(mesh.area)
                    or mesh.area <= 0
                ):
                    raise ValueError("Export requires a finite nonzero-area mesh")
                if np.any(np.abs(mesh.bounds) > np.array(node["size_m"]) / 2 + 1e-6):
                    raise ValueError("Export mesh exceeds graph dimensions")
    saved_record = json.loads(saved.read_text())
    samples = saved_record.get("samples", [])
    if not samples:
        raise ValueError("Saved Scenic samples are empty")
    _validate_exports_samples(samples, graph, expected)
    seed, count = saved_record.get("seed"), saved_record.get("count")
    if type(seed) is not int or type(count) is not int or count != len(samples):
        raise ValueError("Saved Scenic samples require an integer seed and matching count for program replay")
    # Scenic programs are executable Python-adjacent input. Audit only trusted project files;
    # this subprocess provides a timeout, not an untrusted-code security sandbox.
    target = out / "fresh_scenic_samples.json"
    replay_target = out / "replayed_scenic_samples.json"
    code = """import json, sys
try:
    import scenic
    from lidar_video.scenic_export import sample_scenic
except ImportError as error:
    print(f"Scenic runtime unavailable: {error}", file=sys.stderr)
    sys.exit(75)
graph = json.load(open(sys.argv[2]))
sample_scenic(sys.argv[1], graph, sys.argv[3], count=2, seed=17)
sample_scenic(sys.argv[1], graph, sys.argv[4], count=int(sys.argv[5]), seed=int(sys.argv[6]))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(program),
            str(scene / "scene_graph.json"),
            str(target),
            str(replay_target),
            str(count),
            str(seed),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (out / "scenic_compile.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode == 75:
        raise ImportError("Scenic worker runtime is unavailable; see scenic_compile.log")
    if completed.returncode:
        raise ValueError("Fresh Scenic compilation/sampling failed; see scenic_compile.log")
    fresh = json.loads(target.read_text())["samples"]
    _validate_exports_samples(fresh, graph, expected, expected_count=2)
    replayed = json.loads(replay_target.read_text())["samples"]
    _validate_exports_samples(replayed, graph, expected, expected_count=count)
    for actual, replay in zip(samples, replayed):
        # Replay the actual program, rather than inferring its distribution from
        # support bounds or trusting a sample's self-reported variation metadata.
        comparable = copy.deepcopy(actual)
        if len(actual["nodes"]) != len(replay["nodes"]):
            raise ValueError("Saved Scenic sample disagrees with program replay")
        replay_nodes = {n["id"]: n for n in replay["nodes"]}
        for node, copied in zip(actual["nodes"], comparable["nodes"]):
            expected_node = replay_nodes[node["id"]]
            if node.get("position_m") is not None:
                if not np.allclose(node["position_m"], expected_node.get("position_m"), atol=1e-8, rtol=0):
                    raise ValueError("Saved Scenic position disagrees with program replay")
                copied["position_m"] = expected_node["position_m"]
        reference = copy.deepcopy(replay)
        for record in (comparable, reference):
            record["nodes"] = sorted(record["nodes"], key=lambda n: n["id"])
            record["edges"] = sorted(record["edges"], key=lambda e: json.dumps(e, sort_keys=True))
        if comparable != reference:
            raise ValueError("Saved Scenic sample metadata disagrees with program replay")
    return _result(
        "pass",
        "exports.shared_ids",
        "Saved and fresh Scenic samples preserve graph identities, metadata and sizes",
        dict(
            saved_samples=len(samples),
            fresh_samples=len(fresh),
            replayed_samples=len(replayed),
            log="scenic_compile.log",
        ),
    )


def _validate_exports_samples(samples, graph, expected, *, expected_count=None):
    from lidar_video.scenic_export import validate_sample

    if not isinstance(samples, list) or not samples:
        raise ValueError("Scenic samples must be a nonempty list")
    if expected_count is not None and len(samples) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} fresh Scenic samples")
    source_nodes = {n["id"]: n for n in graph["nodes"]}
    dimensions = {n["id"]: n["size_m"] for n in graph["nodes"] if n["kind"] in ("object", "support")}
    for sample in samples:
        physical = validate_graph(sample, expected)
        validate_sample(sample)
        # sample_scenic changes sampling provenance, not the coordinate frame or
        # scene assumptions. Preserve all other graph-level fields exactly.
        sampled_fields = {"nodes", "edges", "source", "sample_index", "sample_seed", "variation", "tracks"}
        if {k: v for k, v in sample.items() if k not in sampled_fields} != {
            k: v for k, v in graph.items() if k not in sampled_fields
        }:
            raise ValueError("Scenic graph metadata disagrees with source")
        # Preserve the complete edge multiset, including relation metadata. Only
        # support-edge evidence has a documented sampled form. Matching consumes
        # edges so duplicates, additions and removals cannot silently pass.
        remaining = copy.deepcopy(graph.get("edges", []))
        for edge in sample.get("edges", []):
            match = next(
                (
                    index
                    for index, original in enumerate(remaining)
                    if edge == original
                    or (
                        original.get("relation") == "on"
                        and edge == dict(original, evidence="sampled_constraint")
                    )
                ),
                None,
            )
            if match is None:
                raise ValueError("Scenic relationships disagree with source graph")
            remaining.pop(match)
        if remaining:
            raise ValueError("Scenic relationships are missing from sample")
        if {n["id"] for n in physical} != set(dimensions):
            raise ValueError("Scenic stable IDs disagree with graph")
        if {n["id"] for n in sample["nodes"]} != set(source_nodes):
            raise ValueError("Scenic node inventory disagrees with graph")
        for node in sample["nodes"]:
            original = source_nodes[node["id"]]
            # sample_scenic intentionally changes object positions and converts physical
            # dimensions to floats. Every other per-ID field is copied unchanged.
            # In particular, a set of correct square labels is insufficient if those
            # labels, asset paths or semantics have moved to different stable IDs.
            excluded = {"size_m"} if original["kind"] in ("object", "support") else set()
            if original["kind"] == "object":
                excluded.add("position_m")
            before = {key: value for key, value in original.items() if key not in excluded}
            after = {key: value for key, value in node.items() if key not in excluded}
            if before != after:
                raise ValueError(f"Scenic per-ID metadata disagrees with graph: {node['id']}")
        for node in physical:
            if not np.allclose(node["size_m"], dimensions[node["id"]], atol=1e-8, rtol=0):
                raise ValueError("Scenic dimensions disagree with graph")


def _adversarial(graph, expected):
    validate_graph(graph, expected)
    mutations = {}
    objects = [i for i, n in enumerate(graph["nodes"]) if n["kind"] == "object"]
    bad = copy.deepcopy(graph)
    del bad["nodes"][objects[0]]
    mutations["missing_object"] = bad
    bad = copy.deepcopy(graph)
    bad["nodes"][objects[0]]["position_m"][0] = float("nan")
    mutations["nonfinite_geometry"] = bad
    bad = copy.deepcopy(graph)
    bad["edges"].append(dict(source="nonexistent", relation="on", target=graph["nodes"][0]["id"]))
    mutations["broken_identity"] = bad
    if len(objects) > 1:
        bad = copy.deepcopy(graph)
        bad["nodes"][objects[1]]["position_m"][:2] = bad["nodes"][objects[0]]["position_m"][:2]
        mutations["overlapping_objects"] = bad
    outcomes = {}
    for label, bad in mutations.items():
        try:
            validate_graph(bad, expected)
        except ValueError as error:
            outcomes[label] = dict(rejected=True, reason=str(error))
        else:
            outcomes[label] = dict(rejected=False)
    passed = all(v["rejected"] for v in outcomes.values())
    return _result(
        "pass" if passed else "fail",
        "adversarial.mutations",
        "Concrete graph mutations tested against the same validator",
        outcomes,
        scope="structural exploit checks only; no reward or robot task audit",
    )


def run_checks(case: dict, root: Path, out: Path) -> dict:
    """Return four role verdicts and write each role's independently inspectable evidence."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if case.get("schema") != "creation-audit-case-v1":
        raise ValueError("Unsupported audit case schema")
    expected = case.get("expected_objects")
    if (
        not isinstance(expected, list)
        or not expected
        or any(not isinstance(x, str) or not x for x in expected)
    ):
        raise ValueError("Nonempty explicit expected_objects are required")
    if len(set(expected)) != len(expected):
        raise ValueError("expected_objects must be unique")
    scene = _path(root, case["scene_dir"]) if case.get("scene_dir") else None
    graph, graph_error = None, None
    if scene is not None:
        try:
            graph = json.loads((scene / "scene_graph.json").read_text())
            validate_graph(graph, expected)
        except (OSError, ValueError, KeyError, TypeError) as error:
            graph_error = str(error)
    results = {}
    checks = dict(
        geometry=lambda: _geometry(case, root, graph),
        physics=lambda: _physics(graph, scene, expected),
        exports=lambda: _exports(graph, scene, expected, out),
        adversarial=lambda: _adversarial(graph, expected),
    )
    for role, check in checks.items():
        try:
            if graph_error and (role != "geometry" or case.get("geometry_scope") == "synthetic_fixture"):
                results[role] = _result("fail", f"{role}.graph_invalid", graph_error)
            elif graph is None and role != "geometry":
                results[role] = _result("blocked", f"{role}.scene_missing", "No scene export available")
            else:
                results[role] = check()
        except (ImportError, subprocess.TimeoutExpired) as error:
            results[role] = _result("blocked", f"{role}.unavailable", str(error))
        except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
            results[role] = _result("fail", f"{role}.rejected", str(error))
        (out / f"{role}.json").write_text(json.dumps(results[role], indent=2, allow_nan=False) + "\n")
    return results
