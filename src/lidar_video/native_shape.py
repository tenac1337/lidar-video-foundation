"""Transfer bounded support-frame visual refinements to matching native assets.

This adapter intentionally accepts only direct, unrotated mesh bodies with a
free joint and explicit per-geom masses. It preserves semantic edges and applies
the same affine map to every visual and collision vertex. A valid export is not
a stability certificate: compare passive native rollouts separately.
"""

import copy
import hashlib
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from lidar_video.scene_program_assets import prepare_asset_graph


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_affine(value):
    """Accept bounded horizontal yaw, uniform width and independent height only."""
    a = np.asarray(value, float)
    if a.shape != (4, 4) or not np.isfinite(a).all() or not np.allclose(a[3], [0, 0, 0, 1], atol=1e-10):
        raise ValueError("Finite affine matrix required")
    linear = a[:3, :3]
    width = np.linalg.norm(linear[:2, 0])
    height = linear[2, 2]
    yaw = np.arctan2(linear[1, 0], linear[0, 0])
    expected = np.array(
        [
            [width * np.cos(yaw), -width * np.sin(yaw), 0],
            [width * np.sin(yaw), width * np.cos(yaw), 0],
            [0, 0, height],
        ]
    )
    if not (
        0.85 <= width <= 1.15 and 0.85 <= height <= 1.15 and abs(yaw) <= np.deg2rad(30)
    ) or not np.allclose(linear, expected, atol=1e-9):
        raise ValueError("Only bounded width/height and horizontal yaw are supported")
    return a


def affine_obj(text, transform):
    """Transform OBJ vertices/normals while preserving UVs, faces and materials."""
    a = np.asarray(transform, float)
    normal_map = np.linalg.inv(a[:3, :3]).T
    lines = []
    for line in text.splitlines(keepends=True):
        fields = line.split()
        if fields and fields[0] in {"v", "vn"}:
            # Reject homogeneous/color extensions instead of corrupting them.
            if len(fields) != 4:
                raise ValueError("Only three-component OBJ vertices/normals are supported")
            values = np.array(fields[1:], float)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite OBJ coordinate")
            if fields[0] == "v":
                values = a[:3, :3] @ values + a[:3, 3]
            else:
                values = normal_map @ values
                if np.linalg.norm(values) <= 1e-12:
                    raise ValueError("Invalid OBJ normal")
                values /= np.linalg.norm(values)
            line = fields[0] + " " + " ".join(format(v, ".17g") for v in values) + "\n"
        lines.append(line)
    return "".join(lines)


def refine_native_assets(graph_file, output, transforms, *, support_tolerance=0.0):
    """Write a separate, graph-validated native variant; never modify the input.

    Matrices act in the native scene's metric support coordinates, not camera
    coordinates. Mass/friction remain declared priors; native inertia must be
    recomputed from the transformed collision parts by the simulator.
    """
    path, out = Path(graph_file).resolve(), Path(output).resolve()
    if out.exists():
        raise FileExistsError(out)
    if out.is_relative_to(path.parent) or not transforms:
        raise ValueError("Nonempty transforms and a separate output required")
    graph = json.loads(path.read_text())
    _, audit = prepare_asset_graph(graph, path.parent, support_tolerance=support_tolerance)
    native = graph.get("native_template", {})
    relative = Path(native.get("scene_path", ""))
    scene = (path.parent / relative).resolve()
    if relative.is_absolute() or not scene.is_relative_to(path.parent) or scene.suffix != ".xml":
        raise ValueError("A local native template is required")
    tree = ET.parse(scene)
    root = tree.getroot()
    if any(root.find(tag) is not None for tag in ["include", "default", "keyframe"]):
        raise ValueError("Implicit defaults, includes and keyframes are unsupported")
    compiler = root.find("compiler")
    if compiler is not None and (
        set(compiler.attrib) - {"meshdir", "texturedir"}
        or any(compiler.get(k, ".") != "." for k in ["meshdir", "texturedir"])
    ):
        raise ValueError("Native asset directories must be explicit and local")
    files = {path: _hash(path), scene: _hash(scene)}
    for element in root.iter():
        if "file" in element.attrib:
            rel = Path(element.get("file"))
            asset = (scene.parent / rel).resolve()
            if rel.is_absolute() or not asset.is_relative_to(path.parent):
                raise ValueError("Native asset escapes source root")
            files[asset] = _hash(asset)
    for rel in audit["assets"]:
        files[path.parent / rel] = audit["assets"][rel]
    result = copy.deepcopy(graph)
    nodes = {n["id"]: n for n in result["nodes"]}
    replacements, records = {}, {}
    mesh_defs = {m.get("name"): m for m in root.findall("asset/mesh")}
    for name, value in transforms.items():
        a = checked_affine(value)
        node = nodes.get(name, {})
        if node.get("kind") != "object" or "mesh_path" not in node:
            raise ValueError("Refinement requires an explicit mesh object")
        body_name = native.get("body_bindings", {}).get(name)
        bodies = root.findall(f"worldbody/body[@name='{body_name}']")
        if len(bodies) != 1:
            raise ValueError("Exactly one direct native body binding required")
        body = bodies[0]
        if (
            set(body.attrib) - {"name", "pos"}
            or body.find("freejoint") is None
            or any(c.tag not in {"freejoint", "geom"} for c in body)
        ):
            raise ValueError("Only unrotated free mesh bodies without explicit inertia are supported")
        position = np.asarray(node["position_m"], float)
        if not np.allclose(np.fromstring(body.get("pos", "0 0 0"), sep=" "), position, atol=1e-9):
            raise ValueError("Native body and graph position disagree")
        paths = [node["mesh_path"], *node["collision_paths"]]
        native_paths = []
        for geom in body.findall("geom"):
            mesh = mesh_defs.get(geom.get("mesh"))
            if (
                geom.get("type") != "mesh"
                or mesh is None
                or set(mesh.attrib) - {"name", "file"}
                or any(
                    k in geom.attrib
                    for k in ["pos", "quat", "euler", "axisangle", "xyaxes", "zaxis", "fromto"]
                )
                or "mass" not in geom.attrib
            ):
                raise ValueError("Explicit untransformed mesh geoms and masses required")
            if len(root.findall(f".//geom[@mesh='{geom.get('mesh')}']")) != 1:
                raise ValueError("Shared native meshes cannot be changed")
            target_file = (scene.parent / mesh.get("file")).resolve()
            if (
                sum(
                    (scene.parent / entry.get("file", "")).resolve() == target_file
                    for entry in root.iter()
                    if "file" in entry.attrib
                )
                != 1
            ):
                raise ValueError("Shared native asset files cannot be changed")
            native_paths.append(str(target_file.relative_to(path.parent)))
        if sorted(native_paths) != sorted(paths):
            raise ValueError("Native meshes and graph assets disagree")
        loaded = [trimesh.load(path.parent / p, force="mesh", process=False) for p in paths]
        for mesh in loaded[1:]:
            if (
                not mesh.is_watertight
                or not mesh.is_convex
                or not mesh.is_winding_consistent
                or mesh.volume <= 0
            ):
                raise ValueError("Finite convex collision solids required")
        transformed = []
        for mesh in loaded:
            moved = mesh.copy()
            moved.apply_translation(position)
            moved.apply_transform(a)
            transformed.append(moved)
        base_before = loaded[0].bounds[0, 2] + position[2]
        if abs(transformed[0].bounds[0, 2] - base_before) > 1e-8:
            raise ValueError("Visual support height must be preserved")
        before_xy = loaded[0].bounds.mean(0)[:2] + position[:2]
        if np.max(np.abs(transformed[0].bounds.mean(0)[:2] - before_xy)) > 0.02:
            raise ValueError("Horizontal displacement exceeds 2 cm")
        bounds = np.array(
            [
                np.min([m.bounds[0] for m in transformed], axis=0),
                np.max([m.bounds[1] for m in transformed], axis=0),
            ]
        )
        center = bounds.mean(axis=0)
        local = a.copy()
        local[:3, 3] += a[:3, :3] @ position - center
        for rel in paths:
            users = [
                n["id"] for n in result["nodes"] if rel in [n.get("mesh_path"), *n.get("collision_paths", [])]
            ]
            if users != [name] or rel in replacements:
                raise ValueError("Shared graph assets cannot be changed")
            replacements[rel] = affine_obj((path.parent / rel).read_text(), local)
        node["position_m"] = center.tolist()
        node["size_m"] = (bounds[1] - bounds[0] + 2e-10).tolist()
        node["geometry_refinement"] = dict(
            scene_affine=a.tolist(),
            source_graph_sha256=files[path],
            mass_policy="unchanged prior; inertia recomputed from collision geometry",
        )
        body.set("pos", " ".join(format(v, ".17g") for v in center))
        records[name] = dict(scene_affine=a.tolist(), asset_count=len(paths), support_height_preserved=True)
    if any(_hash(p) != digest for p, digest in files.items()):
        raise ValueError("Source changed during refinement")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-shape-", dir=out.parent) as tmp:
        stage = Path(tmp) / "scene"
        for source in files:
            target = stage / source.relative_to(path.parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for rel, payload in replacements.items():
            (stage / rel).write_text(payload)
        tree.write(stage / relative, encoding="unicode")
        prepare_asset_graph(result, stage, support_tolerance=support_tolerance)
        result["native_shape_refinement"] = dict(
            objects=records, source_graph_sha256=files[path], physics_validated=False
        )
        (stage / path.name).write_text(json.dumps(result, indent=2) + "\n")
        changed = {path, scene, *(path.parent / p for p in replacements)}
        assert all(
            _hash(stage / p.relative_to(path.parent)) == h for p, h in files.items() if p not in changed
        )
        report = dict(
            objects=records,
            semantic_edges_preserved=result["edges"] == graph["edges"],
            unchanged_files_verified=len(files) - len(changed),
            source_files={str(p.relative_to(path.parent)): h for p, h in files.items()},
            physics_validated=False,
        )
        (stage / "native-shape-report.json").write_text(json.dumps(report, indent=2) + "\n")
        if out.exists():
            raise FileExistsError(out)
        stage.rename(out)
    return report
