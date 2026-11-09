"""Opt-in removal of convex collision padding below an unchanged visual surface."""

import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh

from lidar_video.scene_program_assets import prepare_asset_graph


def clip_convex_padding(mesh, plane_z, *, max_depth=0.00075):
    """Intersect a convex solid with z >= plane_z, never adding volume or a pad."""
    if not np.isfinite([plane_z, max_depth]).all() or not 0 <= max_depth <= 0.001:
        raise ValueError("Clipping requires a finite plane and a depth budget between zero and 1 mm")
    if (
        not len(mesh.faces)
        or not np.isfinite(mesh.vertices).all()
        or not mesh.is_watertight
        or not mesh.is_convex
        or not mesh.is_winding_consistent
        or mesh.volume <= 0
    ):
        raise ValueError("Collision clipping requires a finite, consistently oriented convex solid")
    depth = max(0.0, float(plane_z - mesh.bounds[0, 2]))
    if depth > max_depth + 1e-10:
        raise ValueError("Collision extension exceeds the clipping depth budget")
    if depth == 0:
        return mesh.copy(), dict(
            changed=False,
            depth_m=0.0,
            volume_before_m3=float(mesh.volume),
            volume_after_m3=float(mesh.volume),
        )
    vertices, edges = mesh.vertices, mesh.edges_unique
    z = vertices[:, 2] - plane_z
    a, b = edges[:, 0], edges[:, 1]
    crossing = ((z[a] < 0) & (z[b] > 0)) | ((z[b] < 0) & (z[a] > 0))
    left, right = vertices[a[crossing]], vertices[b[crossing]]
    fraction = (plane_z - left[:, 2]) / (right[:, 2] - left[:, 2])
    cut = left + fraction[:, None] * (right - left)
    cut[:, 2] = plane_z
    points = np.vstack([vertices[z >= 0], cut])
    if len(points) < 4 or np.linalg.matrix_rank(points - points.mean(0)) < 3:
        raise ValueError("Clipping would remove or flatten a collision part")
    result = trimesh.convex.convex_hull(points)
    if (
        not result.is_watertight
        or not result.is_convex
        or result.volume <= 0
        or result.volume > mesh.volume + 1e-12
        or np.any(result.bounds[0] < mesh.bounds[0] - 1e-10)
        or np.any(result.bounds[1] > mesh.bounds[1] + 1e-10)
        or result.bounds[0, 2] < plane_z - 1e-10
    ):
        raise ValueError("Clipped collision part violates the non-expansion contract")
    return result, dict(
        changed=True, depth_m=depth, volume_before_m3=float(mesh.volume), volume_after_m3=float(result.volume)
    )


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trim_collision_padding(graph_file, out, object_ids, *, max_depth=0.00075, support_tolerance=0.0):
    """Create a scene variant, preserving every visual asset, node pose and edge.

    Only named objects on horizontal box supports are eligible. Unspecified
    collision assets remain byte-identical; original inputs are never replaced.
    """
    path, out = Path(graph_file).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if out.is_relative_to(path.parent):
        raise ValueError("Output must be separate from the source scene")
    if not object_ids or len(set(object_ids)) != len(object_ids):
        raise ValueError("Name distinct, nonempty objects explicitly")
    graph = json.loads(path.read_text())
    source_hash = _digest(path)
    _, audit = prepare_asset_graph(graph, path.parent, support_tolerance=support_tolerance)
    resolved = [(path.parent / p).resolve() for p in audit["assets"]]
    if len(set(resolved)) != len(resolved) or any(".." in Path(p).parts for p in audit["assets"]):
        raise ValueError("Aliased or parent-traversing asset paths are unsupported")
    nodes = {n["id"]: n for n in graph["nodes"]}
    visuals = {n["mesh_path"] for n in graph["nodes"] if "mesh_path" in n}
    replacements, objects = {}, {}
    for name in object_ids:
        node = nodes.get(name, {})
        if node.get("kind") != "object" or not node.get("mesh_path"):
            raise ValueError(f"Select a mesh object: {name}")
        parents = [e["target"] for e in graph["edges"] if e["source"] == name and e["relation"] == "on"]
        if len(parents) != 1:
            raise ValueError("Clipping requires one declared on relation")
        parent = nodes[parents[0]]
        if parent["kind"] != "support" or "mesh_path" in parent:
            raise ValueError("Clipping requires a horizontal box support")
        visual = trimesh.load(path.parent / node["mesh_path"], force="mesh", process=False)
        plane = float(visual.bounds[0, 2])
        gap = plane + node["position_m"][2] - parent["position_m"][2] - parent["size_m"][2] / 2
        if abs(gap) > support_tolerance + 1e-8:
            raise ValueError("Visual base must already meet the declared support within explicit tolerance")
        records = {}
        for relative in node["collision_paths"]:
            if relative in visuals or relative in replacements or relative in records:
                raise ValueError("Collision assets must be separate from visuals and not shared")
            users = [n["id"] for n in graph["nodes"] if relative in n.get("collision_paths", [])]
            if users != [name]:
                raise ValueError("Shared collision assets cannot be modified")
            mesh = trimesh.load(path.parent / relative, force="mesh", process=False)
            clipped, record = clip_convex_padding(mesh, plane, max_depth=max_depth)
            records[relative] = record
            if record["changed"]:
                replacements[relative] = clipped
        before = sum(r["volume_before_m3"] for r in records.values())
        after = sum(r["volume_after_m3"] for r in records.values())
        if 1 - after / before > 0.10:
            raise ValueError("Clipping removes more than 10% of total collision-part volume")
        objects[name] = dict(
            plane_local_z_m=plane,
            visual_support_gap_m=gap,
            parts=records,
            total_volume_loss_fraction=1 - after / before,
        )
    if _digest(path) != source_hash or any(_digest(path.parent / p) != h for p, h in audit["assets"].items()):
        raise ValueError("Source changed during collision clipping")
    out.mkdir(parents=True)
    for relative in audit["assets"]:
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative in replacements:
            target.write_text(
                trimesh.exchange.obj.export_obj(
                    replacements[relative],
                    digits=17,
                    include_normals=False,
                    include_color=False,
                    include_texture=False,
                )
            )
        else:
            shutil.copyfile(path.parent / relative, target)
    result = copy.deepcopy(graph)
    result["collision_padding"] = dict(
        source_graph_sha256=source_hash,
        objects=objects,
        max_depth_m=max_depth,
        scope="bounded collision padding removal; no stability certification",
    )
    # Reload and revalidate the actual serialized meshes against the original envelopes and graph constraints.
    prepare_asset_graph(result, out, support_tolerance=support_tolerance)
    if not all(_digest(out / p) == audit["assets"][p] for p in visuals):
        raise ValueError("Visual asset bytes changed during collision export")
    (out / "scene_graph.json").write_text(json.dumps(result, indent=2) + "\n")
    report = dict(
        result["collision_padding"],
        visual_bytes_unchanged=all(_digest(out / p) == audit["assets"][p] for p in visuals),
        poses_dimensions_edges_unchanged=result["nodes"] == graph["nodes"]
        and result["edges"] == graph["edges"],
    )
    (out / "collision_padding.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
