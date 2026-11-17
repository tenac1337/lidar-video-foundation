"""Prepare asset-backed semantic regions and frozen separating-axis witnesses."""

from itertools import product
from pathlib import Path

import numpy as np
import trimesh

from lidar_video.scene_program import compile_scene_program
from lidar_video.scene_program_assets import _support_band_points, prepare_asset_graph


def prepare_spatial_contract(graph, asset_root, *, jitter=0.004, support_tolerance=0.0):
    """Validate original assets and retain 3D band points for rotated membership.

    The contract is derived from the same reference as the constructive program.
    It does not choose new separation axes from a simulated pose or relax bounds.
    Regions are calibrated definitions attached to declared fixed supports, not
    independently tracked native bodies. All geometry remains in metres.
    """
    proxy, audit = prepare_asset_graph(graph, asset_root, jitter=jitter, support_tolerance=support_tolerance)
    program = compile_scene_program(proxy, jitter=jitter)
    nodes = {n["id"]: n for n in graph["nodes"]}
    regions, footprints = {}, {}
    for edge in graph["edges"]:
        if edge["relation"] != "part_of":
            continue
        name, parent = edge["source"], edge["target"]
        offset = np.asarray(nodes[name]["position_m"]) - np.asarray(nodes[parent]["position_m"])
        half = float(nodes[parent]["square_size_m"]) / 2
        regions[name] = dict(
            parent=parent,
            local_bounds_xy_m=[(offset[:2] - half).tolist(), (offset[:2] + half).tolist()],
            local_z_m=float(offset[2]),
            reference_parent_position_m=nodes[parent]["position_m"],
            reference_parent_rotation=np.eye(3).tolist(),
            evidence="Calibrated fixed-support region definition, not independent tracking",
        )
    signs = np.array(list(product([-1, 1], repeat=3)))
    for index, edge in enumerate(graph["edges"]):
        if edge["relation"] != "occupies":
            continue
        node = nodes[edge["source"]]
        if edge.get("footprint_mode", "envelope") == "support_band":
            meshes = [
                trimesh.load(Path(asset_root) / p, force="mesh", process=False)
                for p in [node["mesh_path"], *node["collision_paths"]]
            ]
            points, _ = _support_band_points(meshes, edge["support_band_height_m"])
            points = np.unique(points, axis=0)
        else:
            points = signs * np.asarray(node["size_m"]) / 2
        footprints[str(index)] = dict(
            source=edge["source"],
            target=edge["target"],
            mode=edge.get("footprint_mode", "envelope"),
            local_points_m=points.tolist(),
        )
    return dict(
        version=1,
        physical_envelopes={n["id"]: n["size_m"] for n in proxy["nodes"]},
        regions=regions,
        footprints=footprints,
        separation_witnesses=program.witnesses,
        asset_sha256=audit["assets"],
        scope="Reference-derived semantic geometry and fixed separation branches; not mesh-level collision or force certification",
    )
