"""Generic static-instance association and mesh-to-scene adapter.

No chess cells, object names, measured scale fitting or manual hidden completion.
Input masks/mesh priors are supplied by a separate perception adapter. This module
validates those hypotheses; it is not itself a learned perception model.
"""

import copy
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from lidar_video.capture import validate, write_json
from lidar_video.geometry import fit_support
from lidar_video.mesh_alignment import silhouette_iou
from lidar_video.native_depth import sample_ids, sensor_camera_points
from lidar_video.object_evidence import distinct_view, reprojection_agreement, unique_points
from lidar_video.reconstruction import rigid
from lidar_video.scene_program_assets import prepare_asset_graph


@dataclass(frozen=True)
class InstancePolicy:
    min_native_points: int = 24
    min_valid_fraction: float = 0.65
    association_distance_m: float = 0.04
    association_fraction: float = 0.35
    depth_median_limit_m: float = 0.025
    depth_p90_limit_m: float = 0.05
    silhouette_min_iou: float = 0.45
    support_tolerance_m: float = 0.005
    min_dimension_m: float = 0.004
    max_dimension_m: float = 2.0
    seed: int = 7

    def __post_init__(self):
        if type(self.min_native_points) is not int or self.min_native_points < 3:
            raise ValueError("min_native_points must be at least three")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be uint32")
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"Invalid policy value: {name}")
        for value in (self.min_valid_fraction, self.association_fraction, self.silhouette_min_iou):
            if not 0 < value <= 1:
                raise ValueError("Fractions must lie in (0, 1]")
        if not 0 < self.support_tolerance_m <= 0.005:
            raise ValueError("Support tolerance must be in (0, 5 mm]")
        if not 0 < self.min_dimension_m < self.max_dimension_m:
            raise ValueError("Invalid dimension range")
        if min(self.association_distance_m, self.depth_median_limit_m, self.depth_p90_limit_m) <= 0:
            raise ValueError("Distance tolerances must be positive")


def world_points(data, frame):
    pose = data["world_from_camera"][frame]
    return sensor_camera_points(data, frame) @ pose[:3, :3].T + pose[:3, 3]


def observation(data, frame, mask, policy):
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != data["depth"][frame].shape or not mask.any():
        raise ValueError("Detection mask must be a nonempty registered boolean image")
    ids = sample_ids(data, frame)
    h, w = mask.shape
    nh, nw = data.get("native_depth_shape", (h, w))
    ry, rx = max(1, round(h / nh)), max(1, round(w / nw))
    core = cv2.erode(
        mask.astype("uint8"),
        np.ones((2 * ry + 1, 2 * rx + 1), "uint8"),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    depth = data["depth"][frame]
    good = np.isfinite(depth) & (depth > 0) & (data["confidence"][frame] >= 1) & (ids >= 0)
    cloud = unique_points(world_points(data, frame), ids, core & good)
    fraction = float(good[core].mean()) if core.any() else 0.0
    quality = dict(
        native_points=len(cloud),
        valid_core_fraction=fraction,
        passed=len(cloud) >= policy.min_native_points and fraction >= policy.min_valid_fraction,
    )
    return dict(frame=frame, mask=mask.copy(), trusted=core & good, cloud=cloud, quality=quality)


def associate_instances(data, detections, *, static_scene, policy=None):
    """Conservative correspondence across arbitrary labels; ambiguous groups abstain.

    A pair needs mutual 3D overlap and bidirectional, occlusion-aware reprojection.
    A component may contain at most one instance per source view; transitive
    matches must also agree for every pair. Rejected observations remain reported.
    """
    validate(data)
    if static_scene is not True:
        raise ValueError("Static-scene assumption must be explicitly declared")
    policy = policy or InstancePolicy()
    sources = data.get("source_frame")
    if sources is None or len(set(sources.tolist())) != len(sources):
        raise ValueError("Unique original source_frame IDs required")
    if len(detections) < 2:
        raise ValueError("At least two detection views required")
    observations, rejected = [], []
    for frame, items in sorted(detections.items()):
        if type(frame) is not int or not 0 <= frame < len(data["depth"]):
            raise ValueError("Detection frame outside construction bundle")
        occupied = np.zeros(data["depth"].shape[1:], bool)
        for index, item in enumerate(items):
            obs = observation(data, frame, item["mask"], policy)
            if np.any(occupied & obs["mask"]):
                raise ValueError("Overlapping instance masks require explicit detector disambiguation")
            occupied |= obs["mask"]
            obs.update(detection=index, source_frame=int(sources[frame]))
            if not obs["quality"]["passed"]:
                rejected.append(
                    dict(
                        frame=frame,
                        source_frame=int(sources[frame]),
                        detection=index,
                        reason="unreliable native depth",
                        quality=obs["quality"],
                    )
                )
            else:
                observations.append(obs)
    network = nx.Graph()
    network.add_nodes_from(range(len(observations)))
    pair_checks = {}
    for i, left in enumerate(observations):
        for j in range(i + 1, len(observations)):
            right = observations[j]
            if left["frame"] == right["frame"]:
                continue
            fractions = [
                float(np.mean(cKDTree(b).query(a)[0] <= policy.association_distance_m))
                for a, b in [(left["cloud"], right["cloud"]), (right["cloud"], left["cloud"])]
            ]
            if min(fractions) < policy.association_fraction:
                pair_checks[(i, j)] = dict(
                    spatial_fractions=fractions, passed=False, reason="insufficient mutual surface proximity"
                )
                continue
            a = reprojection_agreement(
                left["cloud"], right["mask"], data, {"board_from_world": np.eye(4)}, right["frame"]
            )
            b = reprojection_agreement(
                right["cloud"], left["mask"], data, {"board_from_world": np.eye(4)}, left["frame"]
            )
            pair_checks[(i, j)] = dict(
                forward=a, backward=b, spatial_fractions=fractions, passed=a["passed"] and b["passed"]
            )
            if a["passed"] and b["passed"]:
                network.add_edge(i, j)
    groups = []
    for indices in nx.connected_components(network):
        indices = sorted(indices)
        group = [observations[i] for i in indices]
        reason = None
        if len(group) < 2:
            reason = "no corroborating instance"
        elif len({o["frame"] for o in group}) != len(group):
            reason = "ambiguous many-to-one association"
        elif any(not network.has_edge(i, j) for n, i in enumerate(indices) for j in indices[n + 1 :]):
            reason = "transitive association lacks pairwise corroboration"
        elif not any(distinct_view(data, a["frame"], b["frame"]) for a in group for b in group):
            reason = "no distinct camera viewpoints"
        if reason:
            rejected.append(
                dict(
                    observations=[dict(frame=o["frame"], detection=o["detection"]) for o in group],
                    reason=reason,
                )
            )
        else:
            groups.append(group)
    groups.sort(key=lambda g: tuple(np.median(np.concatenate([o["cloud"] for o in g]), axis=0)))
    selected = {}
    for index, group in enumerate(groups):
        name = f"object_{index:03d}"
        reference = max(group, key=lambda o: (len(o["cloud"]), -o["source_frame"]))
        selected[name] = dict(reference=reference, observations=group)
    return selected, dict(
        policy=asdict(policy),
        accepted_instances=len(selected),
        observations=[
            dict(
                frame=o["frame"],
                source_frame=o["source_frame"],
                detection=o["detection"],
                quality=o["quality"],
            )
            for o in observations
        ],
        pair_checks=[dict(left=i, right=j, **check) for (i, j), check in pair_checks.items()],
        rejected=rejected,
        scope="Static-instance correspondence hypotheses, not complete object detection.",
    )


def infer_support(data, detections):
    """Dominant horizontal background plane; rejects inadequate support evidence."""
    points = []
    for frame, items in sorted(detections.items()):
        mask = np.isfinite(data["depth"][frame]) & (data["depth"][frame] > 0)
        mask &= data["confidence"][frame] >= 1
        for item in items:
            mask &= ~item["mask"]
        points.append(unique_points(world_points(data, frame), sample_ids(data, frame), mask))
    points = np.concatenate(points)
    if len(points) < 150:
        raise ValueError("Insufficient background points for support")
    transform, residual = fit_support(points)
    p = points @ transform[:3, :3].T + transform[:3, 3]
    floor = p[np.abs(p[:, 2]) < 0.003]
    bounds = np.quantile(floor[:, :2], [0.005, 0.995], axis=0)
    return dict(
        scene_from_world=transform.tolist(),
        bounds_xy_m=bounds.tolist(),
        plane_rmse_m=residual,
        evidence="Dominant near-horizontal construction-depth plane; identity as tabletop is a hypothesis.",
    )


def rectangular_top(meshes, bounds):
    """Certify the envelope's full top rectangle in the supplied geometry.

    This deliberately excludes curved tops, holes, and nonrectangular platforms:
    the current sampler permits a child anywhere in the parent's XY envelope.
    Triangle union, rather than summed area or point probes, detects holes.
    This is a geometric certificate of the prior, not of an unseen real surface.
    """
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    surfaces = []
    for mesh in meshes:
        triangles = mesh.triangles
        flat = np.all(abs(triangles[:, :, 2] - bounds[1, 2]) <= 1e-7, axis=1)
        surfaces.extend(Polygon(t[:, :2]) for t in triangles[flat])
    if not surfaces:
        return False
    target = box(*bounds[:, :2].ravel())
    return bool(unary_union(surfaces).buffer(1e-7).covers(target))


def build_mesh_scene(data, selected, priors, support, out, *, policy=None, jitter=0.03):
    """Validate external mesh priors and generate an asset-bearing generic graph.

    Priors are world-frame visual meshes with world-frame convex collision parts.
    Metric scale and poses are never optimized to hide disagreement. Conservative
    full-footprint support is checked on collision envelopes; unsupported contacts
    fail. A small explicit base adjustment is the only geometry placement change.
    """
    policy = policy or InstancePolicy()
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) or name == "support" for name in selected):
        raise ValueError("Unsafe or reserved object identifier")
    if not selected or set(selected) != set(priors):
        raise ValueError("Mesh prior inventory must exactly match corroborated instances")
    T = rigid(support["scene_from_world"], "scene_from_world")
    bounds = np.asarray(support["bounds_xy_m"], float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
        raise ValueError("Support bounds must be finite increasing XY bounds")
    prepared = {}
    for name, item in selected.items():
        prior = priors[name]
        mesh = prior["visual"].copy()
        collisions = [p.copy() for p in prior["collision"]]
        if not collisions or not isinstance(prior.get("provenance"), dict):
            raise ValueError("Explicit collision parts and mesh provenance required")
        for asset in [mesh, *collisions]:
            if not isinstance(asset, trimesh.Trimesh) or not len(asset.faces):
                raise ValueError("Prior must be a triangle mesh")
            if not np.isfinite(asset.vertices).all() or not np.isfinite(asset.area) or asset.area <= 0:
                raise ValueError("Invalid mesh geometry")
            asset.apply_transform(T)
        if any(not p.is_watertight or not p.is_convex for p in collisions):
            raise ValueError("Collision parts must be closed convex meshes")
        collision_bounds = np.array(
            [
                np.min([p.bounds[0] for p in collisions], axis=0),
                np.max([p.bounds[1] for p in collisions], axis=0),
            ]
        )
        if np.max(abs(collision_bounds - mesh.bounds)) > policy.support_tolerance_m:
            raise ValueError("Visual/collision envelopes disagree beyond the declared contact tolerance")
        extents = mesh.extents
        if min(extents) < policy.min_dimension_m or max(extents) > policy.max_dimension_m:
            raise ValueError("Mesh dimensions outside declared generic-object range")
        probes = trimesh.sample.sample_surface(mesh, 20000, seed=policy.seed)[0]
        checks = []
        for obs in item["observations"]:
            cloud = obs["cloud"] @ T[:3, :3].T + T[:3, 3]
            distances = cKDTree(probes).query(cloud)[0]
            labels = np.zeros(obs["mask"].shape, np.uint16)
            labels[obs["mask"]] = 1
            for other in selected.values():
                if other is item:
                    continue
                for o in other["observations"]:
                    if o["frame"] == obs["frame"]:
                        labels[o["mask"]] = 2
            iou = silhouette_iou(mesh, labels, 1, data, {"board_from_world": T}, obs["frame"])
            camera = (T @ data["world_from_camera"][obs["frame"]])[:3, 3]
            checked = cloud[np.linspace(0, len(cloud) - 1, min(1024, len(cloud)), dtype=int)]
            rays = checked - camera
            observed_range = np.linalg.norm(rays, axis=1)
            rays /= observed_range[:, None]
            hit, ray_id, _ = mesh.ray.intersects_location(
                np.tile(camera, (len(rays), 1)), rays, multiple_hits=True
            )
            first = np.full(len(rays), np.inf)
            np.minimum.at(first, ray_id, np.linalg.norm(hit - camera, axis=1))
            hit_fraction = float(np.isfinite(first).mean())
            ray_error = abs(first[np.isfinite(first)] - observed_range[np.isfinite(first)])
            ray_median = float(np.median(ray_error)) if len(ray_error) else None
            ray_p90 = float(np.quantile(ray_error, 0.9)) if len(ray_error) else None
            if (
                hit_fraction < policy.min_valid_fraction
                or ray_median is None
                or ray_median > policy.depth_median_limit_m
                or ray_p90 > policy.depth_p90_limit_m
            ):
                raise ValueError(f"Mesh prior disagrees with first-visible-surface depth for {name}")
            check = dict(
                source_frame=obs["source_frame"],
                ray_hit_fraction=hit_fraction,
                ray_range_median_error_m=ray_median,
                ray_range_p90_error_m=ray_p90,
                median_m=float(np.median(distances)),
                p90_m=float(np.quantile(distances, 0.9)),
                silhouette_iou=iou,
            )
            if (
                check["median_m"] > policy.depth_median_limit_m
                or check["p90_m"] > policy.depth_p90_limit_m
                or iou < policy.silhouette_min_iou
            ):
                raise ValueError(f"Mesh prior disagrees with construction evidence for {name}: {check}")
            checks.append(check)
        envelope = np.array(
            [
                np.min([p.bounds[0] for p in [mesh, *collisions]], axis=0),
                np.max([p.bounds[1] for p in [mesh, *collisions]], axis=0),
            ]
        )
        prepared[name] = dict(
            mesh=mesh,
            collision=collisions,
            bounds=envelope,
            checks=checks,
            rectangular_top_verified=(
                rectangular_top([mesh], envelope) and rectangular_top(collisions, envelope)
            ),
            provenance=copy.deepcopy(prior["provenance"]),
        )
    nodes = [
        dict(
            id="support",
            kind="support",
            fixed=True,
            position_m=[*bounds.mean(0), -0.015],
            size_m=[*(bounds[1] - bounds[0]), 0.03],
            color_rgb=[0.5, 0.5, 0.5],
            evidence=copy.deepcopy(support),
        )
    ]
    edges = []
    # Bottom-up support selection, including supported objects as potential parents.
    for name in sorted(prepared, key=lambda n: prepared[n]["bounds"][0, 2]):
        item = prepared[name]
        low, high = item["bounds"]
        candidates = []
        for parent in nodes:
            center, size = np.array(parent["position_m"]), np.array(parent["size_m"])
            plow, phigh = center - size / 2, center + size / 2
            gap = low[2] - phigh[2]
            if (
                abs(gap) <= policy.support_tolerance_m
                and np.all(low[:2] >= plow[:2])
                and np.all(high[:2] <= phigh[:2])
            ):
                # Curved parent envelopes do not certify a flat supporting surface.
                if parent["id"] != "support" and not prepared[parent["id"]]["rectangular_top_verified"]:
                    continue
                candidates.append((abs(gap), parent["id"], gap))
        if len(candidates) != 1:
            raise ValueError(f"Missing or ambiguous full-footprint support for {name}")
        _, parent, gap = candidates[0]
        shift = np.array([0.0, 0.0, -gap])
        for asset in [item["mesh"], *item["collision"]]:
            asset.apply_translation(shift)
        low, high = item["bounds"] + shift
        center = (low + high) / 2
        for asset in [item["mesh"], *item["collision"]]:
            asset.apply_translation(-center)
        colors = np.concatenate(
            [data["rgb"][o["frame"]][o["trusted"]] for o in selected[name]["observations"]]
        )
        node = dict(
            id=name,
            kind="object",
            position_m=center.tolist(),
            size_m=(high - low).tolist(),
            color_rgb=(np.median(colors, axis=0) / 255).tolist(),
            mass_kg=0.08,
            mesh_path=f"assets/{name}/visual.obj",
            collision_paths=[f"assets/{name}/collision_{i:03d}.obj" for i in range(len(item["collision"]))],
            evidence=dict(
                mesh_prior=item["provenance"],
                construction_checks=item["checks"],
                checks_basis="unadjusted world-frame prior; support shift separately bounded",
                rectangular_top_verified=item["rectangular_top_verified"],
                support_adjustment_m=-float(gap),
                observed_source_frames=[o["source_frame"] for o in selected[name]["observations"]],
            ),
            uncertainty=dict(
                hidden_geometry="mesh prior, not directly measured",
                mass="fixed engineering prior, not estimated",
                appearance="median observed color; native texture transfer not implemented",
            ),
        )
        nodes.append(node)
        edges.append(dict(source=name, target=parent, relation="on", evidence="flat support hypothesis"))
    graph = dict(
        schema_version=2,
        units="metres",
        axes="scene XYZ, z up",
        time_s=0,
        tracks=[],
        source="generic metric instance meshes",
        scene_from_world=T.tolist(),
        nodes=nodes,
        edges=edges,
        assumptions=dict(
            static_scene=True,
            mass_kg=0.08,
            fixed_mesh_orientations=True,
            relation_scope="support plus conservative non-overlap branch",
            geometry_scope="externally supplied mesh priors; no hidden-shape accuracy guarantee",
        ),
    )
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    for name, item in prepared.items():
        folder = out / "assets" / name
        folder.mkdir(parents=True)
        item["mesh"].export(folder / "visual.obj")
        for index, mesh in enumerate(item["collision"]):
            mesh.export(folder / f"collision_{index:03d}.obj")
    write_json(out / "scene_graph.json", graph)
    # Validation is mandatory; the graph cannot silently omit an unsupported relation.
    prepare_asset_graph(graph, out, jitter=jitter)
    return graph
