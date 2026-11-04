"""Bounded tabletop reconstruction: horizontal support + separated box proxies.

Each frame is reconstructed independently. No moving-object fusion, object-class
recognition, or complete mesh recovery is claimed. A stationary multi-view fusion
mode is intentionally absent until motion segmentation is available.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from lidar_video.capture import world_points


def fit_support(points, seed=0, tolerance=0.003):
    rng = np.random.default_rng(seed)
    p = points[rng.choice(len(points), min(len(points), 8000), replace=False)]
    best, best_n = None, 0
    for _ in range(120):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        if abs(normal[2]) < np.cos(np.radians(15)):
            continue
        normal *= np.sign(normal[2])
        d = -normal @ a
        mask = np.abs(p @ normal + d) < tolerance
        if mask.sum() > best_n:
            best, best_n = (normal, d, mask), mask.sum()
    if best is None or best_n < max(100, len(p) * 0.2):
        raise ValueError(
            "No dominant near-horizontal support plane; tabletop prototype cannot process this scene"
        )
    plane = p[best[2]]
    center = plane.mean(axis=0)
    normal = np.linalg.svd(plane - center, full_matrices=False)[2][-1]
    normal *= np.sign(normal[2])
    d = -normal @ center
    x = np.array([1.0, 0, 0]) - normal[0] * normal
    x /= np.linalg.norm(x)
    rotation = np.stack([x, np.cross(normal, x), normal])
    scene_from_world = np.eye(4)
    scene_from_world[:3, :3] = rotation
    scene_from_world[2, 3] = d
    return scene_from_world, float(np.sqrt(np.mean((plane @ normal + d) ** 2)))


def voxel_points(points, colors, voxel=0.004):
    _, indices = np.unique(np.floor(points / voxel).astype(int), axis=0, return_index=True)
    return points[indices], colors[indices]


def clusters(points, radius=0.014, min_points=15):
    if len(points) < min_points:
        return []
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
    _, labels = connected_components(graph, directed=False)
    return [np.flatnonzero(labels == i) for i, count in enumerate(np.bincount(labels)) if count >= min_points]


def extract_frame(data, frame=0, transform=None):
    n = len(data["depth"])
    k = np.broadcast_to(data["K"], (n, 3, 3))[frame]
    z = data["depth"][frame]
    mask = np.isfinite(z) & (z > 0) & (data["confidence"][frame] >= 1)
    points = world_points(z, k, data["world_from_camera"][frame])[mask]
    colors = data["rgb"][frame][mask] / 255
    if len(points) < 150:
        raise ValueError("Too few points to reconstruct a support plane")
    transform, residual = fit_support(points) if transform is None else (transform, None)
    p = points @ transform[:3, :3].T + transform[:3, 3]
    floor = p[np.abs(p[:, 2]) < 0.003]
    if len(floor) < 100:
        raise ValueError("Support plane is not visible in this frame")
    low, high = np.quantile(floor[:, :2], [0.005, 0.995], axis=0)
    keep = (p[:, 2] > 0.012) & (p[:, 2] < 0.3) & np.all((p[:, :2] >= low) & (p[:, :2] <= high), axis=1)
    objects, rgb = voxel_points(p[keep], colors[keep])
    nodes = []
    for indices in clusters(objects):
        cloud = objects[indices]
        lo, hi = np.quantile(cloud, [0.01, 0.99], axis=0)
        # Bottom is occluded: this is an explicit resting-on-support hypothesis.
        lo[2] = 0
        size = hi - lo
        if np.any(size < 0.012) or np.any(size > 0.3):
            continue
        nodes.append(
            dict(
                id="",
                kind="object",
                semantic_class="unknown",
                geometry="axis_aligned_box_proxy",
                position_m=((hi + lo) / 2).tolist(),
                size_m=size.tolist(),
                color_rgb=np.median(rgb[indices], axis=0).tolist(),
                observed_points=len(indices),
                evidence="depth_cluster",
                assumptions=["resting on support", "axis-aligned box proxy"],
                uncertainty=dict(occluded_geometry="unknown", sensor_accuracy="not calibrated"),
            )
        )
    nodes.sort(key=lambda v: v["position_m"][0])
    for i, node in enumerate(nodes):
        node["id"] = f"object_{i:03d}"
    support = dict(
        id="support",
        kind="support",
        semantic_class="support_surface",
        position_m=[*((low + high) / 2).tolist(), -0.015],
        size_m=[*(high - low).tolist(), 0.03],
        color_rgb=[0.7, 0.7, 0.7],
        evidence="plane_fit",
        assumptions=["fixed support", "3 cm proxy thickness"],
        plane_rmse_m=residual,
    )
    return support, nodes, transform, p, colors


def scene_graph(data):
    support, nodes, transform, cloud, colors = extract_frame(data)
    if not nodes:
        raise ValueError(
            "No separated tabletop objects found; use taller objects or provide a different scene"
        )
    tracks = [dict(frame=0, time_s=float(data["rgb_time"][0]), nodes=nodes)]
    previous = nodes
    next_id = len(nodes)
    for frame in range(1, len(data["depth"])):
        _, current, _, _, _ = extract_frame(data, frame, transform)
        assigned = set()
        if previous and current:
            a = np.array([n["position_m"] for n in previous])
            b = np.array([n["position_m"] for n in current])
            distances = np.linalg.norm(a[:, None] - b, axis=-1)
            rows, cols = linear_sum_assignment(distances)
            for i, j in zip(rows, cols):
                if distances[i, j] < 0.06:
                    current[j]["id"] = previous[i]["id"]
                    assigned.add(j)
        for j, node in enumerate(current):
            if j not in assigned:
                node["id"] = f"object_{next_id:03d}"
                next_id += 1
        tracks.append(dict(frame=frame, time_s=float(data["rgb_time"][frame]), nodes=current))
        previous = current
    edges = [
        dict(
            source=n["id"],
            relation="on",
            target="support",
            evidence="geometric_hypothesis",
            frame=0,
            time_s=float(data["rgb_time"][0]),
        )
        for n in nodes
    ]
    graph = dict(
        schema_version="1.0",
        units="metres",
        coordinate_frame="support-aligned z-up",
        source="RGB-D geometry; no semantic recognition",
        frame=0,
        time_s=float(data["rgb_time"][0]),
        scene_from_world=transform.tolist(),
        nodes=[support, *nodes],
        edges=edges,
        tracks=tracks,
        limits=[
            "box proxies only",
            "horizontal support within 15 degrees of world up",
            "nearest-position tracking with 6 cm gate; identities may change after occlusion",
            "no chess piece classification, legal-move inference, or hand reconstruction",
        ],
    )
    return graph, cloud, colors


def export_ply(path, points, colors):
    with open(path, "w") as file:
        file.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(points, np.clip(colors * 255, 0, 255).astype(int)):
            file.write(" ".join([*(f"{v:.6f}" for v in point), *(str(v) for v in color)]) + "\n")
