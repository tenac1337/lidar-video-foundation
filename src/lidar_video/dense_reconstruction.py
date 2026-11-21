"""Reproducible CPU dense reconstruction from a calibrated RGB capture NPZ.

Run with ``python -m lidar_video.dense_reconstruction --help``. Output is an
evidence packet of open observed surfaces, not automatically promoted assets.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import trimesh

from lidar_video.camera_validation import rigid
from lidar_video.capture import camera_points
from lidar_video.dense_surface import (
    DensePolicy,
    close_view_triangles,
    corroborate_pairs,
    plane_sweep_pair,
)
from lidar_video.foreground_surface import open_depth_mesh
from lidar_video.metric_fusion import _camera
from lidar_video.surface_fragments import support_fragment_hypotheses


def select_neighbors(poses, sources, references, *, minimum_baseline=0.01):
    """Select up to four neighbors from calibration alone, before inference."""
    selected, rejected = {}, {}
    index = {s: i for i, s in enumerate(sources)}
    for s in references:
        p = poses[index[s]]
        eligible, failures = [], []
        for t in sources:
            if t == s:
                continue
            q = poses[index[t]]
            baseline = float(np.linalg.norm(p[:3, 3] - q[:3, 3]))
            angle = float(np.rad2deg(np.arccos(np.clip((np.trace(p[:3, :3].T @ q[:3, :3]) - 1) / 2, -1, 1))))
            if minimum_baseline <= baseline <= 0.35 and angle <= 35:
                eligible.append((abs(baseline - 0.06) + 0.002 * angle, t))
            else:
                failures.append(dict(source=t, baseline_m=baseline, rotation_degrees=angle))
        selected[s] = [t for _, t in sorted(eligible)[:4]]
        rejected[s] = failures
    return selected, rejected


def reconstruct_dense(
    data,
    output,
    *,
    reference_ids,
    excluded_ids=(),
    static_scene=False,
    policy=DensePolicy(),
    max_width=480,
    z_up_support_fragments=False,
):
    """Use construction images only; exclusions are rejected before output writes.

    Input keys: rgb uint8(N,H,W,3), K (3,3) or (N,3,3), world_from_camera
    (N,4,4), source_frame unique integer(N). Other fields, including depth and
    confidence, are ignored. Metric camera provenance remains the caller's
    responsibility. Resource bounds are explicit; no GPU dependency is required.
    """
    policy.validate()
    if type(z_up_support_fragments) is not bool:
        raise ValueError("Explicit boolean z-up support-fragment option required")
    if static_scene is not True:
        raise ValueError("Static-scene assumption must be explicit")
    rgb = np.asarray(data["rgb"])
    ids = np.asarray(data["source_frame"])
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or not 3 <= len(rgb) <= 64:
        raise ValueError("3..64 uint8 RGB construction images required")
    if (
        ids.shape != (len(rgb),)
        or not np.issubdtype(ids.dtype, np.integer)
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("Unique integer source IDs required")
    sources = [int(s) for s in ids]
    references = list(reference_ids)
    if not references or len(set(references)) != len(references) or not set(references) <= set(sources):
        raise ValueError("Unique available reference IDs required")
    if set(sources) & set(excluded_ids):
        raise ValueError("Excluded review/evaluation source occurs in reconstruction input")
    poses = np.asarray(data["world_from_camera"], float)
    if poses.shape != (len(rgb), 4, 4):
        raise ValueError("One rigid pose per image required")
    for p in poses:
        rigid(p, "world_from_camera")
    intrinsics = np.broadcast_to(np.asarray(data["K"], float), (len(rgb), 3, 3)).copy()
    for k in intrinsics:
        _camera(k)
    if type(max_width) is not int or not 32 <= max_width <= 960:
        raise ValueError("Working width must be 32..960")
    h, w = rgb.shape[1:3]
    if min(h, w) < 10:
        raise ValueError("Nonempty images required")
    scale = min(1.0, max_width / w)
    out_w, out_h = round(w * scale), round(h * scale)
    if policy.planes * out_h * out_w > 180_000_000:
        raise ValueError("Working image exceeds cost volume memory budget")
    sx, sy = out_w / w, out_h / h
    intrinsics[:, 0, 0] *= sx
    intrinsics[:, 1, 1] *= sy
    intrinsics[:, 0, 2] = (intrinsics[:, 0, 2] + 0.5) * sx - 0.5
    intrinsics[:, 1, 2] = (intrinsics[:, 1, 2] + 0.5) * sy - 0.5
    images = {
        s: cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA) for s, image in zip(sources, rgb)
    }
    index = {s: i for i, s in enumerate(sources)}
    neighbors, failures = select_neighbors(poses, sources, references, minimum_baseline=policy.min_baseline_m)
    edges = set()
    for s, ns in neighbors.items():
        for a in [s] + ns:
            for b in [s] + ns:
                if a != b:
                    edges.add((a, b))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    for name in ["pairs", "frames", "meshes", "clouds"]:
        (out / name).mkdir()

    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    write(
        "plan.json",
        dict(
            policy=asdict(policy),
            construction_sources=sources,
            reference_sources=references,
            excluded_sources=list(excluded_ids),
            neighbors=neighbors,
            selection_rejections=failures,
            working_size=[out_w, out_h],
            z_up_support_fragments=z_up_support_fragments,
            static_scene_assumed=True,
            automatic_asset_promotion=False,
            modality="RGB geometry using supplied metric camera poses; no sensor depth read",
            scope="Observed surfaces; no object identity, hidden geometry or independent accuracy claim",
        ),
    )
    np.savez(out / "cameras.npz", sources=ids, K=intrinsics, poses=poses)
    ledger = []
    for a, b in sorted(edges):
        i, j = index[a], index[b]
        try:
            r = plane_sweep_pair(
                images[a],
                images[b],
                intrinsics[i],
                intrinsics[j],
                poses[i],
                poses[j],
                source_ids=[a, b],
                policy=policy,
                static_scene=True,
            )
            z = r["depth"]
            metadata = r["metadata"]
        except ValueError as error:
            z = np.zeros((out_h, out_w), np.float32)
            metadata = dict(source_ids=[a, b], failure=str(error))
        np.savez_compressed(out / "pairs" / f"{a}-{b}.npz", depth=z)
        ledger.append(metadata)
    write("pair-ledger.json", ledger)

    def depth(a, b):
        with np.load(out / "pairs" / f"{a}-{b}.npz") as f:
            return f["depth"]

    reports, points, colors = [], [], []
    for s in references:
        i = index[s]
        ns = neighbors[s]
        if len(ns) < 2:
            reports.append(dict(source=s, accepted_pixels=0, reason="insufficient eligible witnesses"))
            continue
        props = [
            dict(
                id=t,
                depth=depth(s, t),
                reverse_depth=depth(t, s),
                K=intrinsics[index[t]],
                pose=poses[index[t]],
            )
            for t in ns
        ]
        r = corroborate_pairs(props, intrinsics[i], poses[i], reference_id=s, policy=policy)
        third_edges = {(a, b): depth(a, b) for a in ns for b in ns if a != b}
        c = close_view_triangles(
            r["depth"], r["source_indices"], intrinsics[i], poses[i], props, third_edges, policy=policy
        )
        z, keep = c["depth"], c["keep"]
        world = camera_points(z, intrinsics[i]) @ poses[i, :3, :3].T + poses[i, :3, 3]
        points.append(world[keep])
        colors.append(images[s][keep])
        mesh, mr = open_depth_mesh(z, intrinsics[i], poses[i], keep, stride=1)
        if len(mesh.faces):
            cp = np.einsum("ij,jk->ik", np.asarray(mesh.vertices) - poses[i, :3, 3], poses[i, :3, :3])
            uv = np.einsum("ij,kj->ik", cp, intrinsics[i])
            xy = np.rint(uv[:, :2] / uv[:, 2:]).astype(int)
            mesh.visual.vertex_colors = images[s][xy[:, 1], xy[:, 0]]
            mesh.export(out / "meshes" / f"{s}-observed.ply")
        np.savez_compressed(
            out / "frames" / f"{s}.npz",
            depth=z,
            reciprocal_depth=r["depth"],
            source_indices=r["source_indices"],
            rgb=images[s],
            K=intrinsics[i],
            pose=poses[i],
        )
        reports.append(dict(source=s, reciprocal=r["metadata"], triangle=c["metadata"], mesh=mr))
    p = np.concatenate(points) if points else np.zeros((0, 3))
    color = np.concatenate(colors) if colors else np.zeros((0, 3), np.uint8)
    _, ix = np.unique(np.floor(p / 0.003).astype(np.int64), axis=0, return_index=True)
    ix = np.sort(ix)
    np.save(out / "clouds/dense_rgb.npy", p[ix])
    np.save(out / "clouds/dense_rgb_colors.npy", color[ix])
    trimesh.points.PointCloud(p[ix], colors=color[ix]).export(out / "clouds/dense_rgb.ply")
    summary = dict(
        frames=reports, before_voxel=len(p), after_voxel=len(ix), complete_object_reconstruction=False
    )
    if z_up_support_fragments:
        fragments = support_fragment_hypotheses(p[ix], z_up=True)
        write("fragment-hypotheses.json", dict(report=fragments["report"], fragments=fragments["fragments"]))
        np.save(out / "clouds/fragment_labels.npy", fragments["labels"])
        (out / "fragments").mkdir()
        for fragment in fragments["fragments"]:
            selected = fragment["point_indices"]
            trimesh.points.PointCloud(p[ix][selected], colors=color[ix][selected]).export(
                out / "fragments" / f"fragment-{fragment['fragment_id']:03d}.ply"
            )
        summary["support_fragments"] = fragments["report"]
    write("results.json", summary)
    write(
        "SHA256.json",
        {
            str(f.relative_to(out)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(out.rglob("*"))
            if f.is_file()
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-source-ids", required=True, nargs="+", type=int)
    parser.add_argument("--excluded-source-ids", nargs="*", type=int, default=[])
    parser.add_argument("--static-scene", action="store_true", required=True)
    parser.add_argument("--max-width", type=int, default=480)
    parser.add_argument(
        "--z-up-support-fragments",
        action="store_true",
        help="Explicitly assume gravity-aligned z-up to report tabletop fragment hypotheses",
    )
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        result = reconstruct_dense(
            data,
            args.output,
            reference_ids=args.reference_source_ids,
            excluded_ids=args.excluded_source_ids,
            static_scene=args.static_scene,
            max_width=args.max_width,
            z_up_support_fragments=args.z_up_support_fragments,
        )
    print(json.dumps(dict(before_voxel=result["before_voxel"], after_voxel=result["after_voxel"])))


if __name__ == "__main__":
    main()
