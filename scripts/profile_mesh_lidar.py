"""Read-only geometry diagnostics. Fits are experiments, never accepted scene edits."""

import argparse
import json
import time
from pathlib import Path

import chess
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from lidar_video.capture import load_capture, write_json
from lidar_video.reconstruction import calibrated_points, collect_squares


def stats(values):
    a = np.asarray(values)
    if not a.size:
        return None
    return dict(
        n=int(a.size),
        median_mm=float(np.median(a) * 1000),
        p90_mm=float(np.quantile(a, 0.9) * 1000),
        mean_mm=float(a.mean() * 1000),
    )


def subset(p, n=2500):
    p = np.asarray(p).reshape(-1, 3)
    return p[np.linspace(0, len(p) - 1, min(n, len(p)), dtype=int)]


def cells(p, cal):
    xy = p[..., :2].copy()
    xy[..., 1] -= np.sign(xy[..., 1]) * cal.get("center_gap_m", 0) / 2
    ij = np.floor(xy / cal["square_size_m"] + 4).astype(int)
    return ij[..., 0] + 8 * ij[..., 1], np.all((ij >= 0) & (ij < 8), axis=-1)


def transformed_distances(tree, points, center, parameters):
    """World-metre distances to a similarity-transformed sampled surface.

    Query inverse-transformed points against the original tree, then restore the
    distance scale. Omitting the last factor falsely rewards mesh enlargement.
    """
    rotation = Rotation.from_rotvec(parameters[3:6]).as_matrix()
    scale = np.exp(parameters[6])
    inverse = ((points - center - parameters[:3]) @ rotation) / scale + center
    return tree.query(inverse)[0] * scale


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rec, learned = a.recording / "reconstruction", a.recording / "gpu-run/learned-partial"
    local, gpu = load_capture(rec / "refined_capture.npz"), load_capture(learned / "capture.npz")
    if local["depth"].shape[1:] != (192, 256) or gpu["depth"].shape[1:] != (720, 960):
        raise ValueError("These diagnostics expect the saved 256x192 native / 960x720 model capture")
    cal = json.loads((rec / "board_calibration.json").read_text())
    frames = json.loads((rec / "fusion_frames.json").read_text())
    manifest = json.loads((learned / "meshes.json").read_text())["objects"]
    if any(entry["frame"] != 0 for entry in manifest.values()):
        raise ValueError("This run-specific profile expects all meshes inferred from model frame 0")
    objects = {x["square"]: x for x in json.loads((learned / "masks/objects.json").read_text())}
    masks = np.array(Image.open(learned / "masks/00000.png"))
    ref = int(np.flatnonzero(local["source_frame"] == gpu["source_frame"][0])[0])
    B = np.asarray(cal["board_from_world"])
    board_from_camera = B @ gpu["world_from_camera"][0]
    gp = calibrated_points(gpu, cal, 0)
    lp = calibrated_points(local, cal, ref)
    gc, inside = cells(gp, cal)
    valid = (gpu["depth"][0] > 0) & (gpu["confidence"][0] >= 1)
    by_frame = {f: collect_squares(local, cal, [f], min_height=0.015)[0] for f in frames}
    old_single = collect_squares(gpu, cal, [0], learned / "masks", 0.006)[0]
    same_all = collect_squares(gpu, cal, [0], None, 0.006)[0]
    train_frames = [f for f in frames if f != ref][::2]
    test_frames = [f for f in frames if f != ref][1::2]
    # Reproduce the actual packager map; nearest neighbour supplies no new sensor samples.
    yy, xx = np.indices(gpu["depth"].shape[1:], dtype=np.float32)
    ny = np.clip(np.rint(yy / 3.75).astype(int), 0, 191)
    nx = np.clip(np.rint(xx / 3.75).astype(int), 0, 255)
    known = dict(
        source_frame=int(gpu["source_frame"][0]),
        reference_frame=ref,
        pose_max_abs_difference=float(
            np.max(np.abs(local["world_from_camera"][ref] - gpu["world_from_camera"][0]))
        ),
        K_scale_max_abs_difference=float(
            np.max(np.abs(np.diag([3.75, 3.75, 1]) @ local["K"][ref] - gpu["K"]))
        ),
        input_depth_max_abs_difference_m=float(
            np.max(np.abs(np.where(valid, local["depth"][ref][ny, nx], 0) - gpu["depth"][0]))
        ),
        upsampling_ray_displacement=stats(np.linalg.norm(gp[valid] - lp[ny[valid], nx[valid]], axis=1)),
    )
    results = []
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    overlay = gpu["rgb"][0].copy()
    cv2.drawContours(
        overlay,
        cv2.findContours((masks > 0).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        -1,
        (0, 230, 200),
        1,
    )
    for i, (name, entry) in enumerate(sorted(manifest.items())):
        start = time.perf_counter()
        square = chess.parse_square(name)
        pose = json.loads((learned / entry["pose"]).read_text())
        mesh = trimesh.load(learned / entry["mesh"], force="mesh", process=True)
        mesh.apply_transform(board_from_camera @ np.asarray(pose["camera_from_object"]))
        probes, _ = trimesh.sample.sample_surface(mesh, 120000, seed=0)
        tree = cKDTree(probes)
        small, _ = trimesh.sample.sample_surface(mesh, 10000, seed=0)
        small_tree = cKDTree(small)
        own = (masks == objects[name]["label"]) & valid
        owned = gp[own]
        own_fore = gp[own & (gp[..., 2] > 0.006)]
        eroded = (
            cv2.erode((masks == objects[name]["label"]).astype("uint8"), np.ones((9, 9), np.uint8)).astype(
                bool
            )
            & valid
        )
        own_high = gp[own & (gpu["confidence"][0] == 2)]
        index = np.unique(ny[own] * 256 + nx[own])
        native = lp.reshape(-1, 3)[index]
        fuse = np.concatenate([np.asarray(by_frame[f][square]).reshape(-1, 3) for f in frames])
        train = np.concatenate([np.asarray(by_frame[f][square]).reshape(-1, 3) for f in train_frames])
        test = np.concatenate([np.asarray(by_frame[f][square]).reshape(-1, 3) for f in test_frames])
        metrics = {}
        for key, cloud in dict(
            own_mask=owned,
            own_mask_foreground=own_fore,
            own_mask_confidence2=own_high,
            own_mask_eroded=gp[eroded],
            native_under_mask=native,
            old_single_square=old_single[square],
            same_frame_unmasked=same_all[square],
            native_reference_square=by_frame[ref][square],
            fused_square=fuse,
        ).items():
            cloud = np.asarray(cloud).reshape(-1, 3)
            metrics[key] = stats(tree.query(cloud)[0]) if len(cloud) else None
        original = stats(small_tree.query(fuse)[0])
        exact_points = subset(fuse, 250)
        _, exact, _ = trimesh.proximity.closest_point(mesh, exact_points)
        approx = tree.query(exact_points)[0]
        per_frame = []
        for f in frames:
            cloud = np.asarray(by_frame[f][square]).reshape(-1, 3)
            per_frame.append(
                dict(
                    frame=f,
                    source_frame=int(local["source_frame"][f]),
                    residual=stats(tree.query(cloud)[0]) if len(cloud) else None,
                    z_mm=(np.quantile(cloud[:, 2], [0.1, 0.5, 0.9]) * 1000).tolist() if len(cloud) else None,
                )
            )
        # Fit only designated training views. Closest-surface objective is one-sided and diagnostic.
        center = mesh.centroid

        def fit(mode):
            if len(train) < 24 or len(test) < 24:
                return dict(
                    status="insufficient train or held-out points", train_n=len(train), heldout_n=len(test)
                )
            dims = {"translation": 3, "rigid": 6, "similarity": 7}[mode]

            def expand(x):
                p = np.zeros(7)
                p[:dims] = x
                return p

            def residual(x):
                pars = expand(x)
                return transformed_distances(tree, subset(train, 1200), center, pars)

            bound = np.array([0.08] * 3 + [0.7] * 3 + [np.log(1.5)])[:dims]
            opt = least_squares(
                residual,
                np.zeros(dims),
                bounds=(-bound, bound),
                diff_step=1e-3,
                loss="soft_l1",
                f_scale=0.006,
                max_nfev=90,
            )
            pars = expand(opt.x)
            return dict(
                translation_mm=(pars[:3] * 1000).tolist(),
                rotation_degrees=float(np.linalg.norm(pars[3:6]) * 180 / np.pi),
                scale=float(np.exp(pars[6])),
                train_before=stats(tree.query(train)[0]),
                train_after=stats(transformed_distances(tree, train, center, pars)),
                heldout_before=stats(tree.query(test)[0]),
                heldout_after=stats(transformed_distances(tree, test, center, pars)),
                own_mask_after=stats(transformed_distances(tree, owned, center, pars)),
                converged=bool(opt.success),
                evaluations=int(opt.nfev),
            )

        experiments = {mode: fit(mode) for mode in ["translation", "rigid", "similarity"]}
        support = np.array([0, 0, -mesh.bounds[0, 2]])
        own_cells, cnt = np.unique(gc[own & (gp[..., 2] > 0.006) & inside], return_counts=True)
        record = dict(
            square=name,
            mask_pixels=int(own.sum()),
            native_depth_samples=int(len(index)),
            mask_points_in_assigned_square_fraction=float(np.mean((gc[own] == square) & inside[own])),
            foreground_square_counts={chess.square_name(int(c)): int(n) for c, n in zip(own_cells, cnt)},
            native_observed_z_mm=(np.quantile(native[:, 2], [0, 0.1, 0.5, 0.9, 1]) * 1000).tolist(),
            mesh_bounds_mm=(mesh.bounds * 1000).tolist(),
            mesh_extents_mm=(mesh.extents * 1000).tolist(),
            confidence2_fraction=float(np.mean(gpu["confidence"][0][own] == 2)),
            metrics=metrics,
            original_10k_fused=original,
            exact_subset=stats(exact),
            sampling_error_120k=stats(approx - exact),
            per_frame=per_frame,
            experiments=experiments,
            support_snap_fused=stats(tree.query(fuse - support)[0]),
            selected_pose_trial=next(t for t in pose["trials"] if t[0] == pose["seed"]),
            elapsed_s=time.perf_counter() - start,
        )
        results.append(record)
        ax = axes.flat[i]
        ax.scatter(fuse[:, 0] * 1000, fuse[:, 2] * 1000, s=3, c="#cf6b37", alpha=0.45, label="Fused depth")
        ax.scatter(
            probes[::50, 0] * 1000,
            probes[::50, 2] * 1000,
            s=1,
            c="#17669a",
            alpha=0.5,
            label="Generated mesh",
        )
        ax.axhline(0, c="black", lw=0.7)
        ax.set_title(f"{name}: median {metrics['fused_square']['median_mm']:.1f} mm")
        ax.set_xlabel("Board X (mm)")
        ax.set_ylabel("Height (mm)")
        ax.set_aspect("equal", adjustable="datalim")
        # Project mesh surface to image, retain sparse points for a legible overlay.
        cam = trimesh.transform_points(probes[::100], np.linalg.inv(board_from_camera))
        uv = cam[:, :2] / cam[:, 2, None] * np.diag(gpu["K"])[:2] + gpu["K"][:2, 2]
        for x, y in np.rint(uv).astype(int):
            if 0 <= x < 960 and 0 <= y < 720:
                cv2.circle(overlay, (x, y), 1, (255, 80, 50), -1)
        middle = np.median(uv, axis=0).astype(int)
        cv2.putText(overlay, name, tuple(middle), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        print(
            name,
            round(original["median_mm"], 2),
            "own",
            round(metrics["own_mask"]["median_mm"], 2),
            "native",
            len(index),
            "held",
            experiments["rigid"].get("heldout_after", {}),
            flush=True,
        )
        write_json(
            a.out / "profile.json",
            dict(contracts=known, train_frames=train_frames, heldout_frames=test_frames, objects=results),
        )
    for ax in axes.flat[len(results) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Original SAM 3D meshes vs ten-view LiDAR: side projections, no fit applied", fontsize=16)
    fig.tight_layout()
    fig.savefig(a.out / "mesh_depth_profiles.png", dpi=130)
    plt.close(fig)
    Image.fromarray(overlay).save(a.out / "reference_overlay.jpg")
    write_json(
        a.out / "profile.json",
        dict(
            contracts=known,
            train_frames=train_frames,
            heldout_frames=test_frames,
            runtime_s=time.perf_counter() - started,
            objects=results,
            caveats=[
                "Square clouds are not instance-labelled in other views.",
                "Held-out views share board calibration; not independent ground truth.",
                "One-sided pose/scale fitting is diagnostic and can reward inflated shapes.",
                "No candidate mesh or simulation was replaced.",
            ],
        ),
    )


if __name__ == "__main__":
    main()
