"""Native-sample mask-interior and empty-board controls for the saved chess run."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from lidar_video.capture import load_capture, write_json
from lidar_video.reconstruction import calibrated_points


def height_stats(p):
    z = p[:, 2] * 1000
    return (
        dict(
            n=len(z),
            median_height_mm=float(np.median(z)),
            p10_height_mm=float(np.quantile(z, 0.1)),
            p90_height_mm=float(np.quantile(z, 0.9)),
            fraction_at_or_below_15mm=float(np.mean(z <= 15)),
            fraction_below_minus_15mm=float(np.mean(z < -15)),
        )
        if len(z)
        else None
    )


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("recording", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--source", type=Path, help="Optional original Stray Scanner folder for byte-value checks"
    )
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    base = args.recording
    g = base / "gpu-run/learned-partial"
    local = load_capture(base / "reconstruction/refined_capture.npz")
    gpu = load_capture(g / "capture.npz")
    if local["depth"].shape[1:] != (192, 256) or gpu["depth"].shape[1:] != (720, 960):
        raise ValueError("These diagnostics expect the saved 256x192 native / 960x720 model capture")
    cal = json.loads((base / "reconstruction/board_calibration.json").read_text())
    frames = json.loads((base / "reconstruction/fusion_frames.json").read_text())
    masks = np.array(Image.open(g / "masks/00000.png"))
    objects = json.loads((g / "masks/objects.json").read_text())
    ref = int(np.flatnonzero(local["source_frame"] == gpu["source_frame"][0])[0])
    lp = calibrated_points(local, cal, ref)
    v, u = np.indices(masks.shape, dtype=np.float32)
    ny = np.clip(np.rint(v / 3.75).astype(int), 0, 191)
    nx = np.clip(np.rint(u / 3.75).astype(int), 0, 255)
    valid = (gpu["depth"][0] > 0) & (gpu["confidence"][0] >= 1)
    rows = []
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    for i, obj in enumerate(sorted(objects, key=lambda x: x["square"])):
        mask = masks == obj["label"]
        core = cv2.erode(mask.astype("uint8"), np.ones((9, 9), np.uint8)).astype(bool)

        def points(selection):
            ids = np.unique(ny[selection & valid] * 256 + nx[selection & valid])
            return lp.reshape(-1, 3)[ids], ids

        allp, ids = points(mask)
        corep, _ = points(core)
        deep_core = cv2.erode(mask.astype("uint8"), np.ones((17, 17), np.uint8)).astype(bool)
        deep_points, _ = points(deep_core)
        conf = local["confidence"][ref].ravel()[ids]
        # Probe a small global RGB/depth offset without changing any stored data.
        core_ids = np.unique(ny[core & valid] * 256 + nx[core & valid])
        offset_medians = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ys = core_ids // 256 + dy
                xs = core_ids % 256 + dx
                ok = (ys >= 0) & (ys < 192) & (xs >= 0) & (xs < 256)
                ys, xs = ys[ok], xs[ok]
                ok = (local["depth"][ref, ys, xs] > 0) & (local["confidence"][ref, ys, xs] >= 1)
                heights = lp[ys[ok], xs[ok], 2] * 1000
                if len(heights):
                    offset_medians.append(
                        dict(
                            dx_native_px=dx,
                            dy_native_px=dy,
                            n=len(heights),
                            median_height_mm=float(np.median(heights)),
                        )
                    )
        row = dict(
            square=obj["square"],
            mask_pixels=int(mask.sum()),
            valid_depth_fraction=float(valid[mask].mean()),
            native_mask=height_stats(allp),
            native_core=height_stats(corep),
            native_deep_core=height_stats(deep_points),
            native_confidence_counts={str(k): int((conf == k).sum()) for k in [0, 1, 2]},
            core_erosion_rgb_pixels=4,
            core_erosion_native_pixels=4 / 3.75,
            offset_sensitivity=offset_medians,
        )
        rows.append(row)
        yy, xx = np.where(mask)
        x0 = max(0, xx.min() - 15)
        x1 = min(960, xx.max() + 16)
        y0 = max(0, yy.min() - 15)
        y1 = min(720, yy.max() + 16)
        ax = axes.flat[i]
        ax.imshow(gpu["rgb"][0])
        ax.contour(core, levels=[0.5], colors=["cyan"], linewidths=0.7)
        z = lp.reshape(-1, 3)[ids, 2] * 1000
        scatter = ax.scatter(
            (ids % 256) * 3.75,
            (ids // 256) * 3.75,
            c=z,
            s=11,
            cmap="viridis",
            vmin=0,
            vmax=100,
            edgecolors="none",
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_xticks([])
        ax.set_yticks([])
        coremed = "none" if not row["native_core"] else f"{row['native_core']['median_height_mm']:.1f}"
        ax.set_title(f"{obj['square']}: core median {coremed} mm\n{len(ids)} native samples", fontsize=10)
    for ax in axes.flat[len(rows) :]:
        ax.axis("off")
    fig.subplots_adjust(top=0.92, bottom=0.1, hspace=0.35)
    cb = fig.colorbar(scatter, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.025, pad=0.035)
    cb.set_label("Native LiDAR height above calibrated board (mm); colors clipped to 0–100 mm")
    fig.suptitle(
        "Reference frame 5217: measured depth within visible pieces\nCyan contour = mask eroded 4 RGB pixels; each dot = one native depth sample",
        fontsize=14,
    )
    fig.savefig(args.out / "native_depth_controls.png", dpi=130)
    plt.close(fig)
    source_checks = []
    if args.source:
        for f in frames:
            source_id = int(local["source_frame"][f])
            raw = cv2.imread(str(args.source / "depth" / f"{source_id:06d}.png"), cv2.IMREAD_UNCHANGED)
            conf = cv2.imread(str(args.source / "confidence" / f"{source_id:06d}.png"), cv2.IMREAD_UNCHANGED)
            if raw is None or conf is None:
                raise ValueError(f"Missing original sensor files for {source_id}")
            error = float(np.max(abs(raw.astype(np.float32) / 1000 - local["depth"][f])))
            equal = bool(np.array_equal(conf, local["confidence"][f]))
            if error != 0 or not equal:
                raise ValueError(f"Source conversion mismatch in frame {source_id}")
            source_checks.append(
                dict(source_frame=source_id, max_depth_difference_m=error, confidence_equal=equal)
            )
    controls = []
    for f in frames:
        p = calibrated_points(local, cal, f)
        s = cal["square_size_m"]
        gap = cal.get("center_gap_m", 0)
        # Central empty board band, excluding seam; no height-based filtering.
        select = (abs(p[..., 0]) < 3 * s) & (abs(p[..., 1]) > gap / 2 + 0.015) & (abs(p[..., 1]) < 1.5 * s)
        select &= (local["depth"][f] > 0) & (local["confidence"][f] >= 1)
        z = p[select, 2] * 1000
        controls.append(
            dict(
                frame=f,
                source_frame=int(local["source_frame"][f]),
                n=len(z),
                median_signed_height_mm=float(np.median(z)),
                median_absolute_height_mm=float(np.median(abs(z))),
                p90_absolute_height_mm=float(np.quantile(abs(z), 0.9)),
                fraction_over_15mm=float(np.mean(abs(z) > 15)),
            )
        )
    write_json(
        args.out / "depth_controls.json",
        dict(
            objects=rows,
            empty_board_controls=controls,
            original_sensor_checks=source_checks,
            selection="Native depth sample identities deduplicated within SAM masks; no foreground-height cut.",
            limitations="Empty band assumed unoccupied; masks and board calibration are estimates. Boundary sensor footprints may mix surfaces.",
        ),
    )
    print(json.dumps(dict(objects=rows, empty_board_controls=controls), indent=2))


if __name__ == "__main__":
    main()
