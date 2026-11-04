"""Process an imported static chess scan with reviewable board annotations."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lidar_video.annotated_board import align_views, calibrate_annotations, metric_grid
from lidar_video.board_calibration import project
from lidar_video.capture import load_capture, write_json
from lidar_video.geometry import export_ply, voxel_points
from lidar_video.reconstruction import calibrated_points, reconstruct


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("imported", type=Path)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-height", type=float, default=0.015)
    parser.add_argument("--no-simulation", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data = load_capture(args.imported / "capture.npz")
    annotation = json.loads(args.annotations.read_text())
    reference = annotation["frame"]
    image = cv2.imread(str(args.imported / "rgb" / f"{int(data['source_frame'][reference]):06d}.jpg"))
    cal = calibrate_annotations(data, image, annotation)
    write_json(args.out / "annotations.json", annotation)
    write_json(args.out / "board_calibration.json", cal)
    accepted, audit = align_views(data, args.imported / "rgb", cal)
    write_json(
        args.out / "alignment_audit.json",
        dict(
            attempts=audit,
            accepted=len(accepted),
            stationary_assumption="board and pieces visually stationary across selected views; not motion-certified",
        ),
    )
    frames = sorted(set([reference, *[item["frame"] for item in accepted]]))
    data["original_world_from_camera"] = data["world_from_camera"].copy()
    for item in accepted:
        data["world_from_camera"][item["frame"]] = item["world_from_camera"]
    np.savez_compressed(args.out / "refined_capture.npz", **data)
    write_json(args.out / "fusion_frames.json", frames)
    board_pose = np.linalg.inv(cal["board_from_world"])
    size, gap = cal["square_size_m"], cal["center_gap_m"]
    outline = metric_grid([[-4, -4, 0], [4, -4, 0], [4, 4, 0], [-4, 4, 0]], size, gap)
    cloud, colors, previews = [], [], []
    for frame in frames:
        native = cv2.imread(str(args.imported / "rgb" / f"{int(data['source_frame'][frame]):06d}.jpg"))
        uv = project(
            outline @ board_pose[:3, :3].T + board_pose[:3, 3],
            data["world_from_camera"][frame],
            data["K"][frame],
        )
        uv *= np.array(native.shape[:2][::-1]) / np.array(data["depth"].shape[1:][::-1])
        cv2.polylines(native, [np.rint(uv).astype(int)], True, (0, 255, 0), 4)
        cv2.putText(
            native, f"source frame {int(data['source_frame'][frame])}", (30, 60), 0, 1.5, (0, 255, 0), 3
        )
        if frame == reference:
            for pixel in (
                np.asarray(cal["corners"])
                * np.array(native.shape[:2][::-1])
                / np.array(data["depth"].shape[1:][::-1])
            ):
                cv2.circle(native, tuple(np.rint(pixel).astype(int)), 7, (0, 0, 255), -1)
            cv2.imwrite(str(args.out / "calibration_overlay.jpg"), native)
        previews.append(cv2.resize(native, (480, 360)))
        xyz = calibrated_points(data, cal, frame)
        keep = (data["confidence"][frame] >= 1) & (data["depth"][frame] > 0)
        keep &= (abs(xyz[..., 0]) < 4 * size) & (abs(xyz[..., 1]) < 4 * size + gap / 2)
        keep &= (xyz[..., 2] > -0.02) & (xyz[..., 2] < 0.2)
        cloud.extend(xyz[keep])
        colors.extend(data["rgb"][frame][keep] / 255)
    while len(previews) % 4:
        previews.append(np.zeros((360, 480, 3), np.uint8))
    cv2.imwrite(
        str(args.out / "aligned_views.jpg"),
        np.concatenate([np.concatenate(previews[i : i + 4], 1) for i in range(0, len(previews), 4)], 0),
    )
    xyz, rgb = voxel_points(np.asarray(cloud), np.asarray(colors), voxel=0.002)
    export_ply(args.out / "observed_board.ply", xyz, rgb)
    result = reconstruct(
        args.out / "refined_capture.npz",
        args.out / "board_calibration.json",
        args.out / "result",
        static_scene=True,
        fusion_frames=frames,
        min_height=args.min_height,
        reference_frame=reference,
        simulation=not args.no_simulation,
    )
    result.update(
        source="real iPhone recording",
        sampled_frames=int(len(data["rgb"])),
        source_frames=json.loads((args.imported / "import_report.json").read_text())["source_frames"],
        fused_frames=len(frames),
        calibration="annotation-assisted, folding seam fitted",
        learned_inference_run=False,
        piece_identity_verified=False,
        completeness="partial observed geometry; do not interpret absent nodes as empty squares",
    )
    write_json(args.out / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
