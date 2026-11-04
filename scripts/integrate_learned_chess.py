"""Validate learned assets against a compatible multi-view LiDAR reconstruction."""

import argparse
import json
from pathlib import Path

import numpy as np

from lidar_video.capture import load_capture, write_json
from lidar_video.reconstruction import reconstruct


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("reconstruction", type=Path)
    parser.add_argument("learned", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-simulation", action="store_true")
    args = parser.parse_args()
    local = load_capture(args.reconstruction / "refined_capture.npz")
    gpu = load_capture(args.learned / "capture.npz")
    calibration = json.loads((args.reconstruction / "board_calibration.json").read_text())
    gpu_calibration = json.loads((args.learned / "calibration.json").read_text())
    for key in ("board_from_world", "square_size_m", "center_gap_m"):
        if not np.allclose(calibration.get(key, 0), gpu_calibration.get(key, 0), atol=1e-8):
            raise ValueError(f"Incompatible calibration: {key}")
    source = json.loads((args.learned / "meshes.json").read_text())
    if source["schema"] != "roboskills-chess-meshes-v1":
        raise ValueError("Unknown mesh manifest schema")
    mapped = dict(schema=source["schema"], objects={})
    for name, entry in source["objects"].items():
        frame = entry["frame"]
        matches = np.flatnonzero(local["source_frame"] == gpu["source_frame"][frame])
        if len(matches) != 1:
            raise ValueError("Mesh source frame is not uniquely present in local capture")
        target = int(matches[0])
        if not np.allclose(local["world_from_camera"][target], gpu["world_from_camera"][frame], atol=1e-8):
            raise ValueError("Mesh and fused capture use different camera poses")
        mapped["objects"][name] = dict(
            mesh=str((args.learned / entry["mesh"]).resolve()),
            pose=str((args.learned / entry["pose"]).resolve()),
            frame=target,
        )
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "mapped_mesh_manifest.json", mapped)
    report = reconstruct(
        args.reconstruction / "refined_capture.npz",
        args.reconstruction / "board_calibration.json",
        args.out / "result",
        static_scene=True,
        fusion_frames=json.loads((args.reconstruction / "fusion_frames.json").read_text()),
        min_height=0.015,
        reference_frame=calibration["reference_frame"],
        mesh_manifest=args.out / "mapped_mesh_manifest.json",
        mesh_policy="fallback",
        simulation=not args.no_simulation,
    )
    report.update(
        learned_inference_run=True,
        piece_identity_verified=False,
        completeness="Partial geometry; absent pieces are not verified empty squares",
        segmentation_audit=json.loads((args.learned / "segmentation_audit.json").read_text()),
        inference_status=json.loads((args.learned / "status.json").read_text()),
    )
    write_json(args.out / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
