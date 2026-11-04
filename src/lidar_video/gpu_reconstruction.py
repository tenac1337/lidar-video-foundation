"""A100 chess capture -> upstream SAM 3 instances -> SAM 3D -> shared outputs.

Runs in Colab after scripts/colab_sam3d.py setup/weights. Each occupied square
gets its own SAM 3D call. Detection does not infer chess piece identity.
"""

import json
import os
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chess
import numpy as np
from PIL import Image

from lidar_video.capture import load_capture, normalize, write_json
from lidar_video.object_evidence import assess_depth, select_views
from lidar_video.pipeline import prepare_upstream
from lidar_video.reconstruction import calibrated_points, reconstruct


def assign_instances(data, cal, reference_frame, raw, ambiguous_policy="error"):
    """Reject ambiguous associations by default; explicit skip excludes every conflicting instance."""
    if ambiguous_policy not in ("error", "skip"):
        raise ValueError("ambiguous_policy must be error or skip")
    points = calibrated_points(data, cal, reference_frame)
    valid = (
        (data["depth"][reference_frame] > 0)
        & (data["confidence"][reference_frame] >= 1)
        & (points[..., 2] > 0.006)
    )
    labels = np.zeros(data["depth"].shape[1:], dtype=np.uint16)
    detected = []
    occupied = {}
    candidates, excluded = [], []
    for index, detection in enumerate(raw):
        mask = np.asarray(detection["mask"], dtype=bool)
        if mask.shape != labels.shape:
            raise ValueError("SAM 3 returned an unregistered mask")
        seen = points[mask & valid]
        if len(seen) < 24:
            excluded.append(dict(index=index, reason="insufficient registered depth"))
            continue
        center = np.median(seen[:, :2], axis=0)
        gap = cal.get("center_gap_m", 0)
        if abs(center[1]) < gap / 2:
            excluded.append(dict(index=index, reason="center lies in folding seam"))
            continue
        center[1] -= np.sign(center[1]) * gap / 2
        cell = np.floor(center / cal["square_size_m"] + 4).astype(int)
        if np.any((cell < 0) | (cell > 7)):
            excluded.append(dict(index=index, reason="center lies outside board"))
            continue
        name = chess.square_name(int(cell[0] + 8 * cell[1]))
        candidates.append((name, index, mask, detection))
    conflicts = {name for name, *_ in candidates if sum(c[0] == name for c in candidates) > 1}
    if conflicts and ambiguous_policy == "error":
        raise ValueError(f"Multiple SAM 3 instances map to {sorted(conflicts)}; inspect segmentation")
    for name, index, mask, detection in candidates:
        if name in conflicts:
            excluded.append(dict(index=index, square=name, reason="ambiguous square assignment"))
            continue
        quality, _, _ = assess_depth(data, cal, reference_frame, mask, points=points, square=name)
        if not quality["passed"]:
            excluded.append(dict(index=index, square=name, reason="unreliable object depth", quality=quality))
            continue
        label = len(occupied) + 1
        occupied[name] = label
        labels[mask & (labels == 0)] = label
        detected.append(
            dict(
                square=name,
                raw_index=index,
                label=label,
                score=float(detection["score"]),
                depth_quality=quality,
                frame=reference_frame,
                source="upstream SAM 3 instance detector",
            )
        )
    return (
        labels,
        detected,
        dict(
            policy=ambiguous_policy,
            excluded=excluded,
            ambiguous_squares=sorted(conflicts),
            raw_instances=len(raw),
            accepted_instances=len(detected),
        ),
    )


def plan_instances(data, calibration, detections_by_frame, ambiguous_policy="error"):
    """CPU-testable planning stage; no GPU or model credentials are needed."""
    candidates, masks, audits = [], {}, {}
    for frame, raw in sorted(detections_by_frame.items()):
        labels, detected, audit = assign_instances(data, calibration, frame, raw, ambiguous_policy)
        masks[frame], audits[str(frame)] = labels, audit
        for obj in detected:
            mask = labels == obj["label"]
            if not mask.any():
                labels[mask] = 0
                audit["accepted_instances"] -= 1
                audit["excluded"].append(
                    dict(
                        index=obj["raw_index"],
                        square=obj["square"],
                        reason="mask empty after overlap removal",
                    )
                )
                continue
            quality, trusted, cloud = assess_depth(data, calibration, frame, mask, square=obj["square"])
            if not quality["passed"]:
                # Resolve pixel overlap before the final quality decision.
                audit["excluded"].append(
                    dict(
                        index=obj["raw_index"],
                        square=obj["square"],
                        reason="unreliable depth after overlap removal",
                        quality=quality,
                    )
                )
                labels[mask] = 0
                audit["accepted_instances"] -= 1
                continue
            candidates.append(
                dict(
                    square=obj["square"],
                    frame=frame,
                    label=obj["label"],
                    mask=mask,
                    trusted=trusted,
                    cloud=cloud,
                    quality=quality,
                )
            )
    selected, rejected = select_views(candidates, data, calibration)
    return (
        masks,
        selected,
        dict(
            frames=audits,
            rejected_associations=rejected,
            selected_reference_frames={name: item["reference"]["frame"] for name, item in selected.items()},
        ),
    )


def run_gpu(
    capture,
    calibration,
    out,
    native_script,
    samples=20,
    pose_samples=8,
    mesh_policy="strict",
    reference_frame=0,
    ambiguous_policy="error",
    workers=1,
    candidate_frames=None,
    static_scene=False,
):
    import torch
    from roboskills.tools.sam3.sam3 import Sam

    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise ValueError("This run requires the requested NVIDIA A100")
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("Load HF_TOKEN from Colab Secrets first")
    if not Path("/opt/venv/bin/python").exists():
        raise ValueError("Run the SAM 3D native environment setup and weights stages first")
    if workers not in (1, 2):
        raise ValueError("workers must be 1 or 2")
    if workers == 2 and torch.cuda.get_device_properties(0).total_memory < 70 * 2**30:
        raise ValueError("Two workers require an A100 with at least 70 GiB memory")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    data = normalize(load_capture(capture))
    if type(reference_frame) is not int or not 0 <= reference_frame < len(data["depth"]):
        raise ValueError("Reference frame outside capture")
    frames = list(candidate_frames) if candidate_frames is not None else [reference_frame]
    if not static_scene or len(frames) < 2 or len(set(frames)) != len(frames):
        raise ValueError("Learned reconstruction requires --static-scene and at least two candidate frames")
    if any(type(f) is not int or not 0 <= f < len(data["depth"]) for f in frames):
        raise ValueError("Candidate frame outside capture")
    if calibration is None:
        from lidar_video.board_calibration import calibrate

        cal = calibrate(data, out / "board_detection")
    else:
        cal = json.loads(Path(calibration).read_text())
    np.savez_compressed(out / "capture.npz", **data)
    write_json(out / "calibration.json", cal)
    upstream = prepare_upstream(data, out / "upstream", reference_frame=reference_frame)
    masks_dir = out / "masks"
    masks_dir.mkdir()
    raw_by_frame = {}
    sam = Sam()
    try:
        for frame in frames:
            raw = sam.detect(data["rgb"][frame], ["chess piece"]).get("chess piece", [])
            raw_by_frame[frame] = raw
            folder = out / "segmentation" / f"{frame:05d}"
            folder.mkdir(parents=True)
            np.savez_compressed(folder / "raw_masks.npz", **{str(i): d["mask"] for i, d in enumerate(raw)})
            write_json(folder / "detections.json", [{k: v for k, v in d.items() if k != "mask"} for d in raw])
    finally:
        sam.close()
    masks, selected, audit = plan_instances(data, cal, raw_by_frame, ambiguous_policy)
    write_json(out / "segmentation_audit.json", audit)
    write_json(
        out / "status.json",
        dict(
            gpu=torch.cuda.get_device_name(0),
            stages={"sam3": "passed", "instance_evidence": "passed" if selected else "abstained"},
            status="ready_for_inference" if selected else "abstained",
            candidate_frames=frames,
            reliable_objects=len(selected),
        ),
    )
    if not selected:
        raise ValueError(
            "No objects have reliable native depth and corroborating segmented views; inspect segmentation_audit.json"
        )
    trusted_masks_dir = out / "trusted_masks"
    trusted_masks_dir.mkdir()
    assignments = {}
    for frame in frames:
        assignments[str(frame)] = {}
        Image.fromarray(masks[frame]).save(masks_dir / f"{frame:05d}.png")
        trusted_labels = np.zeros_like(masks[frame])
        for name, item in selected.items():
            for observation in item["observations"]:
                if observation["frame"] == frame:
                    trusted_labels[observation["trusted"]] = observation["label"]
                    assignments[str(frame)][str(observation["label"])] = name
        Image.fromarray(trusted_labels).save(trusted_masks_dir / f"{frame:05d}.png")
    write_json(
        trusted_masks_dir / "instances.json", dict(schema="chess-instance-assignments-v1", frames=assignments)
    )
    write_json(
        masks_dir / "objects.json",
        [
            dict(
                square=name,
                frame=item["reference"]["frame"],
                label=item["reference"]["label"],
                depth_quality=item["reference"]["quality"],
            )
            for name, item in selected.items()
        ],
    )
    manifest = dict(schema="roboskills-chess-meshes-v1", objects={})
    status = dict(
        gpu=torch.cuda.get_device_name(0),
        stages={"sam3": "passed"},
        candidate_frames=frames,
        reliable_objects=len(selected),
    )

    def infer_one(name, item):
        observation = item["reference"]
        frame, label = observation["frame"], observation["label"]
        # The full mask still describes the silhouette; only trustworthy native-depth
        # evidence may condition its geometry. No depth filling or metric rescaling.
        with np.load(upstream / "depth/depth.npz", allow_pickle=False) as archive:
            depth_input = {key: archive[key] for key in archive.files}
        depth_input["valid"][frame] &= observation["trusted"]
        depth_input["depth"][frame] = np.where(depth_input["valid"][frame], depth_input["depth"][frame], 0)
        depth_file = out / f"depth-{name}.npz"
        np.savez_compressed(depth_file, **depth_input)
        shape = out / "shapes" / name
        log = out / f"sam3d-{name}.log"
        with log.open("w") as handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(native_script),
                    "infer",
                    "--video",
                    str(upstream / "input/video.mp4"),
                    "--masks",
                    str(masks_dir),
                    "--depth",
                    str(depth_file),
                    "--out",
                    str(shape),
                    "--label",
                    str(label),
                    "--frame",
                    str(frame),
                    "--samples",
                    str(pose_samples),
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=1800)
            except BaseException:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
            if returncode:
                raise RuntimeError(f"SAM 3D failed for {name}; inspect {log}")
        return name

    status.update(segmentation=audit, workers=workers, failures={})
    write_json(out / "status.json", status)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(infer_one, name, item): name for name, item in selected.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as error:
                status["failures"][name] = str(error)
                status["stages"][f"sam3d-{name}"] = "failed"
            else:
                manifest["objects"][name] = dict(
                    mesh=f"shapes/{name}/mesh.ply",
                    pose=f"shapes/{name}/pose.json",
                    frame=selected[name]["reference"]["frame"],
                    validation="instance-multiview-v1",
                    static_scene=True,
                    observations=[
                        dict(frame=o["frame"], mask=f"masks/{o['frame']:05d}.png", label=o["label"])
                        for o in selected[name]["observations"]
                    ],
                )
                status["stages"][f"sam3d-{name}"] = "passed"
            write_json(out / "status.json", status)
            write_json(out / "meshes.json", manifest)
    if status["failures"] and mesh_policy == "strict":
        raise RuntimeError("SAM 3D failures; inspect status.json")
    # Only corroborated instances supply geometry; missing nodes are not verified empty cells.
    return reconstruct(
        out / "capture.npz",
        out / "calibration.json",
        out / "result",
        masks=trusted_masks_dir,
        static_scene=True,
        fusion_frames=frames,
        min_height=0.015,
        mesh_manifest=out / "meshes.json",
        mesh_policy=mesh_policy,
        samples=samples,
        reference_frame=reference_frame,
    )
