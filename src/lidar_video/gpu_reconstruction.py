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
        label = len(occupied) + 1
        occupied[name] = label
        labels[mask & (labels == 0)] = label
        detected.append(
            dict(
                square=name,
                raw_index=index,
                label=label,
                score=float(detection["score"]),
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
    sam = Sam()
    try:
        detections = sam.detect(data["rgb"][reference_frame], ["chess piece"])
    finally:
        sam.close()
    raw = detections.get("chess piece", [])
    np.savez_compressed(out / "sam3_raw_masks.npz", **{str(i): d["mask"] for i, d in enumerate(raw)})
    write_json(out / "sam3_raw_detections.json", [{k: v for k, v in d.items() if k != "mask"} for d in raw])
    write_json(
        out / "status.json",
        dict(
            gpu=torch.cuda.get_device_name(0),
            stages={"sam3": "passed"},
            raw_instances=len(raw),
            reference_frame=reference_frame,
        ),
    )
    labels, detected, audit = assign_instances(data, cal, reference_frame, raw, ambiguous_policy)
    write_json(out / "segmentation_audit.json", audit)
    if not detected:
        raise ValueError("SAM 3 has no unambiguous pieces with sufficient registered depth")
    Image.fromarray(labels).save(masks_dir / f"{reference_frame:05d}.png")
    write_json(masks_dir / "objects.json", detected)
    occupied = {item["square"]: item["label"] for item in detected}
    manifest = dict(schema="roboskills-chess-meshes-v1", objects={})
    status = dict(gpu=torch.cuda.get_device_name(0), stages={"sam3": "passed"})

    def infer_one(name, label):
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
                    str(upstream / "depth/depth.npz"),
                    "--out",
                    str(shape),
                    "--label",
                    str(label),
                    "--frame",
                    str(reference_frame),
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
        futures = {pool.submit(infer_one, name, label): name for name, label in occupied.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as error:
                status["failures"][name] = str(error)
                status["stages"][f"sam3d-{name}"] = "failed"
            else:
                manifest["objects"][name] = dict(
                    mesh=f"shapes/{name}/mesh.ply", pose=f"shapes/{name}/pose.json", frame=reference_frame
                )
                status["stages"][f"sam3d-{name}"] = "passed"
            write_json(out / "status.json", status)
            write_json(out / "meshes.json", manifest)
    if status["failures"] and mesh_policy == "strict":
        raise RuntimeError("SAM 3D failures; inspect status.json")
    # One selected frame: masks and learned geometry describe one observed placement.
    return reconstruct(
        out / "capture.npz",
        out / "calibration.json",
        out / "result",
        masks=masks_dir,
        mesh_manifest=out / "meshes.json",
        mesh_policy=mesh_policy,
        samples=samples,
        reference_frame=reference_frame,
    )
