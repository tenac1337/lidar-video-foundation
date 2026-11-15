"""Development-only SAM3/SAM3D adapter for generic static tabletop objects.

Requires the existing provisioned Colab native SAM3D worker. No chess calibration,
per-object identity list, pose fitting, box substitution, or manual mesh completion.
This adapter is not a frozen evaluation recipe: model weights/runtime must still
be pinned and the complete learned path validated on development recordings.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from lidar_video.capture import load_capture, normalize, validate, write_json
from lidar_video.general_run import digest, execute, freeze
from lidar_video.general_scene import InstancePolicy, associate_instances
from lidar_video.model_depth import object_conditioning_depth
from lidar_video.native_depth import native_intrinsics, native_shape, sample_ids
from lidar_video.reconstruction import rigid
from lidar_video.synthetic import encode_video


def model_transport(data, out):
    """Index-only image transport for single-frame inference, not sensor timing.

    The worker selects one image by integer index. Original timestamps/poses/source
    IDs are preserved separately; the 1 FPS container must never drive temporal
    estimation. This avoids relabeling irregular sensor times as uniform capture.
    """
    validate(data)
    if np.asarray(data["K"]).shape != (3, 3):
        raise ValueError("Normalize camera intrinsics before model transport")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    valid = np.isfinite(data["depth"]) & (data["depth"] > 0) & (data["confidence"] >= 1)
    np.savez_compressed(
        out / "depth.npz",
        depth=np.where(valid, data["depth"], 0).astype(np.float32),
        valid=valid,
        K=data["K"],
        frames=np.arange(len(valid)),
        native_K=native_intrinsics(data),
        native_depth_shape=np.array(native_shape(data)),
        depth_sample_id=np.stack([sample_ids(data, f) for f in range(len(valid))]),
    )
    np.savez_compressed(
        out / "sensor_metadata.npz",
        **{k: data[k] for k in ["source_frame", "world_from_camera", "rgb_time", "depth_time"]},
    )
    encode_video(data["rgb"], out / "video.mp4", fps=1)
    write_json(
        out / "transport.json",
        dict(
            schema="single-frame-model-transport-v1",
            timing="Index-only 1 FPS container; original sensor times in sensor_metadata.npz",
            temporal_inference_allowed=False,
            rgb_encoding="Existing H264 encoder; masks use original RGB",
            source_frames=[int(s) for s in data["source_frame"]],
        ),
    )
    return out


def world_prior(shape, data, frame, decompose):
    """Apply the upstream rigid pose once; mesh.ply already includes model scale."""
    shape = Path(shape)
    pose = json.loads((shape / "pose.json").read_text())
    transform = rigid(data["world_from_camera"][frame], "world_from_camera") @ rigid(
        pose["camera_from_object"], "camera_from_object"
    )
    mesh = trimesh.load(shape / "mesh.ply", force="mesh", process=True)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("Invalid model mesh")
    mesh.apply_transform(transform)
    world = shape / "world"
    world.mkdir(exist_ok=False)
    mesh.export(world / "visual.ply")
    decompose(world / "visual.ply", world, 0.05, 20000, 32)
    parts = json.loads((world / "parts.json").read_text())["parts"]
    if not parts:
        raise ValueError("No convex collision parts")
    paths = []
    for index, relative in enumerate(parts):
        path = (world / relative).resolve()
        if not path.is_relative_to(world.resolve()) or not path.is_file():
            raise ValueError("Collision output escapes its directory")
        part = trimesh.load(path, force="mesh", process=True)
        name = f"collision_{index:03d}.ply"
        part.export(world / name)
        paths.append(name)
    return dict(
        visual="visual.ply",
        collision=paths,
        provenance=dict(
            source="SAM3D native worker metric mesh",
            source_frame=int(data["source_frame"][frame]),
            mesh_sha256=digest(shape / "mesh.ply"),
            pose_sha256=digest(shape / "pose.json"),
            transform=transform.tolist(),
            extra_scale_applied=False,
            collision=dict(method="CoACD", threshold=0.05, max_faces=20000, max_hulls=32, seed=0),
        ),
    )


def run_development(
    capture,
    out,
    native_script,
    *,
    frames,
    prompt="tabletop object",
    pose_samples=8,
    samples=100,
    jitter=0.03,
    seed=7,
):
    """One sequential development attempt; preserve errors and all raw detections."""
    from roboskills.tools.coacd.coacd import decompose
    from roboskills.tools.sam3.sam3 import Sam

    if not prompt.strip() or type(pose_samples) is not int or not 1 <= pose_samples <= 32:
        raise ValueError("Nonempty prompt and 1..32 pose samples required")
    if type(samples) is not int or not 1 <= samples <= 1000:
        raise ValueError("samples must be 1..1000")
    if isinstance(jitter, bool) or not np.isfinite(jitter) or jitter < 0:
        raise ValueError("jitter must be finite and nonnegative")
    InstancePolicy(seed=seed)
    native_script = Path(native_script).resolve()
    worker_adapter = native_script.with_name("sam3d_native_rays.py")
    if (
        not native_script.is_file()
        or not worker_adapter.is_file()
        or not Path("/opt/venv/bin/python").is_file()
    ):
        raise ValueError("Provision the native Colab SAM3D runtime and its ray adapter first")
    data = normalize(load_capture(capture))
    if (
        len(frames) < 2
        or len(set(frames)) != len(frames)
        or any(type(f) is not int or not 0 <= f < len(data["depth"]) for f in frames)
    ):
        raise ValueError("Provide at least two distinct construction frame indices")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = dict(
        status="started",
        evaluation_scope="development",
        frame_indices=frames,
        prompt=prompt,
        detector_threshold=0.3,
        native_script_sha256=digest(native_script),
        ray_adapter_sha256=digest(worker_adapter),
        capture_sha256=digest(capture),
        pose_samples=pose_samples,
        frozen_evaluation_ready=False,
    )
    write_json(out / "status.json", status)
    started = time.monotonic()
    try:
        np.savez_compressed(out / "capture.npz", **data)
        transport = model_transport(data, out / "transport")
        raw = {}
        sam = Sam()
        try:
            for frame in frames:
                raw[frame] = sam.detect(data["rgb"][frame], [prompt], threshold=0.3).get(prompt, [])
                np.savez_compressed(
                    out / f"raw-masks-{frame}.npz", **{str(i): d["mask"] for i, d in enumerate(raw[frame])}
                )
                write_json(
                    out / f"raw-detections-{frame}.json",
                    [{k: v for k, v in d.items() if k != "mask"} for d in raw[frame]],
                )
        finally:
            sam.close()
        selected, audit = associate_instances(data, raw, static_scene=True, policy=InstancePolicy(seed=seed))
        write_json(out / "association.json", audit)
        if not selected:
            raise ValueError("No corroborated object detections")
        masks = out / "masks"
        masks.mkdir()
        detection_frames = {}
        for frame in frames:
            labels = np.zeros(data["depth"].shape[1:], np.uint16)
            for index, det in enumerate(raw[frame], 1):
                if index > 65535:
                    raise ValueError("Too many indexed instances")
                labels[det["mask"]] = index
            Image.fromarray(labels).save(masks / f"{frame:05d}.png")
            detection_frames[str(data["source_frame"][frame])] = dict(labels=f"masks/{frame:05d}.png")
        write_json(out / "detections.json", dict(schema="generic-instance-masks-v1", frames=detection_frames))
        entries = {}
        for name, item in selected.items():
            obs = item["reference"]
            with np.load(transport / "depth.npz", allow_pickle=False) as archive:
                conditioned = object_conditioning_depth(archive, obs["frame"], obs["mask"], obs["trusted"])
            depth_path = out / f"depth-{name}.npz"
            np.savez_compressed(depth_path, **conditioned)
            shape = out / "shapes" / name
            command = [
                sys.executable,
                str(native_script),
                "infer",
                "--video",
                str(transport / "video.mp4"),
                "--masks",
                str(masks),
                "--depth",
                str(depth_path),
                "--out",
                str(shape),
                "--frame",
                str(obs["frame"]),
                "--label",
                str(obs["detection"] + 1),
                "--samples",
                str(pose_samples),
            ]
            with (out / f"{name}.log").open("w") as log:
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                try:
                    code = process.wait(timeout=1800)
                except BaseException:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
                if code:
                    raise RuntimeError(f"Native SAM3D failed for {name}; inspect its local log")
            entry = world_prior(shape, data, obs["frame"], decompose)
            prefix = f"shapes/{name}/world/"
            entry["visual"] = prefix + entry["visual"]
            entry["collision"] = [prefix + p for p in entry["collision"]]
            entries[name] = entry
        write_json(
            out / "meshes.json", dict(schema="generic-world-mesh-priors-v1", units="metres", objects=entries)
        )
        write_json(
            out / "recipe.json",
            dict(
                schema="general-mesh-recipe-v1",
                scene_id=Path(capture).stem,
                evaluation_scope="development",
                static_scene=True,
                capture="capture.npz",
                detections="detections.json",
                mesh_manifest="meshes.json",
                policy=dict(seed=seed),
                jitter_m=jitter,
                samples=samples,
                seed=seed,
                simulate=False,
            ),
        )
        receipt = freeze(out / "recipe.json", out / "mesh-run")
        write_json(out / "mesh-run-receipt.json", receipt)
        result = execute(out / "mesh-run", receipt["lock_sha256"])
        status.update(status="completed", mesh_stage=result)
    except Exception as error:
        status.update(status="failed", error=str(error), elapsed_s=time.monotonic() - started)
        write_json(out / "status.json", status)
        raise
    status["elapsed_s"] = time.monotonic() - started
    write_json(out / "status.json", status)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--native-script", type=Path, required=True)
    p.add_argument("--frames", nargs="+", type=int, required=True)
    p.add_argument("--prompt", default="tabletop object")
    p.add_argument("--pose-samples", type=int, default=8)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--jitter", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    print(json.dumps(run_development(**vars(a)), indent=2))


if __name__ == "__main__":
    main()
