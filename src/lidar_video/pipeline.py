"""Orchestrate deterministic files; agent orchestration can wrap these CLI stages."""

import importlib.metadata
import importlib.util
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from lidar_video.capture import load_capture, normalize, validate, write_json
from lidar_video.geometry import export_ply, scene_graph
from lidar_video.native_depth import native_intrinsics, native_shape, sample_ids
from lidar_video.scenic_export import export_scenic, sample_scenic
from lidar_video.simulation import simulate
from lidar_video.synthetic import encode_video, generate

UPSTREAM_COMMIT = "eca4691d2dfbb808dd58f3dd279fed154de6828a"


def doctor():
    dependencies = {}
    for name in ("numpy", "scipy", "mujoco", "scenic", "roboskills", "lzfse", "chess", "scikit-learn"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "missing"
    return dict(
        python=sys.version,
        platform=platform.platform(),
        architecture=platform.machine(),
        dependencies=dependencies,
        upstream_commit=UPSTREAM_COMMIT,
        torch_installed=importlib.util.find_spec("torch") is not None,
        full_perception_baseline="not run: requires separate model setup and supported compute",
        current_scope="measured depth bridge, tabletop proxies, synthetic chess recognition, legal Scenic sampling, known-scene Panda grasp",
    )


def prepare_upstream(data, out, reference_frame=0):
    """Write measured depth in the exact upstream contract and the matching video.

    Frames MUST be uniform-rate here: the upstream launcher uses frame-index/FPS
    timing. Reject irregular captures instead of hiding dropped frames by renumbering.
    """
    n = len(data["depth"])
    if type(reference_frame) is not int or not 0 <= reference_frame < n:
        raise ValueError("Reference frame outside capture")
    if n < 2:
        raise ValueError("Upstream video needs at least two frames")
    dt = np.diff(data["rgb_time"])
    period = float(np.median(dt))
    if np.max(np.abs(dt - period)) > 0.001:
        raise ValueError(
            "Variable frame timing / dropped frames: resample explicitly before upstream video export"
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "depth").mkdir()
    (out / "input").mkdir()
    z = data["depth"].astype(np.float32)
    valid = np.isfinite(z) & (z > 0) & (data["confidence"] >= 1)
    np.savez_compressed(
        out / "depth/depth.npz",
        depth=np.where(valid, z, 0),
        valid=valid,
        K=data["K"],
        frames=np.arange(n),
        native_K=native_intrinsics(data),
        native_depth_shape=np.array(native_shape(data)),
        depth_sample_id=np.stack([sample_ids(data, f) for f in range(n)]),
    )
    np.savez_compressed(
        out / "input/sensor_metadata.npz",
        confidence=data["confidence"],
        world_from_camera=data["world_from_camera"],
        rgb_time=data["rgb_time"],
        depth_time=data["depth_time"],
    )
    encode_video(data["rgb"], out / "input/video.mp4", fps=1 / period)
    write_json(out / "input/reference.json", dict(frame=reference_frame))
    (out / "input/object.txt").write_text("tabletop object\n")
    write_json(
        out / "input/depth_source.json",
        dict(
            source="measured_or_synthetic_metric_depth",
            units="metres",
            camera="OpenCV",
            preserved_metric_scale=True,
            skipped_stages=["moge3", "align", "fit"],
            note="Do not run default estimated-depth rescaling or retry sets on this measured-depth workspace.",
        ),
    )
    return out


def preview(data, graph, out):
    z = data["depth"][0]
    good = (z > 0) & np.isfinite(z)
    gray = np.zeros_like(z, dtype=np.uint8)
    lo, hi = np.quantile(z[good], [0.02, 0.98])
    gray[good] = np.clip((z[good] - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)[..., ::-1]
    color[~good] = 0
    h, w = z.shape
    image = Image.new("RGB", (3 * w, h + 34), "#15191c")
    image.paste(Image.fromarray(data["rgb"][0]), (0, 34))
    image.paste(Image.fromarray(color), (w, 34))
    image.paste(Image.fromarray((data["rgb"][0] * 0.5 + color * 0.5).astype(np.uint8)), (2 * w, 34))
    draw = ImageDraw.Draw(image)
    for i, label in enumerate(("RGB · frame 0", "Metric depth · invalid = black", "Registered overlay")):
        draw.text((i * w + 10, 10), label, fill="white")
    image.save(Path(out) / "alignment.png")
    # Simple visual of real exported graph nodes, not an alternative inferred graph.
    canvas = Image.new("RGB", (960, 340), "#f6f4ef")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 20), "Reconstructed scene graph · geometric hypotheses", fill="#15242b")
    nodes = [n for n in graph["nodes"] if n["kind"] == "object"]
    for i, node in enumerate(nodes):
        x = 30 + i * 900 / max(len(nodes), 1)
        draw.rectangle((x, 65, x + 230, 160), outline="#627278", width=2)
        draw.text((x + 12, 80), node["id"], fill="#15242b")
        dims = " × ".join(f"{1000 * v:.0f}" for v in node["size_m"])
        draw.text((x + 12, 108), dims + " mm · box proxy", fill="#15242b")
        draw.line((x + 110, 160, 480, 245), fill="#627278", width=2)
    draw.rectangle((365, 245, 595, 310), outline="#627278", width=2)
    draw.text((385, 265), "support · inferred on edges", fill="#15242b")
    canvas.save(Path(out) / "scene_graph.png")


def process(capture, out, samples=20, seed=7, simulate_output=True):
    data = normalize(load_capture(capture))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "capture_validation.json", validate(data))
    prepare_upstream(data, out / "upstream_workspace")
    graph, cloud, colors = scene_graph(data)
    write_json(out / "scene_graph.json", graph)
    export_ply(out / "frame0.ply", cloud, colors)
    preview(data, graph, out)
    export_scenic(graph, out / "scenario.scenic")
    variants = sample_scenic(out / "scenario.scenic", graph, out / "scenic_samples.json", samples, seed)
    physics = None
    if simulate_output:
        report, check = simulate(graph, out / "simulation")
        physics = dict(check, max_penetration_m=report["max_penetration_m"], warnings=report["warnings"])
        simulate(variants[0], out / "sample_simulation", frames=30)
    result = dict(
        status="passed",
        capture=str(Path(capture).resolve()),
        objects=len(graph["nodes"]) - 1,
        scenic_samples=len(variants),
        physics=physics,
        upstream=doctor(),
        real_device_validation="pending iPhone recording",
        scope="RGB-D -> box proxies -> RoboSkills CPU rollout + graph + Scenic -> sampled rollout",
        omitted=[
            "learned video perception",
            "complete mesh reconstruction",
            "robot/hand manipulation",
            "chess recognition",
            "OMAR deployment",
        ],
    )
    write_json(out / "result.json", result)
    return result


def demo(out, seed=7):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    capture = generate(out / "synthetic", seed=seed)
    result = process(capture, out / "processed", seed=seed)
    import json

    truth = json.loads((out / "synthetic/ground_truth.json").read_text())["objects"]
    graph = json.loads((out / "processed/scene_graph.json").read_text())
    observed = [n for n in graph["nodes"] if n["kind"] == "object"]
    if len(truth) != len(observed):
        raise RuntimeError(f"Synthetic object count mismatch: {len(truth)} vs {len(observed)}")
    transform = np.array(graph["scene_from_world"])
    truth_positions = np.array([n["position"] for n in truth]) @ transform[:3, :3].T + transform[:3, 3]
    positions = np.array([n["position_m"] for n in observed])
    distances = np.linalg.norm(positions[:, None] - truth_positions, axis=-1)
    rows, cols = linear_sum_assignment(distances)
    position_error = float(distances[rows, cols].max())
    size_error = float(
        np.max(
            np.abs(
                np.array([observed[i]["size_m"] for i in rows]) - np.array([truth[j]["size"] for j in cols])
            )
        )
    )
    metrics = dict(
        max_center_error_m=position_error,
        max_dimension_error_m=size_error,
        matched_objects=len(rows),
        position_tolerance_m=0.02,
        dimension_tolerance_m=0.025,
        interpretation="synthetic fixture only; not a real iPhone accuracy measurement",
    )
    write_json(out / "synthetic_evaluation.json", metrics)
    if position_error > 0.02 or size_error > 0.025:
        raise RuntimeError(f"Synthetic geometry acceptance failed: {metrics}")
    return dict(result, synthetic_evaluation=metrics)
