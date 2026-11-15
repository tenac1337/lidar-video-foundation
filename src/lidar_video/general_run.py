"""Hash-locked generic mesh-stage development runs, without recipe overrides.

The detector and mesh priors must already exist and be provenance-labelled.
This is intentionally not labelled end-to-end learned reconstruction readiness.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from lidar_video.capture import load_capture, write_json
from lidar_video.general_scene import InstancePolicy, associate_instances, build_mesh_scene, infer_support
from lidar_video.scene_program_assets import run_asset_scene_program

FIELDS = {
    "schema",
    "scene_id",
    "evaluation_scope",
    "capture",
    "detections",
    "mesh_manifest",
    "static_scene",
    "policy",
    "jitter_m",
    "samples",
    "seed",
    "simulate",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    return {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}


def runtime():
    return dict(
        python=platform.python_version(),
        packages={
            n: importlib.metadata.version(n)
            for n in [
                "numpy",
                "scipy",
                "trimesh",
                "pillow",
                "opencv-python",
                "networkx",
                "scenic",
                "mujoco",
                "rtree",
                "shapely",
            ]
        },
    )


def child(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Asset must be a relative file inside its manifest directory")
    return path


def recipe(config_file):
    path = Path(config_file).resolve()
    config = json.loads(path.read_text())
    if set(config) != FIELDS or config["schema"] != "general-mesh-recipe-v1":
        raise ValueError("Expected exact general-mesh-recipe-v1 fields")
    if config["evaluation_scope"] != "development" or config["static_scene"] is not True:
        raise ValueError("This adapter currently requires explicit static development scope")
    if not isinstance(config["scene_id"], str) or not config["scene_id"].strip():
        raise ValueError("Nonempty scene_id required")
    if type(config["samples"]) is not int or not 1 <= config["samples"] <= 1000:
        raise ValueError("samples must be 1..1000")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be uint32")
    if type(config["simulate"]) is not bool:
        raise ValueError("simulate must be boolean")
    if (
        type(config["jitter_m"]) not in (int, float)
        or not np.isfinite(config["jitter_m"])
        or config["jitter_m"] < 0
    ):
        raise ValueError("Invalid jitter")
    InstancePolicy(**config["policy"])
    for key in ["capture", "detections", "mesh_manifest"]:
        config[key] = str((path.parent / config[key]).resolve())
    paths = [path, *(Path(config[k]) for k in ["capture", "detections", "mesh_manifest"])]
    detection = json.loads(Path(config["detections"]).read_text())
    if detection.get("schema") != "generic-instance-masks-v1" or not isinstance(
        detection.get("frames"), dict
    ):
        raise ValueError("Expected generic-instance-masks-v1")
    for record in detection["frames"].values():
        paths.append(child(Path(config["detections"]).parent, record["labels"]))
    priors = json.loads(Path(config["mesh_manifest"]).read_text())
    if priors.get("schema") != "generic-world-mesh-priors-v1" or priors.get("units") != "metres":
        raise ValueError("Expected metric generic-world-mesh-priors-v1")
    if not isinstance(priors.get("objects"), dict):
        raise ValueError("Mesh prior objects must be a mapping")
    for entry in priors["objects"].values():
        for relative in [entry["visual"], *entry["collision"]]:
            p = child(Path(config["mesh_manifest"]).parent, relative)
            if p.suffix.lower() != ".ply":
                raise ValueError("Locked priors must be self-contained world-frame PLY files")
            paths.append(p)
    return config, {str(p): digest(p) for p in paths}


def freeze(config_file, out):
    out = Path(out).resolve()
    config, hashes = recipe(config_file)
    if any(out == Path(p) or out in Path(p).parents for p in hashes):
        raise ValueError("Output overlaps input files")
    lock = dict(
        schema="general-mesh-run-lock-v1",
        config_file=str(Path(config_file).resolve()),
        config=config,
        inputs=hashes,
        code=code_hashes(),
        runtime=runtime(),
        scope="Generic instance/mesh integration development stage; external perception not evaluated.",
    )
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "lock.json", lock)
    return dict(status="frozen", lock_sha256=digest(out / "lock.json"), run=str(out))


def verify(run, expected):
    run = Path(run)
    if digest(run / "lock.json") != expected:
        raise ValueError("Run lock hash mismatch")
    lock = json.loads((run / "lock.json").read_text())
    if lock.get("schema") != "general-mesh-run-lock-v1":
        raise ValueError("Unsupported run lock")
    config, hashes = recipe(lock["config_file"])
    if config != lock["config"] or hashes != lock["inputs"]:
        raise ValueError("Frozen configuration or inputs changed")
    if code_hashes() != lock["code"] or runtime() != lock["runtime"]:
        raise ValueError("Frozen code or runtime changed")
    return lock


def load_detections(config, data):
    manifest = Path(config["detections"])
    record = json.loads(manifest.read_text())
    sources = {int(s): i for i, s in enumerate(data["source_frame"])}
    result = {}
    for source, entry in record["frames"].items():
        if not source.isdigit() or int(source) not in sources:
            raise ValueError("Detection source frame not in construction capture")
        frame = sources[int(source)]
        if frame in result:
            raise ValueError("Duplicate source frame spelling")
        labels = np.asarray(Image.open(child(manifest.parent, entry["labels"])))
        if labels.shape != data["depth"][frame].shape or labels.dtype.kind not in "ui":
            raise ValueError("Labels must be registered indexed integer PNGs")
        result[frame] = [dict(mask=labels == value) for value in np.unique(labels) if value > 0]
    return result


def execute(run, expected):
    run = Path(run).resolve()
    lock = verify(run, expected)
    with (run / "execution.json").open("x") as f:
        f.write('{"status":"started"}\n')
    started = time.monotonic()
    try:
        config = lock["config"]
        data = load_capture(config["capture"])
        detections = load_detections(config, data)
        policy = InstancePolicy(**config["policy"])
        selected, audit = associate_instances(
            data, detections, static_scene=config["static_scene"], policy=policy
        )
        write_json(run / "association.json", audit)
        if not selected:
            raise ValueError("No corroborated instances; no proxy fallback allowed")
        root = Path(config["mesh_manifest"]).parent
        entries = json.loads(Path(config["mesh_manifest"]).read_text())["objects"]
        priors = {
            name: dict(
                visual=trimesh.load(child(root, e["visual"]), force="mesh", process=False),
                collision=[trimesh.load(child(root, p), force="mesh", process=False) for p in e["collision"]],
                provenance=e["provenance"],
                flat_top_support=e.get("flat_top_support", False),
            )
            for name, e in entries.items()
        }
        support = infer_support(data, detections)
        write_json(run / "support.json", support)
        build_mesh_scene(
            data, selected, priors, support, run / "scene", policy=policy, jitter=config["jitter_m"]
        )
        verify(run, expected)
        replay = run_asset_scene_program(
            run / "scene/scene_graph.json",
            run / "replay",
            count=1,
            seed=config["seed"],
            jitter=config["jitter_m"],
            simulate=config["simulate"],
            frames=51,
            fps=5,
        )
        variants = run_asset_scene_program(
            run / "scene/scene_graph.json",
            run / "variants",
            count=config["samples"],
            randomize=True,
            jitter=config["jitter_m"],
            seed=config["seed"],
        )
        verify(run, expected)
        result = dict(
            status="completed_with_exclusions" if audit["rejected"] else "completed",
            scene_id=config["scene_id"],
            evaluation_scope="development",
            associated_objects=len(selected),
            replay_exact=replay["replay_nodes_edges_exact"],
            variation_attempts=variants["attempts"],
            elapsed_s=time.monotonic() - started,
            lock_sha256=expected,
            physics=replay["physics"],
            limits=[
                "External detector/mesh priors supplied, not learned end-to-end in this run",
                "Detection completeness and hidden geometry not certified",
                "Native appearance uses observed median color; texture transfer unimplemented",
                "Fixed-orientation envelope/support constraints; no task-success guarantee",
            ],
        )
    except Exception as error:
        write_json(
            run / "result.json",
            dict(
                status="failed", error=str(error), lock_sha256=expected, elapsed_s=time.monotonic() - started
            ),
        )
        raise
    write_json(run / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("freeze")
    a.add_argument("config", type=Path)
    a.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("run")
    a.add_argument("run", type=Path)
    a.add_argument("--lock-sha256", required=True)
    a = p.parse_args()
    result = freeze(a.config, a.out) if a.command == "freeze" else execute(a.run, a.lock_sha256)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
