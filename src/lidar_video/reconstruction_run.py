"""Lock a development reconstruction recipe and execute it without CLI overrides."""

import platform
import time
from importlib.metadata import version
from pathlib import Path

import chess
import numpy as np
import trimesh
from PIL import Image

from lidar_video.capture import load_capture
from lidar_video.comparison import compare_exports, seal_exports
from lidar_video.creation_audit import digest, inputs, policy_hashes, read_json, save
from lidar_video.reconstruction import reconstruct, rigid

FIELDS = {
    "schema", "scene_id", "evaluation_scope", "capture", "calibration", "masks",
    "mesh_manifest", "expected_objects", "fusion_source_frames", "reference_source_frame",
    "static_scene", "mesh_policy", "min_height_m", "samples", "seed", "simulation",
}


def _runtime():
    return dict(
        python=platform.python_version(), platform=platform.platform(),
        packages={name: version(name) for name in (
            "numpy", "scipy", "pillow", "opencv-python", "trimesh", "scenic",
            "coacd", "fast-simplification", "mujoco", "roboskills",
        )},
    )


def _config(path):
    config = read_json(path)
    if not isinstance(config, dict) or set(config) != FIELDS:
        raise ValueError("Reconstruction config must contain exactly the documented fields")
    if config["schema"] != "reconstruction-run-v1" or config["evaluation_scope"] != "development":
        raise ValueError("Expected reconstruction-run-v1 with development scope")
    if not isinstance(config["scene_id"], str) or not config["scene_id"].strip():
        raise ValueError("scene_id must be nonempty")
    expected = config["expected_objects"]
    if (not isinstance(expected, list) or not expected
            or any(not isinstance(s, str) or s not in chess.SQUARE_NAMES for s in expected)
            or len(set(expected)) != len(expected)):
        raise ValueError("expected_objects requires unique chess square IDs")
    frames = config["fusion_source_frames"]
    if (not isinstance(frames, list) or not frames or any(type(f) is not int for f in frames)
            or len(set(frames)) != len(frames)):
        raise ValueError("fusion_source_frames requires unique integer source IDs")
    if type(config["reference_source_frame"]) is not int or config["reference_source_frame"] not in frames:
        raise ValueError("reference_source_frame must be in fusion_source_frames")
    if config["static_scene"] is not True or config["mesh_policy"] != "strict":
        raise ValueError("Locked runs require static_scene=true and mesh_policy=strict")
    if type(config["simulation"]) is not bool:
        raise ValueError("simulation must be boolean")
    if type(config["samples"]) is not int or config["samples"] < 1:
        raise ValueError("samples must be a positive integer")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be a uint32 integer")
    height = config["min_height_m"]
    if type(height) not in (int, float) or not np.isfinite(height) or not 0 < height < .2:
        raise ValueError("min_height_m must be finite and between zero and 0.2 metres")
    for key in ("capture", "calibration", "masks", "mesh_manifest"):
        if key == "mesh_manifest" and config[key] is None:
            continue
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} requires a path")
        config[key] = str((path.parent / config[key]).resolve())
    return config


def _snapshot(config, config_path):
    hashes = inputs(config)
    masks = Path(config["masks"])
    if not masks.is_dir():
        raise ValueError("masks must be an instance-assigned mask directory")
    hashes.update({str(p.resolve()): digest(p) for p in masks.rglob("*") if p.is_file()})
    hashes[str(config_path)] = digest(config_path)
    if any(value is None for value in hashes.values()):
        raise ValueError("A required reconstruction input is missing")
    return hashes


def _preflight(config):
    data = load_capture(config["capture"])
    sources = np.asarray(data.get("source_frame"))
    if (sources.shape != (len(data["depth"]),) or sources.dtype.kind not in "iu"
            or len(np.unique(sources)) != len(sources)):
        raise ValueError("Capture must retain unique integer source_frame IDs")
    index = {int(source): i for i, source in enumerate(sources)}
    if not set(config["fusion_source_frames"]) <= set(index):
        raise ValueError("A selected source frame is absent from the capture")
    frames = [index[f] for f in config["fusion_source_frames"]]
    cal = read_json(config["calibration"])
    rigid(cal["board_from_world"], "board_from_world")
    size, gap = cal["square_size_m"], cal.get("center_gap_m", 0)
    if (type(size) not in (int, float) or not np.isfinite(size) or not .01 <= size <= .2
            or type(gap) not in (int, float) or not np.isfinite(gap) or not 0 <= gap <= size):
        raise ValueError("Invalid board square size or center gap")
    if "reference_frame" in cal:
        ref = cal["reference_frame"]
        if type(ref) is not int or ref not in frames:
            raise ValueError("Calibration reference_frame must be among selected capture indices")
        if "source_frame" in cal and cal["source_frame"] != int(sources[ref]):
            raise ValueError("Calibration source_frame disagrees with its capture index")
    masks = Path(config["masks"])
    assignments = read_json(masks / "instances.json")
    if assignments.get("schema") != "chess-instance-assignments-v1":
        raise ValueError("Require chess-instance-assignments-v1; broad unassigned masks are unsupported")
    expected, observed = set(config["expected_objects"]), set()
    for frame in frames:
        mapping = assignments.get("frames", {}).get(str(frame))
        if not isinstance(mapping, dict):
            raise ValueError(f"Missing instance assignments for capture index {frame}")
        labels = np.asarray(Image.open(masks / f"{frame:05d}.png"))
        if labels.shape != data["depth"][frame].shape or labels.dtype.kind not in "ui":
            raise ValueError("Masks must be registered indexed integer PNGs")
        keys = {str(int(label)) for label in np.unique(labels) if label != 0}
        if keys != set(mapping):
            raise ValueError("Every nonzero mask label must have exactly one present assignment")
        if (any(not isinstance(s, str) or s not in expected for s in mapping.values())
                or len(set(mapping.values())) != len(mapping)):
            raise ValueError("Mask assignments must be unique expected objects within each frame")
        observed.update(mapping.values())
    if observed != expected:
        raise ValueError("Mask inventory does not match expected_objects")
    if config["mesh_manifest"] is not None:
        path = Path(config["mesh_manifest"])
        manifest = read_json(path)
        entries = manifest.get("objects", {})
        if manifest.get("schema") != "roboskills-chess-meshes-v1" or set(entries) != expected:
            raise ValueError("Mesh manifest inventory must exactly match expected_objects")
        for entry in entries.values():
            # PLY has no external material/geometry references omitted by the lock.
            if Path(entry["mesh"]).suffix.lower() != ".ply":
                raise ValueError("Locked learned candidates require self-contained PLY meshes")
            mesh = trimesh.load(path.parent / entry["mesh"], force="mesh")
            if (not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces)
                    or not np.isfinite(mesh.vertices).all() or not np.isfinite(mesh.area) or mesh.area <= 0):
                raise ValueError("Learned candidates require finite nonzero-area meshes")
            rigid(read_json(path.parent / entry["pose"])["camera_from_object"], "camera_from_object")
            used = [entry.get("frame", 0)] + [o["frame"] for o in entry.get("observations", [])]
            if any(type(f) is not int or f not in frames for f in used):
                raise ValueError("Candidate reference and observation indices must be selected")
            for obs in entry.get("observations", []):
                labels = np.asarray(Image.open(path.parent / obs["mask"]))
                if (labels.shape != data["depth"][obs["frame"]].shape or labels.dtype.kind not in "ui"
                        or type(obs["label"]) is not int or obs["label"] < 1
                        or not np.any(labels == obs["label"])):
                    raise ValueError("Candidate observation mask/label is invalid")
    return dict(fusion_frames=frames, reference_frame=index[config["reference_source_frame"]])


def freeze_reconstruction(config_file, out):
    config_file, out = Path(config_file).resolve(), Path(out).resolve()
    initial = digest(config_file)
    config = _config(config_file)
    before = _snapshot(config, config_file)
    selection = _preflight(config)
    if before != _snapshot(config, config_file) or digest(config_file) != initial:
        raise ValueError("Reconstruction inputs changed during preflight")
    # Reject output/input overlap before writing anything, including mask inventory additions.
    directories = [Path(config["masks"])]
    if config["mesh_manifest"]:
        manifest = Path(config["mesh_manifest"])
        for entry in read_json(manifest)["objects"].values():
            directories += [(manifest.parent / entry[k]).resolve()
                            for k in ("mask_dir", "masks") if entry.get(k)]
    if (any(out == Path(p) or out in Path(p).parents for p in before)
            or any(out.is_relative_to(d) for d in directories)):
        raise ValueError("Run outputs must be separate from reconstruction inputs")
    lock = dict(schema="locked-reconstruction-v1", config=config, config_file=str(config_file),
                inputs=before, selection=selection, code=policy_hashes(), runtime=_runtime())
    out.mkdir(parents=True, exist_ok=False)
    save(out / "lock.json", lock)
    return dict(status="frozen", out=str(out), lock_sha256=digest(out / "lock.json"), selection=selection)


def _verify(lock, run, expected):
    if digest(run / "lock.json") != expected:
        raise ValueError("Reconstruction lock hash mismatch")
    if (lock["inputs"] != _snapshot(lock["config"], Path(lock["config_file"]))
            or lock["code"] != policy_hashes() or lock["runtime"] != _runtime()):
        raise ValueError("Locked inputs, code or runtime changed; create a new run")


def run_reconstruction(run, *, expected_lock_sha256):
    run = Path(run).resolve()
    if digest(run / "lock.json") != expected_lock_sha256:
        raise ValueError("Reconstruction lock hash mismatch")
    lock = read_json(run / "lock.json")
    if lock.get("schema") != "locked-reconstruction-v1":
        raise ValueError("Expected locked-reconstruction-v1")
    _verify(lock, run, expected_lock_sha256)
    config = lock["config"]
    selection = _preflight(config)
    if selection != lock["selection"]:
        raise ValueError("Frame selection differs from the locked configuration")
    _verify(lock, run, expected_lock_sha256)
    # Claim execution once; partial and failed attempts are preserved, never retried in place.
    with (run / "execution.json").open("x") as stream:
        stream.write('{"status":"started"}\n')
    started = time.perf_counter()
    try:
        report = reconstruct(
            config["capture"], config["calibration"], run / "scene",
            masks=config["masks"], mesh_manifest=config["mesh_manifest"], mesh_policy=config["mesh_policy"],
            static_scene=config["static_scene"], min_height=config["min_height_m"],
            samples=config["samples"], seed=config["seed"], simulation=config["simulation"], **selection,
        )
        _verify(lock, run, expected_lock_sha256)
        seal_exports(run / "scene", run / "scene-seal.json")
        save(run / "comparison-case.json", dict(
            schema="export-comparison-v1", case_id=config["scene_id"] + "-reconstruction",
            scene_id=config["scene_id"], scene_dir="scene", expected_objects=config["expected_objects"],
            evidence_seal="scene-seal.json",
        ))
        comparison = compare_exports(run / "comparison-case.json", run / "comparison")
        _verify(lock, run, expected_lock_sha256)
        decisions = {d["decision"] for d in comparison["decisions"]}
        status = "rejected" if "reject" in decisions else "blocked" if "blocked" in decisions else "completed"
        result = dict(status=status, lock_sha256=expected_lock_sha256,
                      elapsed_seconds=time.perf_counter() - started, reconstruction=report,
                      comparison=comparison, evaluation_scope="development",
                      limitation="Completion is a software contract check, not physical truth or calibrated uncertainty.")
    except Exception as error:
        save(run / "result.json", dict(status="failed", error=str(error),
                                      lock_sha256=expected_lock_sha256,
                                      elapsed_seconds=time.perf_counter() - started))
        raise ValueError(f"Reconstruction run failed; partial outputs preserved: {error}") from error
    save(run / "result.json", result)
    return result
