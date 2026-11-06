"""Measure frozen predictions without fitting, selection, repair, or acceptance gates.

Hashes detect changes after freezing; they cannot prove a human never saw test data.
This module does not call the development creation/audit loop.
"""

from importlib.metadata import version
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from lidar_video.capture import load_capture, validate
from lidar_video.creation_audit import digest, read_json, save, verify_hashes
from lidar_video.mesh_alignment import silhouette_iou
from lidar_video.native_depth import sample_ids
from lidar_video.reconstruction import calibrated_points, rigid

METRICS = {
    "surface_samples": 60000,
    "surface_seed": 0,
    "min_confidence": 1,
    "silhouette_max_width_px": 320,
    "depth_selection": "all valid unique native samples inside the reference instance mask",
    "distance": "one-way observation-to-sampled-mesh Euclidean distance in metres",
    "silhouette": "projected mesh mask excluding pixels labeled as other instances",
}


def _ids(values, name):
    if (
        not isinstance(values, list)
        or not values
        or any(type(v) is not int or v < 0 for v in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} must contain unique nonnegative integer source frame IDs")
    return set(values)


def _names(values, name):
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(v, str) or not v.strip() for v in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} must contain unique nonempty IDs")
    return set(values)


def split_evaluation_capture(capture, plan_file, out):
    """Materialize separate pixel bundles before giving construction data to a builder."""
    capture, plan_file, out = Path(capture).resolve(), Path(plan_file).resolve(), Path(out).resolve()
    original_hashes = {str(p): digest(p) for p in (capture, plan_file)}
    plan = read_json(plan_file)
    if plan.get("schema") != "scene-evaluation-split-v1":
        raise ValueError("Expected scene-evaluation-split-v1")
    construction = _ids(plan.get("construction_source_frames"), "construction_source_frames")
    evaluation = _ids(plan.get("evaluation_source_frames"), "evaluation_source_frames")
    if construction & evaluation:
        raise ValueError("Construction and evaluation frames overlap")
    data = load_capture(capture)
    n = len(data["depth"])
    sources = np.asarray(data.get("source_frame"))
    if (
        sources.shape != (n,) or sources.dtype.kind not in "iu"
        or np.any(sources < 0) or len(set(sources.tolist())) != n
    ):
        raise ValueError("Capture requires unique original source_frame IDs")
    if not (construction | evaluation) <= set(sources.tolist()):
        raise ValueError("Split references a source frame absent from the capture")
    per_frame = {
        "rgb", "depth", "confidence", "world_from_camera", "rgb_time", "depth_time",
        "depth_sample_id", "source_frame", "source_time",
    }
    static = {"native_depth_shape", "source_kind"}
    unknown = set(data) - per_frame - static - {"K", "native_K"}
    if unknown:
        raise ValueError(f"Declare slicing semantics for unknown capture fields: {sorted(unknown)}")
    bundles = {}
    for role, selected in (("construction", construction), ("evaluation", evaluation)):
        indices = np.flatnonzero(np.isin(sources, list(selected)))
        bundle = {}
        for key, value in data.items():
            if key in per_frame or (key in ("K", "native_K") and value.ndim == 3):
                if value.shape[0] != n:
                    raise ValueError(f"Frame count mismatch in {key}")
                bundle[key] = value[indices]
            else:
                bundle[key] = value.copy()
        validate(bundle)
        bundles[role] = bundle
    verify_hashes(original_hashes)
    out.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for role, bundle in bundles.items():
        folder = out / role
        folder.mkdir()
        path = folder / "capture.npz"
        np.savez_compressed(path, **bundle)
        outputs[role] = dict(path=str(path), sha256=digest(path), source_frames=bundle["source_frame"].tolist())
    result = dict(
        status="split", inputs=original_hashes, outputs=outputs,
        omitted_source_frames=[int(f) for f in sources if f not in construction | evaluation],
        limitation="Separate files are not access controls. Give builders only the construction folder.",
    )
    save(out / "split.json", result)
    return result


def _runtime():
    return {name: version(name) for name in ("numpy", "scipy", "trimesh", "pillow", "opencv-python")}


def _code():
    return {str(p): digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def freeze_evaluation(case_file, out):
    """Freeze supplied files/settings. Does not read image pixels or run inference."""
    case_file, out = Path(case_file).resolve(), Path(out).resolve()
    case = read_json(case_file)
    if case.get("schema") != "scene-evaluation-case-v1":
        raise ValueError("Expected scene-evaluation-case-v1")
    scope = case.get("evaluation_scope")
    if scope not in ("development_replay", "confirmatory"):
        raise ValueError("Explicit development_replay or confirmatory scope is required")
    if scope == "confirmatory" and (
        case.get("previously_inspected") is not False
        or not isinstance(case.get("custodian_attestation"), str)
        or not case["custodian_attestation"].strip()
    ):
        raise ValueError("Confirmatory scope requires an explicit data-history attestation")
    if not isinstance(case.get("scene_id"), str) or not case["scene_id"].strip():
        raise ValueError("scene_id is required")
    expected = _names(case.get("expected_objects"), "expected_objects")
    held = _ids(case.get("evaluation_source_frames"), "evaluation_source_frames")
    for key in ("prediction", "capture", "calibration"):
        case[key] = str((case_file.parent / case[key]).resolve())
    prediction = read_json(case["prediction"])
    if prediction.get("schema") != "frozen-scene-prediction-v1":
        raise ValueError("Expected frozen-scene-prediction-v1")
    if prediction.get("scene_id") != case["scene_id"]:
        raise ValueError("Prediction scene_id differs from evaluation scene_id")
    if not isinstance(prediction.get("method_id"), str) or not prediction["method_id"].strip():
        raise ValueError("method_id is required")
    if prediction.get("method_decision") not in ("accept", "reject", "abstain"):
        raise ValueError("Record the method decision before evaluation")
    if prediction.get("units") != "metres":
        raise ValueError("Prediction units must be metres")
    seen = _ids(prediction.get("construction_source_frames"), "construction_source_frames")
    seen |= _ids(prediction.get("calibration_source_frames"), "calibration_source_frames")
    if held & seen:
        raise ValueError("Evaluation source frames overlap construction/calibration source frames")
    objects = prediction.get("objects")
    if not isinstance(objects, list):
        raise ValueError("objects must be a list (empty predictions are recorded as missing)")
    names = [o["id"] for o in objects]
    if names:
        _names(names, "prediction object IDs")
    paths = {case_file, *(Path(case[k]) for k in ("prediction", "capture", "calibration"))}
    for obj in objects:
        obj["mesh"] = str((Path(case["prediction"]).parent / obj["mesh"]).resolve())
        if Path(obj["mesh"]).suffix.lower() != ".ply":
            raise ValueError("Use self-contained PLY meshes; external asset references are unsupported")
        rigid(obj["board_from_mesh"], "board_from_mesh")
        paths.add(Path(obj["mesh"]))
    observations = case.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("Nonempty evaluation observations are required")
    pairs, represented, frames = set(), set(), set()
    masks_by_frame, labels_by_frame = {}, {}
    for obs in observations:
        obj, source, label = obs["object_id"], obs["source_frame"], obs["label"]
        if obj not in expected or type(source) is not int or source not in held:
            raise ValueError("Observation object/frame is outside the declared evaluation inventory")
        if type(label) is not int or label < 1 or (obj, source) in pairs:
            raise ValueError("Duplicate observation or invalid positive instance label")
        obs["mask"] = str((case_file.parent / obs["mask"]).resolve())
        if source in masks_by_frame and masks_by_frame[source] != obs["mask"]:
            raise ValueError("All objects in a frame must share one indexed instance mask")
        if label in labels_by_frame.setdefault(source, set()):
            raise ValueError("Two objects cannot share an instance label in the same frame")
        labels_by_frame[source].add(label)
        masks_by_frame[source] = obs["mask"]
        pairs.add((obj, source))
        represented.add(obj)
        frames.add(source)
        paths.add(Path(obs["mask"]))
    if represented != expected or frames != held:
        raise ValueError("Observations must cover every expected object and evaluation frame")
    if any(p == out or out in p.parents for p in paths):
        raise ValueError("Evaluation outputs must be separate from frozen input files")
    hashes = {str(p): digest(p) for p in sorted(paths)}
    lock = dict(
        schema="frozen-scene-evaluation-v1", case=case, prediction=prediction,
        files=hashes, code=_code(), runtime=_runtime(), metrics=METRICS,
        limitation="Data history is a custodian declaration, not proven by hashes.",
    )
    out.mkdir(parents=True, exist_ok=False)
    save(out / "lock.json", lock)
    return {"status": "frozen", "lock_sha256": digest(out / "lock.json"), "out": str(out)}


def _load_mesh(obj):
    mesh = trimesh.load(obj["mesh"], force="mesh", process=False)
    if (
        not isinstance(mesh, trimesh.Trimesh)
        or not len(mesh.vertices)
        or not len(mesh.faces)
        or not np.isfinite(mesh.vertices).all()
        or not np.isfinite(mesh.area)
        or mesh.area <= 0
    ):
        raise ValueError("Prediction has no finite nonzero-area mesh")
    mesh.apply_transform(rigid(obj["board_from_mesh"], "board_from_mesh"))
    sampled, _ = trimesh.sample.sample_surface(mesh, METRICS["surface_samples"], seed=0)
    return mesh, cKDTree(sampled)


def score_evaluation(run, *, expected_lock_sha256):
    """Measure once from a caller-pinned lock; never align or alter candidates."""
    run = Path(run).resolve()
    if (run / "metrics.json").exists():
        raise FileExistsError("Evaluation already scored; preserve the original result")
    lock_path = run / "lock.json"
    if digest(lock_path) != expected_lock_sha256:
        raise ValueError("Evaluation lock differs from the precommitted SHA256")
    lock = read_json(lock_path)
    if lock.get("schema") != "frozen-scene-evaluation-v1" or lock["metrics"] != METRICS:
        raise ValueError("Unsupported evaluation lock or metric policy")
    verify_hashes(lock["files"])
    verify_hashes(lock["code"])
    if _code() != lock["code"] or _runtime() != lock["runtime"]:
        raise ValueError("Evaluator code or numerical runtime changed after freezing")
    case, prediction = lock["case"], lock["prediction"]
    data = load_capture(case["capture"])
    sources = np.asarray(data.get("source_frame"))
    if (
        sources.shape != (len(data["depth"]),) or sources.dtype.kind not in "iu"
        or np.any(sources < 0) or len(set(sources.tolist())) != len(sources)
    ):
        raise ValueError("Capture requires unique original source_frame IDs")
    indices = {int(source): i for i, source in enumerate(sources)}
    if not set(case["evaluation_source_frames"]) <= indices.keys():
        raise ValueError("An evaluation source frame is absent from the capture")
    calibration = read_json(case["calibration"])
    rigid(calibration["board_from_world"], "board_from_world")
    objects = {o["id"]: o for o in prediction["objects"]}
    loaded, invalid = {}, {}
    for name, obj in objects.items():
        try:
            loaded[name] = _load_mesh(obj)
        except (ValueError, TypeError, IndexError) as error:
            invalid[name] = str(error)
    rows = []
    for obs in case["observations"]:
        frame, name = indices[obs["source_frame"]], obs["object_id"]
        labels = np.asarray(Image.open(obs["mask"]))
        if labels.shape != data["depth"].shape[1:] or labels.dtype.kind not in "iu":
            raise ValueError("Evaluation masks must be indexed integers matching the capture grid")
        mask = labels == obs["label"]
        row = dict(object_id=name, source_frame=obs["source_frame"], reference_pixels=int(mask.sum()))
        if not mask.any():
            row.update(status="unscorable_reference", reason="Instance label absent from reference mask")
        elif name not in objects:
            row.update(status="missing_prediction", silhouette_iou=0.0)
        elif name in invalid:
            row.update(status="invalid_prediction", reason=invalid[name])
        else:
            mesh, tree = loaded[name]
            z, ids = data["depth"][frame], sample_ids(data, frame)
            valid = mask & np.isfinite(z) & (z > 0) & (data["confidence"][frame] >= 1) & (ids >= 0)
            selected = np.flatnonzero(valid)
            _, first = np.unique(ids.ravel()[selected], return_index=True)
            selected = selected[first]
            cloud = calibrated_points(data, calibration, frame).reshape(-1, 3)[selected]
            row.update(
                status="scored" if len(cloud) else "no_usable_depth",
                native_samples=len(cloud),
                silhouette_iou=silhouette_iou(mesh, labels, obs["label"], data, calibration, frame),
            )
            if len(cloud):
                distances = tree.query(cloud)[0]
                row.update(median_m=float(np.median(distances)), p90_m=float(np.quantile(distances, .9)))
        rows.append(row)
    object_summaries = []
    for name in case["expected_objects"]:
        subset = [r for r in rows if r["object_id"] == name]
        depths = [r["median_m"] for r in subset if "median_m" in r]
        silhouettes = [r["silhouette_iou"] for r in subset if "silhouette_iou" in r]
        object_summaries.append(dict(
            object_id=name, views=len(subset), depth_scored_views=len(depths),
            mean_view_median_m=float(np.mean(depths)) if depths else None,
            mean_view_silhouette_iou=float(np.mean(silhouettes)) if silhouettes else None,
        ))
    expected = set(case["expected_objects"])
    result = dict(
        schema="scene-evaluation-metrics-v1", status="scored", scene_id=case["scene_id"],
        method_id=prediction["method_id"], method_decision=prediction["method_decision"],
        evaluation_scope=case["evaluation_scope"], lock_sha256=expected_lock_sha256,
        missing_objects=sorted(expected - objects.keys()),
        unexpected_objects=sorted(objects.keys() - expected), invalid_predictions=invalid,
        unscorable_views=sum(r["status"] != "scored" for r in rows),
        views=rows, objects=object_summaries, metrics=METRICS,
        limits=[
            "Scored is execution status, not scene acceptance or independent correctness certification.",
            "Observed surface agreement does not measure hidden geometry or sensor accuracy.",
            "Distance uses surface sampling, not exact triangle distance; no fitting is performed.",
            "Mask other-object pixels are excluded; this is not a full visibility renderer.",
            "No pooled average is provided that could hide missing objects or unscorable views.",
        ],
    )
    verify_hashes(lock["files"])
    verify_hashes(lock["code"])
    if digest(lock_path) != expected_lock_sha256:
        raise ValueError("Evaluation lock changed during scoring")
    save(run / "metrics.json", result)
    return result
