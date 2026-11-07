"""Read-only Stray export inspection before import, inference or frame splitting."""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from lidar_video.creation_audit import digest, read_json
from lidar_video.stray import read_odometry, uniform_selection


def _files(source):
    paths = [source / "odometry.csv", source / "rgb.mp4"]
    for folder in ("depth", "confidence"):
        paths.extend(sorted((source / folder).glob("*.png")))
    return {str(p.relative_to(source)): digest(p) for p in paths if p.is_file()}


def _video_info(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-ignore_editlist", "1", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode:
        raise ValueError(f"Video decode inspection failed: {result.stderr[-2000:]}")
    try:
        streams = json.loads(result.stdout)["streams"]
        if len(streams) != 1:
            raise ValueError("Require one selected video stream")
        info = {k: int(streams[0][k]) for k in ("width", "height", "nb_read_frames")}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Video dimensions or decoded frame count are unavailable") from error
    if min(info.values()) <= 0:
        raise ValueError("Video dimensions and frame count must be positive")
    if result.stderr.strip():
        raise ValueError(f"Video decoder reported errors: {result.stderr[-2000:]}")
    return info


def _check_plan(path, selected):
    plan = read_json(path)
    if not isinstance(plan, dict) or plan.get("schema") != "scene-evaluation-split-v1":
        raise ValueError("Expected scene-evaluation-split-v1")
    sets = {}
    for name in ("construction_source_frames", "evaluation_source_frames"):
        values = plan.get(name)
        if (
            not isinstance(values, list) or not values
            or any(type(v) is not int or v < 0 for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"{name} must contain unique nonnegative integer source IDs")
        sets[name] = set(values)
    if sets["construction_source_frames"] & sets["evaluation_source_frames"]:
        raise ValueError("Construction and evaluation source frames overlap")
    if not set.union(*sets.values()) <= set(selected):
        raise ValueError("Split plan references frames absent from the selected import timeline")
    return {name: sorted(values) for name, values in sets.items()}


def inspect_stray(source, out, *, fps=2.0, plan=None):
    """Write an intake report even on invalid/blocked input; never modify the source."""
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.is_relative_to(source):
        raise ValueError("Intake report must be outside the source export")
    if out.exists():
        raise FileExistsError("Intake report already exists")
    plan = Path(plan).resolve() if plan is not None else None
    errors, blocked, warnings = [], [], []
    report = dict(
        schema="stray-intake-report-v1", source=str(source), requested_fps=fps,
        errors=errors, blocked=blocked, warnings=warnings,
        scope="Format and frame-association checks only; no reconstruction or data-quality certification",
        independent_depth_timestamps=False, video_edit_list_ignored=True,
        limits=[
            "App-provided RGB/depth association does not measure hardware exposure-time skew.",
            "Readiness does not verify sharpness, static objects, physical scale or calibration accuracy.",
            "A split plan does not prove its evaluation images were never inspected.",
        ],
    )
    before = {}
    plan_digest = None
    times, selected, shape, video = None, None, None, None
    try:
        if isinstance(fps, bool) or not np.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be finite and positive")
        if not source.is_dir():
            raise ValueError("Source export directory is missing")
        before = _files(source)
        report["source_files_sha256"] = before
        if (source / "distortion").exists():
            errors.append(dict(stage="format", reason="Distortion lookup tables are unsupported"))
        for name in ("odometry.csv", "rgb.mp4"):
            if name not in before:
                errors.append(dict(stage="inventory", reason=f"Missing required file: {name}"))
        if "odometry.csv" in before:
            try:
                times, _, intrinsics = read_odometry(source / "odometry.csv")
                if np.any(intrinsics[:, (0, 1), (0, 1)] <= 0):
                    raise ValueError("Focal lengths must be positive")
                selected, targets = uniform_selection(times, fps)
                report.update(
                    source_frames=len(times), source_duration_s=float(times[-1] - times[0]),
                    selected_source_frames=selected.tolist(), source_time_s=times[selected].tolist(),
                    import_time_s=targets.tolist(),
                    max_time_displacement_s=float(np.max(np.abs(times[selected] - targets))),
                )
                if len(selected) < 2:
                    warnings.append(dict(reason="Sampling selects fewer than two source views"))
            except (ValueError, OSError, TypeError) as error:
                errors.append(dict(stage="odometry", reason=str(error)))
        if times is not None:
            expected = {f"{i:06d}.png" for i in range(len(times))}
            complete = True
            for name in ("depth", "confidence"):
                actual = {p.name for p in (source / name).glob("*.png") if p.is_file()}
                missing, extra = sorted(expected - actual), sorted(actual - expected)
                if missing or extra:
                    complete = False
                    errors.append(dict(stage="inventory", folder=name, missing=missing, extra=extra,
                                       reason="Frame IDs disagree with odometry"))
            if complete:
                empty = []
                valid_fractions = []
                for frame in range(len(times)):
                    name = f"{frame:06d}.png"
                    depth = cv2.imread(str(source / "depth" / name), cv2.IMREAD_UNCHANGED)
                    confidence = cv2.imread(str(source / "confidence" / name), cv2.IMREAD_UNCHANGED)
                    if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
                        errors.append(dict(stage="depth", source_frame=frame,
                                           reason="Require a uint16 millimetre depth image"))
                        continue
                    if shape is None:
                        shape = depth.shape
                    if depth.shape != shape:
                        errors.append(dict(stage="depth", source_frame=frame,
                                           reason="Depth grid dimensions change during recording"))
                        continue
                    if (confidence is None or confidence.shape != depth.shape
                            or confidence.dtype != np.uint8 or not np.isin(confidence, [0, 1, 2]).all()):
                        errors.append(dict(stage="confidence", source_frame=frame,
                                           reason="Require matching uint8 confidence with levels 0, 1, 2"))
                        continue
                    valid = (depth > 0) & (confidence >= 1)
                    valid_fractions.append(float(valid.mean()))
                    if not valid.any():
                        empty.append(frame)
                report["native_depth_shape"] = list(shape) if shape is not None else None
                report["depth_frames_inspected"] = len(times)
                report["valid_depth_fraction_min"] = min(valid_fractions) if valid_fractions else None
                if empty:
                    warnings.append(dict(reason="Frames without usable depth", source_frames=empty))
                    selected_empty = sorted(set(empty) & set(selected if selected is not None else []))
                    if selected_empty:
                        errors.append(dict(stage="sampling", source_frames=selected_empty,
                                           reason="Selected frames have no usable depth"))
        if "rgb.mp4" in before:
            try:
                video = _video_info(source / "rgb.mp4")
                report["video"] = video
                if times is not None and video["nb_read_frames"] != len(times):
                    errors.append(dict(stage="video", reason="Decoded RGB count differs from odometry"))
                if shape is not None and not np.isclose(video["width"] / video["height"], shape[1] / shape[0]):
                    errors.append(dict(stage="video", reason="RGB/depth aspect ratios differ"))
            except (FileNotFoundError, subprocess.TimeoutExpired) as error:
                blocked.append(dict(stage="video", reason=f"Video inspection unavailable: {error}"))
            except (ValueError, OSError) as error:
                errors.append(dict(stage="video", reason=str(error)))
        if plan is not None:
            try:
                plan_digest = digest(plan)
                report["split_plan"] = _check_plan(plan, selected if selected is not None else [])
                report["split_plan_sha256"] = plan_digest
            except (ValueError, OSError, TypeError) as error:
                errors.append(dict(stage="split", reason=str(error)))
        if _files(source) != before:
            errors.append(dict(stage="integrity", reason="Export changed during inspection"))
        if plan_digest is not None and digest(plan) != plan_digest:
            errors.append(dict(stage="integrity", reason="Split plan changed during inspection"))
    except (ValueError, OSError, TypeError) as error:
        errors.append(dict(stage="intake", reason=str(error)))
    report["status"] = "invalid" if errors else "blocked" if blocked else "ready"
    # Never serialize a non-finite invalid CLI argument into an otherwise usable error report.
    if not isinstance(fps, (int, float)) or not np.isfinite(fps):
        report["requested_fps"] = None
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
