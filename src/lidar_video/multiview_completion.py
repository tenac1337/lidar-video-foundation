"""Development-only completion from explicit construction views and physical priors.

The assigned-square footprint and board support are hard modelling assumptions,
not independent evidence. Reserved views never construct the candidate. All views
must subsequently pass the existing measured geometry gates.
"""

import copy
import shutil
from pathlib import Path

import chess
import cv2
import numpy as np
import trimesh
from PIL import Image

from lidar_video.capture import load_capture, validate
from lidar_video.creation_audit import canonical_case, digest, inputs, read_json, save, verify_hashes
from lidar_video.mesh_alignment import construction_split
from lidar_video.object_evidence import assess_depth
from lidar_video.reconstruction import rigid


def constrained_carve(mesh, views, lower, upper, *, pitch=0.001, max_cells=4_000_000):
    """Intersect a learned volume with construction silhouettes and declared priors."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    if lower.shape != (2,) or upper.shape != (2,) or not np.isfinite([lower, upper]).all():
        raise ValueError("Footprint bounds must be finite XY pairs")
    if not np.isfinite(pitch) or not 0.0005 <= pitch <= 0.002 or np.any(upper - lower <= 2 * pitch):
        raise ValueError("Invalid voxel pitch or footprint extent")
    if len(views) < 2:
        raise ValueError("At least two construction views are required")
    if not len(mesh.faces) or not mesh.is_watertight or not np.isfinite(mesh.vertices).all():
        raise ValueError("A finite watertight learned prior is required")
    if np.prod(np.ceil(mesh.extents / pitch) + 3) > max_cells:
        raise ValueError("Candidate voxel grid exceeds the bounded cell budget")
    voxel = mesh.voxelized(pitch=pitch).fill()
    points = voxel.points
    support = points[:, 2] >= pitch / 2
    footprint = np.all((points[:, :2] >= lower + pitch / 2) & (points[:, :2] <= upper - pitch / 2), axis=1)
    keep = support & footprint
    evidence = dict(
        method="multiview_constrained_completion_v1",
        voxel_pitch_m=float(pitch),
        mask_margin_rgb_pixels=1,
        occupied_voxels_before=len(points),
        below_board_voxels=int((~support).sum()),
        outside_footprint_voxels=int((~footprint).sum()),
        physical_priors={
            "support": "board plane z=0; below-board completion removed, not translated",
            "footprint": "entire object inside assigned square; assumed, not measured",
            "bounds_xy_m": [lower.tolist(), upper.tolist()],
        },
        scale_reapplied=False,
        accepted=False,
        limitations="learned completion, voxel discretization and physical priors; development evidence only",
    )
    for view in views:
        labels, label, k = np.asarray(view["labels"]), view["label"], np.asarray(view["K"])
        if labels.ndim != 2 or labels.dtype.kind not in "ui" or type(label) is not int or label < 1:
            raise ValueError("Construction masks must be nonnegative indexed integer images")
        if np.any(labels < 0) or not np.any(labels == label):
            raise ValueError("Construction instance is missing")
        if (
            k.shape != (3, 3)
            or not np.isfinite(k).all()
            or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1])
            or k[0, 1] != 0
            or k[1, 0] != 0
        ):
            raise ValueError("Invalid construction camera intrinsics")
        pose = rigid(view["board_from_camera"], "board_from_camera")
        camera = (points - pose[:3, 3]) @ pose[:3, :3]
        front = camera[:, 2] > 1e-8
        xy = np.zeros((len(points), 2), dtype=np.int64)
        uv = camera[front, :2] / camera[front, 2:] * np.diag(k)[:2] + k[:2, 2]
        xy[front] = np.rint(np.clip(uv, -1e9, 1e9)).astype(np.int64)
        inside = (
            front
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < labels.shape[1])
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < labels.shape[0])
        )
        allowed = cv2.dilate((labels == label).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        allowed |= (labels != 0) & (labels != label)
        visible_keep = ~inside
        visible_keep[inside] = allowed[xy[inside, 1], xy[inside, 0]]
        keep &= visible_keep
    if not keep.any():
        raise ValueError("Construction constraints removed the entire candidate")
    matrix = np.zeros(voxel.shape, dtype=bool)
    matrix[tuple(voxel.sparse_indices[keep].T)] = True
    candidate = trimesh.voxel.ops.matrix_to_marching_cubes(matrix, pitch=pitch)
    candidate.apply_translation(voxel.transform[:3, 3])
    if not candidate.is_watertight or not np.isfinite(candidate.vertices).all():
        raise ValueError("Completion did not produce a finite watertight candidate")
    evidence.update(occupied_voxels_after=int(keep.sum()), retained_fraction=float(keep.mean()))
    return candidate, evidence


def complete_case(case_file, plan_file, out, *, pitch=0.001):
    """Write a new complete manifest; explicitly transition selected entries to V2."""
    case = canonical_case(case_file)
    if case.get("geometry_scope") != "measured" or not all(
        case.get(k) for k in ("capture", "calibration", "mesh_manifest")
    ):
        raise ValueError("Completion requires a complete measured development case")
    plan = read_json(plan_file)
    source = Path(case["mesh_manifest"])
    manifest = read_json(source)
    objects = manifest.get("objects", {})
    if manifest.get("schema") != "roboskills-chess-meshes-v1" or set(objects) != set(
        case["expected_objects"]
    ):
        raise ValueError("Manifest must contain the complete expected inventory")
    if (
        plan.get("schema") != "multiview-completion-plan-v1"
        or not isinstance(plan.get("objects"), dict)
        or not plan["objects"]
        or not set(plan["objects"]).issubset(objects)
    ):
        raise ValueError("Plan must name nonempty existing objects")
    frozen = inputs(case)
    frozen[str(Path(plan_file).resolve())] = digest(plan_file)
    data = load_capture(case["capture"])
    validate(data)
    cal = read_json(case["calibration"])
    board_from_world = rigid(cal["board_from_world"], "board_from_world")
    size, gap = cal["square_size_m"], cal.get("center_gap_m", 0)
    if not np.isfinite([size, gap]).all() or not 0.01 <= size <= 0.2 or not 0 <= gap <= size:
        raise ValueError("Invalid calibrated square dimensions")
    out = Path(out).resolve()
    if any(out == Path(p) or out.is_relative_to(Path(p)) for p in frozen) or (
        case.get("scene_dir") and out.is_relative_to(Path(case["scene_dir"]))
    ):
        raise ValueError("Output must not replace frozen evidence")
    out.mkdir(parents=True, exist_ok=False)
    result = copy.deepcopy(manifest)
    records = {}
    for name, entry in result["objects"].items():
        for key in ("mesh", "pose", "mask_dir", "masks"):
            if entry.get(key):
                entry[key] = str((source.parent / entry[key]).resolve())
        for observation in entry.get("observations", []):
            observation["mask"] = str((source.parent / observation["mask"]).resolve())
        if name not in plan["objects"]:
            continue
        choice = plan["objects"][name]
        if choice.get("physical_priors") != ["board_plane", "assigned_square"]:
            raise ValueError("Explicitly declare board_plane and assigned_square physical priors")
        if not entry.get("static_scene") or entry.get("validation") != "instance-multiview-v1":
            raise ValueError("Start from a static V1 candidate, not an already fitted multiview candidate")
        entry.update(
            validation="instance-multiview-v2",
            construction_frames=choice.get("construction_frames"),
            validation_frames=choice.get("validation_frames"),
        )
        construction, held_out = construction_split(entry, data)
        views = []
        for observation in entry["observations"]:
            f = observation["frame"]
            if f not in construction:
                continue
            labels = np.asarray(Image.open(observation["mask"]))
            quality, _, _ = assess_depth(data, cal, f, labels == observation["label"], square=name)
            if not quality["passed"]:
                raise ValueError(f"Unreliable construction depth for {name} frame {f}: {quality['reasons']}")
            views.append(
                dict(
                    labels=labels,
                    label=observation["label"],
                    K=data["K"] if data["K"].ndim == 2 else data["K"][f],
                    board_from_camera=board_from_world @ data["world_from_camera"][f],
                )
            )
        pose = rigid(read_json(entry["pose"])["camera_from_object"], "camera_from_object")
        transform = board_from_world @ data["world_from_camera"][entry["frame"]] @ pose
        mesh = trimesh.load(entry["mesh"], force="mesh", process=True)
        mesh.apply_transform(transform)
        square = chess.parse_square(name)
        center = (np.array([chess.square_file(square), chess.square_rank(square)]) - 3.5) * size
        center[1] += np.sign(center[1]) * gap / 2
        candidate, record = constrained_carve(mesh, views, center - size / 2, center + size / 2, pitch=pitch)
        folder = out / "objects" / name
        folder.mkdir(parents=True)
        candidate.apply_transform(np.linalg.inv(transform))
        candidate.export(folder / "mesh.ply")
        shutil.copyfile(entry["pose"], folder / "pose.json")
        record.update(
            construction_frames=construction,
            validation_frames=held_out,
            source_mesh_sha256=digest(entry["mesh"]),
            source_pose_sha256=digest(entry["pose"]),
            output_mesh_sha256=digest(folder / "mesh.ply"),
        )
        entry.update(mesh=str(folder / "mesh.ply"), pose=str(folder / "pose.json"), completion=record)
        records[name] = record
    verify_hashes(frozen)
    save(out / "meshes.json", result)
    case.pop("scene_dir", None)
    case["mesh_manifest"] = str(out / "meshes.json")
    save(out / "case.json", case)
    save(out / "construction.json", dict(plan=plan, frozen_inputs=frozen, objects=records, accepted=False))
    return dict(case=str(out / "case.json"), manifest=str(out / "meshes.json"), accepted=False)
