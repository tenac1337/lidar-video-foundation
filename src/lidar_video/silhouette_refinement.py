"""Create reference-only silhouette-carved candidates; never confer acceptance.

The learned mesh remains a shape prior. A metric voxelization removes occupied
cells whose centers project onto reference-image background. Other segmented
instances and out-of-frame regions are unknown, so they are retained. Remeshing
introduces voxel-scale discretization; this is not an exact measured surface.
"""

import copy
import shutil
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

from lidar_video.capture import load_capture, validate
from lidar_video.creation_audit import canonical_case, digest, inputs, read_json, save, verify_hashes
from lidar_video.object_evidence import assess_depth
from lidar_video.reconstruction import rigid


def carve_reference(
    mesh, labels, label, k, board_from_camera, *, pitch=0.001, max_cells=4_000_000, normal_field_path=None
):
    """Pure candidate construction: this API receives no other-view observations."""
    labels, k = np.asarray(labels), np.asarray(k, dtype=float)
    if labels.ndim != 2 or labels.dtype.kind not in "ui" or np.any(labels < 0):
        raise ValueError("Reference labels must be a nonnegative indexed image")
    if type(label) is not int or label < 1 or not np.any(labels == label):
        raise ValueError("Reference instance must have a nonempty positive label")
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("Invalid reference intrinsics")
    if not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0:
        raise ValueError("Reference intrinsics must use the capture's zero-skew camera convention")
    pose = rigid(board_from_camera, "board_from_camera")
    if not np.isfinite(pitch) or not 0.0005 <= pitch <= 0.002:
        raise ValueError("Voxel pitch must be between 0.5 and 2 mm")
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all() or not mesh.is_watertight:
        raise ValueError("Carving requires a finite, nonempty, watertight learned mesh")
    cells = np.ceil(mesh.extents / pitch) + 3
    if np.prod(cells) > max_cells:
        raise ValueError("Candidate voxel grid exceeds the bounded cell budget")
    voxel = mesh.voxelized(pitch=pitch).fill()
    points = voxel.points
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    front = camera[:, 2] > 1e-8
    xy = np.zeros((len(points), 2), dtype=np.int64)
    projected = camera[front, :2] / camera[front, 2:] * np.diag(k)[:2] + k[:2, 2]
    # Clip only for safe integer conversion; out-of-frame voxels stay unknown.
    xy[front] = np.rint(np.clip(projected, -1e9, 1e9)).astype(np.int64)
    inside = (
        front
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < labels.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < labels.shape[0])
    )
    allowed = cv2.dilate((labels == label).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    allowed |= (labels != 0) & (labels != label)
    keep = ~inside
    keep[inside] = allowed[xy[inside, 1], xy[inside, 0]]
    removed = int((~keep).sum())
    if not keep.any():
        raise ValueError("Reference carving removed the entire candidate")
    evidence = dict(
        method="reference_background_voxel_carving_v1",
        voxel_pitch_m=float(pitch),
        mask_margin_rgb_pixels=1,
        occupied_voxels_before=int(len(points)),
        occupied_voxels_after=int(keep.sum()),
        removed_voxels=removed,
        removed_fraction=float(removed / len(points)),
        other_instances="unknown; retained",
        out_of_frame="unknown; retained",
        scale_reapplied=False,
        accepted=False,
        limitations="learned completion plus voxel discretization; requires unchanged multiview validation",
    )
    matrix = np.zeros(voxel.shape, dtype=bool)
    indices = voxel.sparse_indices[keep]
    matrix[tuple(indices.T)] = True
    if removed == 0:
        result = mesh.copy()
    else:
        result = trimesh.voxel.ops.matrix_to_marching_cubes(matrix, pitch=pitch)
        result.apply_translation(voxel.transform[:3, 3])
    if not len(result.faces) or not result.is_watertight or not np.isfinite(result.vertices).all():
        raise ValueError("Carving did not produce a finite watertight candidate")
    if normal_field_path is not None:
        from lidar_video.visual_normals import save_normal_field

        evidence["normal_field"] = save_normal_field(normal_field_path, matrix, voxel.transform[:3, 3], pitch)
    return result, evidence


def refine_case(case_file, out, *, pitch=0.001, save_normal_field=False):
    """Write a new complete candidate manifest, preserving the measured inputs.

    Only each object's declared reference mask constructs its candidate. The
    separate audit/import stage must evaluate every original observation.
    """
    case = canonical_case(case_file)
    if case.get("geometry_scope") != "measured" or not all(
        case.get(key) for key in ("capture", "calibration", "mesh_manifest")
    ):
        raise ValueError("Silhouette refinement requires a complete measured case")
    source = Path(case["mesh_manifest"])
    manifest = read_json(source)
    if manifest.get("schema") != "roboskills-chess-meshes-v1" or set(manifest.get("objects", {})) != set(
        case["expected_objects"]
    ):
        raise ValueError("Manifest must contain exactly the declared object inventory")
    frozen = inputs(case)
    data = load_capture(case["capture"])
    validate(data)
    calibration = read_json(case["calibration"])
    board_from_world = rigid(calibration["board_from_world"], "board_from_world")
    out = Path(out).resolve()
    if case.get("scene_dir") and out.is_relative_to(Path(case["scene_dir"])):
        raise ValueError("Refinement output must be outside the frozen scene directory")
    if any(out == Path(p) or out.is_relative_to(Path(p)) for p in frozen):
        raise ValueError("Refinement output must not replace an input artifact")
    out.mkdir(parents=True, exist_ok=False)
    proposed = copy.deepcopy(manifest)
    records = {}
    for name, entry in manifest["objects"].items():
        if entry.get("validation") != "instance-multiview-v1" or not entry.get("static_scene"):
            raise ValueError("Refinement requires the static instance-multiview-v1 contract")
        frame = entry["frame"]
        if type(frame) is not int or not 0 <= frame < len(data["depth"]):
            raise ValueError("Invalid reference frame")
        observations = entry.get("observations", [])
        references = [o for o in observations if o["frame"] == frame]
        if len(references) != 1 or len({o["frame"] for o in observations}) < 2:
            raise ValueError("Declare one reference observation and at least one other view")
        observation = references[0]
        mask_path = (source.parent / observation["mask"]).resolve()
        labels = np.asarray(Image.open(mask_path))
        quality, _, _ = assess_depth(data, calibration, frame, labels == observation["label"], square=name)
        if not quality["passed"]:
            raise ValueError(f"Unreliable reference depth for {name}: {quality['reasons']}")
        mesh_path = (source.parent / entry["mesh"]).resolve()
        pose_path = (source.parent / entry["pose"]).resolve()
        camera_from_object = rigid(read_json(pose_path)["camera_from_object"], "camera_from_object")
        board_from_camera = board_from_world @ data["world_from_camera"][frame]
        board_from_object = board_from_camera @ camera_from_object
        mesh = trimesh.load(mesh_path, force="mesh", process=True)
        mesh.apply_transform(board_from_object)
        k = data["K"] if data["K"].ndim == 2 else data["K"][frame]
        folder = out / "objects" / name
        folder.mkdir(parents=True)
        candidate, evidence = carve_reference(
            mesh,
            labels,
            observation["label"],
            k,
            board_from_camera,
            pitch=pitch,
            normal_field_path=folder / "normal_field.npz" if save_normal_field else None,
        )
        candidate.apply_transform(np.linalg.inv(board_from_object))
        candidate.export(folder / "mesh.ply")
        shutil.copyfile(pose_path, folder / "pose.json")
        updated = proposed["objects"][name]
        updated.update(mesh=str(folder / "mesh.ply"), pose=str(folder / "pose.json"))
        updated["observations"] = [
            dict(o, mask=str((source.parent / o["mask"]).resolve())) for o in observations
        ]
        for key in ("mask_dir", "masks"):
            if updated.get(key):
                updated[key] = str((source.parent / updated[key]).resolve())
        records[name] = dict(
            evidence,
            construction_frames=[frame],
            reference_depth_quality=quality,
            source_mesh_sha256=digest(mesh_path),
            reference_mask_sha256=digest(mask_path),
            output_mesh_sha256=digest(folder / "mesh.ply"),
        )
    verify_hashes(frozen)
    save(out / "meshes.json", proposed)
    candidate_case = dict(case, mesh_manifest=str(out / "meshes.json"))
    candidate_case.pop("scene_dir", None)  # An old export does not represent these new candidates.
    save(out / "case.json", candidate_case)
    report = dict(
        status="candidate_created; not accepted",
        case=str(out / "case.json"),
        inputs=frozen,
        objects=records,
        next_step="Run audit-prepare on case.json; all existing gates and inventory still apply",
    )
    save(out / "refinement.json", report)
    return report
