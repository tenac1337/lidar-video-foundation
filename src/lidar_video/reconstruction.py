"""Metric chess reconstruction with one graph driving meshes, physics and Scenic.

Board calibration is supplied or detected. Static fusion is opt-in. Without learned meshes,
visible RGB-D points plus a declared resting base form convex completion proxies.
SAM 3D mesh.ply already includes its predicted scale: never apply pose.json scale twice.
"""

import json
from pathlib import Path

import chess
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from lidar_video.capture import REQUIRED, load_capture, validate, write_json
from lidar_video.chess_recognition import recognize
from lidar_video.chess_scene import board_from_observation
from lidar_video.native_depth import sensor_camera_points
from lidar_video.pipeline import UPSTREAM_COMMIT, prepare_upstream
from lidar_video.scenic_export import export_scenic, sample_scenic


class MeshQualityError(ValueError):
    """Valid mesh data disagrees with measured geometry or support constraints."""

    def __init__(self, message, *, evidence=None):
        super().__init__(message)
        self.evidence = evidence


def check_mesh_bounds(mesh, lower, upper):
    extents = np.asarray(mesh.extents)
    if min(extents) < 0.002 or max(extents) > 0.2:
        raise MeshQualityError(f"Implausible reconstructed dimensions: {extents}")
    if np.any(mesh.bounds[0, :2] < lower) or np.any(mesh.bounds[1, :2] > upper):
        raise MeshQualityError("Geometry crosses square boundary; inspect calibration/segmentation")


def rigid(value, name):
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError(f"{name} must be a finite proper rigid transform")
    return matrix


def calibrated_points(data, calibration, frame):
    transform = rigid(calibration["board_from_world"], "board_from_world")
    camera = sensor_camera_points(data, frame)
    pose = transform @ data["world_from_camera"][frame]
    return camera @ pose[:3, :3].T + pose[:3, 3]


def collect_squares(data, calibration, frames, masks=None, min_height=0.006):
    """Fuse only declared stationary views; mask 0 excludes background when provided."""
    size = float(calibration["square_size_m"])
    gap = float(calibration.get("center_gap_m", 0))
    if not np.isfinite(gap) or not 0 <= gap <= size:
        raise ValueError("center_gap_m must be between zero and one square width")
    if not np.isfinite(size) or not 0.01 <= size <= 0.2:
        raise ValueError("square_size_m must be between 0.01 and 0.2 metres")
    if not np.isfinite(min_height) or not 0 < min_height < 0.2:
        raise ValueError("min_height must be between zero and 0.2 metres")
    collected = {s: [] for s in chess.SQUARES}
    colors = {s: [] for s in chess.SQUARES}
    observations = {s: [] for s in chess.SQUARES}
    assignments = None
    if masks is not None and (Path(masks) / "instances.json").exists():
        metadata = json.loads((Path(masks) / "instances.json").read_text())
        if metadata.get("schema") != "chess-instance-assignments-v1" or not isinstance(
            metadata.get("frames"), dict
        ):
            raise ValueError("Invalid instance assignment manifest")
        assignments = metadata["frames"]
    for frame in frames:
        points = calibrated_points(data, calibration, frame)
        z = data["depth"][frame]
        good = np.isfinite(z) & (z > 0) & (data["confidence"][frame] >= 1)
        good &= (points[..., 2] > min_height) & (points[..., 2] < 0.2)
        board_xy = points[..., :2].copy()
        good &= np.abs(board_xy[..., 1]) >= gap / 2
        board_xy[..., 1] -= np.sign(board_xy[..., 1]) * gap / 2
        uv = board_xy / size + 4
        good &= np.all((uv >= 0) & (uv < 8), axis=-1)
        if masks is not None:
            labels = np.asarray(Image.open(Path(masks) / f"{frame:05d}.png"))
            if labels.shape != z.shape or labels.dtype.kind not in "ui":
                raise ValueError("Masks must be indexed PNGs registered to the original RGB-D pixels")
            good &= labels > 0
        grid = np.floor(uv[good]).astype(int)
        squares = grid[:, 0] + 8 * grid[:, 1]
        if assignments is not None:
            mapping = assignments.get(str(frame))
            if not isinstance(mapping, dict) or any(
                name not in chess.SQUARE_NAMES for name in mapping.values()
            ):
                raise ValueError("Invalid frame instance assignments")
            if any(str(label) not in mapping for label in np.unique(labels[labels > 0])):
                raise ValueError("Every nonzero mask label requires an instance assignment")
            # Preserve segmented object identity. Cross-square points must remain
            # evidence against that object, never become an invented neighbor.
            squares = np.array([chess.parse_square(mapping[str(label)]) for label in labels[good]], dtype=int)
        xyz, rgb = points[good], data["rgb"][frame][good]
        for square in np.unique(squares):
            selected = squares == square
            collected[int(square)].extend(xyz[selected])
            colors[int(square)].extend(rgb[selected])
            observations[int(square)].append(dict(frame=int(frame), points=int(selected.sum())))
    return collected, colors, observations


def completed_hull(points):
    points = np.asarray(points)
    lo, hi = np.quantile(points, [0.01, 0.99], axis=0)
    clean = points[np.all((points >= lo) & (points <= hi), axis=1)]
    # A hidden base on z=0 is an explicit physical hypothesis, not observed geometry.
    base = np.array([[x, y, 0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
    return trimesh.convex.convex_hull(np.vstack([clean, base]))


def import_learned_mesh(entry, root, data, calibration, observed):
    frame = entry.get("frame", 0)
    if type(frame) is not int or not 0 <= frame < len(data["depth"]):
        raise ValueError("Mesh reference frame is outside the capture")
    source = (root / entry["mesh"]).resolve()
    pose = json.loads((root / entry["pose"]).read_text())
    camera_from_object = rigid(pose["camera_from_object"], "camera_from_object")
    mesh = trimesh.load(source, force="mesh", process=True)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("Learned mesh is empty or non-finite")
    transform = rigid(calibration["board_from_world"], "board_from_world")
    mesh.apply_transform(transform @ data["world_from_camera"][frame] @ camera_from_object)
    if "validation" in entry:
        if entry["validation"] not in ("instance-multiview-v1", "instance-multiview-v2"):
            raise ValueError("Unknown learned-mesh validation contract")
        from lidar_video.mesh_alignment import validate_and_align

        mesh, evidence = validate_and_align(mesh, entry, root, data, calibration)
        if entry.get("completion"):
            evidence["completion"] = entry["completion"]
        return mesh, dict(evidence, source=str(source))
    # Keep input scale; reject gross registration errors instead of hiding them with rescaling.
    probes, _ = trimesh.sample.sample_surface(mesh, 10000, seed=0)
    error = float(np.median(cKDTree(probes).query(np.asarray(observed))[0]))
    tolerance = float(calibration["square_size_m"]) * 0.25
    if error > tolerance:
        raise MeshQualityError(f"Mesh fails measured-depth agreement: {error:.4f}m > {tolerance:.4f}m")
    if abs(float(mesh.bounds[0, 2])) > 0.01:
        raise MeshQualityError(
            f"Learned mesh base z={mesh.bounds[0, 2]:.6f}m is more than 10 mm from the supplied board plane"
        )
    # Record this explicit support alignment separately from metric reconstruction.
    offset = -float(mesh.bounds[0, 2])
    mesh.apply_translation([0, 0, offset])
    return mesh, dict(
        source=str(source),
        validation="legacy_square_cloud",
        reference_frame=frame,
        depth_residual_m=error,
        support_adjustment_m=offset,
        scale_reapplied=False,
    )


def reconstruct(
    capture,
    calibration,
    out,
    *,
    masks=None,
    mesh_manifest=None,
    mesh_policy="strict",
    model=None,
    context=None,
    static_scene=False,
    fusion_frames=None,
    min_height=0.006,
    reference_frame=0,
    samples=20,
    seed=7,
    simulation=True,
):
    if mesh_policy not in ("strict", "fallback"):
        raise ValueError("mesh_policy must be strict or fallback")
    data = load_capture(capture)
    validate(data)
    if type(reference_frame) is not int or not 0 <= reference_frame < len(data["depth"]):
        raise ValueError("Reference frame outside capture")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if calibration is None:
        if mesh_manifest is not None:
            raise ValueError("Supply the matching saved calibration when importing an external mesh manifest")
        from lidar_video.board_calibration import calibrate

        calibration = calibrate(data, out / "calibration")
    else:
        calibration = json.loads(Path(calibration).read_text())
    rigid(calibration["board_from_world"], "board_from_world")
    orientation = calibration.get("orientation", dict(resolved=True, source="supplied board axes"))
    if context is not None and not orientation["resolved"]:
        raise ValueError("Resolve the physical a1 orientation before creating chess game state")
    frames = list(range(reference_frame, len(data["depth"]))) if static_scene else [reference_frame]
    if fusion_frames is not None:
        if (
            not static_scene
            or not fusion_frames
            or any(type(f) is not int or not 0 <= f < len(data["depth"]) for f in fusion_frames)
            or len(set(fusion_frames)) != len(fusion_frames)
        ):
            raise ValueError("fusion_frames requires a static scene and unique valid frame indices")
        frames = list(fusion_frames)
    points, colors, observations = collect_squares(data, calibration, frames, masks, min_height)
    entries, manifest_root = {}, None
    if mesh_manifest:
        manifest = json.loads(Path(mesh_manifest).read_text())
        if manifest.get("schema") != "roboskills-chess-meshes-v1":
            raise ValueError("Unsupported mesh manifest schema")
        entries = manifest["objects"]
        if not isinstance(entries, dict) or any(not isinstance(v, dict) for v in entries.values()):
            raise ValueError("Mesh manifest objects must map square names to mesh/pose records")
        manifest_root = Path(mesh_manifest).resolve().parent
        if any(s not in chess.SQUARE_NAMES for s in entries):
            raise ValueError("Mesh manifest keys must be chess square names")
    size = float(calibration["square_size_m"])
    gap = float(calibration.get("center_gap_m", 0))
    if model is not None and gap:
        raise ValueError("The trained recognizer does not support boards with a center gap")
    nodes = [
        dict(
            id="board",
            kind="support",
            position_m=[0, 0, -0.0075],
            size_m=[8 * size, 8 * size + gap, 0.015],
            color_rgb=[0.45, 0.35, 0.22],
            checkerboard=True,
            square_size_m=size,
            center_gap_m=gap,
            evidence=calibration.get("evidence", "supplied board calibration"),
        )
    ]
    edges = []
    selected = {k: data[k] if k == "K" and data[k].ndim == 2 else data[k][frames] for k in REQUIRED}
    for key in ("native_K", "native_depth_shape", "depth_sample_id"):
        if key in data:
            value = data[key]
            selected[key] = (
                value[frames]
                if key == "depth_sample_id" or (key == "native_K" and value.ndim == 3)
                else value
            )
    recognition = recognize(selected, calibration, model) if model else None
    if recognition:
        write_json(out / "recognition.json", recognition)
    accepted, rejected = [], {}
    for square in chess.SQUARES:
        name = chess.square_name(square)
        center = (np.array([chess.square_file(square), chess.square_rank(square)]) - 3.5) * size
        center[1] += np.sign(center[1]) * gap / 2
        nodes.append(dict(id=name, kind="square", position_m=[*center, 0]))
        edges.append(dict(source=name, relation="part_of", target="board", evidence="calibration"))
        if len(points[square]) < 24:
            if name in entries:
                raise ValueError(f"No usable measured support for mesh at {name}")
            continue
        geometry = "measured visible points with convex hidden-base completion"
        evidence = dict(
            observations=observations[square],
            masks="SAM-compatible indexed masks"
            if masks
            else "height above calibrated board; one piece per square assumed",
        )
        lower, upper = center - size / 2, center + size / 2
        mesh = None
        if name in entries:
            try:
                mesh, learned = import_learned_mesh(
                    dict(entries[name], square=name), manifest_root, data, calibration, points[square]
                )
                check_mesh_bounds(mesh, lower, upper)
            except MeshQualityError as error:
                if mesh_policy == "strict":
                    raise MeshQualityError(f"{name}: {error}", evidence=error.evidence) from error
                rejected[name] = str(error)
                evidence["rejected_learned_mesh"] = dict(
                    entry=entries[name], reason=str(error), alignment=error.evidence
                )
                mesh = None
            else:
                accepted.append(name)
                geometry = "imported SAM 3D mesh; CoACD collision parts"
                if entries[name].get("completion"):
                    geometry = "SAM 3D prior with multiview silhouette and physical-prior completion; CoACD collision parts"
                evidence["learned_mesh"] = learned
        if mesh is None:
            mesh = completed_hull(points[square])
        check_mesh_bounds(mesh, lower, upper)
        extents = np.asarray(mesh.extents)
        position = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-position)
        folder = out / "assets" / f"piece_{name}"
        folder.mkdir(parents=True)
        mesh.export(folder / "mesh.ply")
        if name in accepted:
            from roboskills.tools.coacd.coacd import decompose

            decompose(folder / "mesh.ply", folder, 0.05, 20000, 32)
            collision = json.loads((folder / "parts.json").read_text())["parts"]
        else:
            mesh.export(folder / "visual.obj")
            collision = ["visual.obj"]  # This geometry is already convex.
        # CoACD can slightly expand the source surface; Scenic must contain the
        # actual collision parts as well as the decimated visual mesh.
        surface_extents = extents.copy()
        asset_bounds = [mesh.bounds]
        for asset in {"visual.obj", *collision}:
            asset_bounds.append(trimesh.load(folder / asset, force="mesh").bounds)
        extents = 2 * np.max(np.abs(np.asarray(asset_bounds)), axis=(0, 1))
        if np.any(position[:2] - extents[:2] / 2 < lower) or np.any(position[:2] + extents[:2] / 2 > upper):
            raise MeshQualityError(f"Collision envelope crosses square boundary at {name}")
        symbol = recognition["symbols"][square] if recognition else "?"
        if symbol == ".":
            symbol = "?"  # Occupied geometry contradicts recognizer: never drop the object.
        node = dict(
            id=f"piece_{name}",
            kind="object",
            square=name,
            symbol=symbol,
            semantic_type=chess.piece_name(chess.Piece.from_symbol(symbol).piece_type)
            if symbol not in "?."
            else "unknown_chess_piece",
            position_m=position.tolist(),
            size_m=extents.tolist(),
            surface_size_m=surface_extents.tolist(),
            square_bounds_m=[lower.tolist(), upper.tolist()],
            color_rgb=(np.median(colors[square], axis=0) / 255).tolist(),
            geometry=geometry,
            evidence=evidence,
            mass_kg=0.08,
            mesh_path=str((folder / "visual.obj").relative_to(out)),
            collision_paths=[str((folder / p).relative_to(out)) for p in collision],
        )
        if recognition:
            node["recognition_vote_fraction"] = recognition["vote_fraction"][square]
        nodes.append(node)
        edges.extend(
            [
                dict(source=node["id"], relation="occupies", target=name, evidence="calibrated geometry"),
                dict(source=node["id"], relation="on", target="board", evidence="resting hypothesis"),
            ]
        )
    objects = [n for n in nodes if n["kind"] == "object"]
    if not objects:
        raise ValueError("No separated chess objects reconstructed")
    graph = dict(
        schema_version=2,
        source="metric RGB-D chess reconstruction",
        frame=reference_frame,
        time_s=float(data["rgb_time"][reference_frame]),
        tracks=[],
        nodes=nodes,
        edges=edges,
        scene_from_world=calibration["board_from_world"],
        units="metres",
        axes="x assigned h-file, y assigned rank 8, z up",
        board_orientation=orientation,
        center_gap_m=gap,
        assumptions=dict(
            static_scene=static_scene,
            fused_frames=frames,
            foreground_min_height_m=min_height,
            one_piece_per_square=True,
            board_pose=calibration.get("evidence", "supplied"),
            mass_kg=0.08,
            friction=[0.8, 0.005, 0.0001],
        ),
    )
    if context is not None:
        if recognition is None:
            raise ValueError("Game context requires a recognition model")
        occupied = {chess.parse_square(n["square"]) for n in objects}
        predicted = {s for s, v in enumerate(recognition["symbols"]) if v != "."}
        if occupied != predicted:
            raise ValueError(
                "Recognition and geometric occupancy disagree; review before creating game state"
            )
        board = board_from_observation(recognition, json.loads(Path(context).read_text()))
        graph["chess"] = dict(fen=board.fen(en_passant="fen"), context_source="explicit supplied history")
    write_json(out / "scene_graph.json", graph)
    write_json(out / "board_calibration.json", calibration)
    # Normalize only the separate GPU handoff; masks above always index original pixels.
    from lidar_video.capture import normalize

    prepare_upstream(normalize(data), out / "upstream_workspace", reference_frame=reference_frame)
    export_scenic(graph, out / "scenario.scenic", jitter=0.002)
    variants = sample_scenic(out / "scenario.scenic", graph, out / "scenic_samples.json", samples, seed)
    physics = None
    if simulation:
        from lidar_video.simulation import simulate

        report, replay = simulate(graph, out / "simulation", frames=30, asset_root=out)
        simulate(variants[0], out / "sample_simulation", frames=30, asset_root=out)
        physics = dict(replay, warnings=report["warnings"], max_penetration_m=report["max_penetration_m"])
    result = dict(
        status="passed",
        objects=len(objects),
        imported_learned_meshes=len(accepted),
        geometric_completions=len(objects) - len(accepted),
        mesh_policy=mesh_policy,
        rejected_learned_meshes=rejected,
        scenic_samples=len(variants),
        physics=physics,
        upstream_commit=UPSTREAM_COMMIT,
        contract="one scene graph supplies simulation geometry, IDs, dimensions and Scenic constraints",
        limits=[
            "real iPhone capture processed; absolute accuracy unverified"
            if str(data.get("source_kind", "")).startswith("real_iphone")
            else "real iPhone validation pending",
            "board orientation unresolved" if not orientation["resolved"] else "physical board axes supplied",
            "hidden geometry inferred",
            "synthetic recognizer not validated on real chess sets",
            "no robot manipulation",
        ],
    )
    write_json(out / "verification.json", result)
    return result
