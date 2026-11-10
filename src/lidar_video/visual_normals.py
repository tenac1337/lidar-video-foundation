"""Opt-in voxel-field shading; never move vertices or alter collision geometry."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from lidar_video.reconstruction import rigid


def save_normal_field(path, matrix, origin, pitch):
    """Save the retained carving grid in its construction frame, without filtering."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, matrix=matrix, origin=origin, pitch_m=float(pitch))
    return dict(path=str(path.resolve()), sha256=_digest(path), frame="construction board coordinates")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def field_normals(vertices, matrix, origin, pitch, field_from_mesh, *, sigma=0.75, max_cells=4_000_000):
    """Interpolate outward Gaussian occupancy gradients; return mesh-frame normals.

    Filtering affects normals only. It neither extracts nor validates a surface.
    The transform maps mesh-local vertices into the original carving-grid frame.
    """
    matrix, origin, vertices = np.asarray(matrix), np.asarray(origin, float), np.asarray(vertices, float)
    transform = rigid(field_from_mesh, "field_from_mesh")
    if matrix.ndim != 3 or matrix.dtype != np.bool_ or not matrix.any():
        raise ValueError("Normal field must be a nonempty 3D boolean occupancy grid")
    if np.prod(np.array(matrix.shape, dtype=float) + 2) > max_cells:
        raise ValueError("Normal field exceeds padded cell budget")
    if origin.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("Normal field requires a finite origin and positive pitch")
    if not np.isfinite(sigma) or not 0 < sigma <= 2:
        raise ValueError("Normal smoothing sigma must be in (0, 2] voxels")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must be nonempty finite XYZ points")
    points = vertices @ transform[:3, :3].T + transform[:3, 3]
    coordinates = ((points - origin) / pitch + 1).T
    if np.any(coordinates < 0) or np.any(coordinates > (np.array(matrix.shape) + 1)[:, None]):
        raise ValueError("Mesh vertices fall outside the padded normal field; check the transform")
    smooth = gaussian_filter(np.pad(matrix.astype(float), 1), sigma, mode="constant", cval=0)
    normals = -np.array(
        [map_coordinates(g, coordinates, order=1, mode="constant", cval=0) for g in np.gradient(smooth)]
    ).T
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-8):
        raise ValueError("Undefined occupancy gradient at mesh vertices; refusing guessed normals")
    return (normals / lengths[:, None]) @ transform[:3, :3]


def _obj_geometry(text):
    """Read triangle OBJ without loader welding, normal splitting or reindexing."""
    vertices, faces = [], []
    for line in text.splitlines():
        tokens = line.split("#", 1)[0].split()
        if not tokens:
            continue
        if tokens[0] == "v":
            if faces or len(tokens) != 4:
                raise ValueError("Require XYZ-only vertices before all triangle faces")
            vertices.append([float(v) for v in tokens[1:]])
        elif tokens[0] == "f":
            if len(tokens) != 4:
                raise ValueError("Visual normal export requires triangle OBJ faces")
            indices = [int(v.split("/")[0]) for v in tokens[1:]]
            if any(i == 0 or not -len(vertices) <= i <= len(vertices) for i in indices):
                raise ValueError("Invalid OBJ vertex reference")
            faces.append([i - 1 if i > 0 else len(vertices) + i for i in indices])
    if not vertices or not faces or not np.isfinite(vertices).all():
        raise ValueError("OBJ requires finite vertices and triangle faces")
    return np.asarray(vertices), np.asarray(faces)


def export_visual_normals(mesh_path, field_path, field_from_mesh, out, *, sigma=0.75):
    """Write a new OBJ with only normal records/references changed, plus provenance."""
    mesh_path, field_path, out = Path(mesh_path), Path(field_path), Path(out)
    report_path = out.with_suffix(".normals.json")
    if out.exists() or report_path.exists():
        raise FileExistsError("Normal export must use new output and provenance paths")
    source = mesh_path.read_text()
    vertices, faces = _obj_geometry(source)
    source_hash, field_hash = _digest(mesh_path), _digest(field_path)
    with np.load(field_path, allow_pickle=False) as field:
        normals = field_normals(
            vertices, field["matrix"], field["origin"], float(field["pitch_m"]), field_from_mesh, sigma=sigma
        )
    lines = ["# Visual-only voxel gradient normals; original geometry retained."]
    lines.extend("vn " + " ".join(format(v, ".17g") for v in n) for n in normals)
    face_index = 0
    for line in source.splitlines():
        tokens = line.split("#", 1)[0].split()
        if tokens and tokens[0] == "vn":
            continue
        if tokens and tokens[0] == "f":
            corners = []
            for token, index in zip(tokens[1:], faces[face_index]):
                parts = token.split("/")
                if len(parts) > 3:
                    raise ValueError("Invalid OBJ face corner")
                uv = parts[1] if len(parts) > 1 else ""
                corners.append(f"{index + 1}/{uv}/{index + 1}")
            line = "f " + " ".join(corners)
            face_index += 1
        lines.append(line)
    result = "\n".join(lines) + "\n"
    exported_vertices, exported_faces = _obj_geometry(result)
    if not np.array_equal(vertices, exported_vertices) or not np.array_equal(faces, exported_faces):
        raise ValueError("Normal export changed geometry")
    if source_hash != _digest(mesh_path) or field_hash != _digest(field_path):
        raise ValueError("Source changed during normal export")
    report = dict(
        method="gaussian_occupancy_gradient_normals_v1",
        sigma_voxels=sigma,
        source_obj_sha256=source_hash,
        field_sha256=field_hash,
        field_from_mesh=np.asarray(field_from_mesh).tolist(),
        vertices=len(vertices),
        faces=len(faces),
        geometry_identical=True,
        scope="visual shading only; no geometry repair or acceptance",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        stream.write(result)
    report["output_obj_sha256"] = _digest(out)
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report
