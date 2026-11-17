"""Remove cancelling zero-thickness triangle pairs from extracted voxel surfaces.

No smoothing, hole filling, component selection or vertex welding is performed.
Only exactly identical indexed triangles with opposite orientations can cancel.
The caller must still validate closure, geometry and observational agreement.
"""

import numpy as np
import trimesh


def cancel_opposed_faces(mesh):
    """Return a new mesh and a record of exact cancelling face pairs.

    Multiplicity other than two or a same-orientation duplicate is ambiguous and
    rejected. Face orientation is compared combinatorially, not via tolerances
    on normals or vertex positions. All original vertices are retained.
    """
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or faces.dtype.kind not in "iu"
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
        or np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0)
    ):
        raise ValueError("Finite indexed noncollapsed triangles required")
    _, inverse, counts = np.unique(np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True)
    duplicate_keys = np.flatnonzero(counts > 1)
    removed = []
    for key in duplicate_keys:
        ids = np.flatnonzero(inverse == key)
        if len(ids) != 2:
            raise ValueError("Ambiguous triangle multiplicity; refusing cancellation")
        a, b = faces[ids]
        # Reverse b, then test all cyclic rotations for equality to a.
        if not any(np.array_equal(a, np.roll(b[::-1], shift)) for shift in range(3)):
            raise ValueError("Same-orientation duplicate; refusing cancellation")
        removed.extend(ids.tolist())
    keep = np.ones(len(faces), bool)
    keep[removed] = False
    if not keep.any():
        raise ValueError("Cancellation would remove the entire surface")
    result = trimesh.Trimesh(vertices=vertices.copy(), faces=faces[keep].copy(), process=False)
    return result, dict(
        input_faces=len(faces),
        output_faces=len(result.faces),
        removed_opposed_pairs=len(removed) // 2,
        removed_face_indices=sorted(removed),
        vertices_preserved_exactly=True,
        oriented_triangle_sum_preserved=True,
        occupancy_modified=False,
        scope="Exact zero-thickness face cancellation only; no closed-surface or accuracy guarantee",
    )
