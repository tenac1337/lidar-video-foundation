"""Allocate an object's mass across its convex collision components.

The uniform-density prior is over the compound collision representation, so
intersecting parts count overlapping volume more than once. This is not a mass
estimate from RGB/depth, and is not an exact density model for hollow objects.
"""

from pathlib import Path

import numpy as np
import trimesh


def allocate_collision_mass(paths, total_mass):
    """Use actual convex-envelope volumes rather than one equal mass per file."""
    if (
        isinstance(total_mass, bool)
        or not np.isscalar(total_mass)
        or not np.isfinite(total_mass)
        or total_mass <= 0
    ):
        raise ValueError("Positive finite total mass required")
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("At least one collision component required")
    volumes = []
    for path in paths:
        mesh = trimesh.load(path, force="mesh", process=False)
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError("Finite collision mesh required")
        try:
            # MuJoCo uses convex envelopes for mesh collisions. Raw OBJ triangle
            # volume can be wrong for open or oppositely wound source surfaces.
            volume = float(mesh.convex_hull.volume)
        except (ValueError, RuntimeError) as error:
            raise ValueError("Full-dimensional collision component required") from error
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError("Positive convex component volume required")
        volumes.append(volume)
    volumes = np.asarray(volumes)
    fractions = volumes / np.max(volumes)
    fractions /= fractions.sum()
    masses = float(total_mass) * fractions
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("Invalid component mass allocation")
    return masses, dict(
        method="uniform_collision_volume",
        total_mass_kg=float(total_mass),
        convex_volumes_m3=volumes.tolist(),
        component_masses_kg=masses.tolist(),
        overlap_accounting="Overlaps count in each component; no union-volume claim",
        density_measured=False,
    )
