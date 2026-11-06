# Multiview constrained completion

A learned prior can match visible depth yet contain below-board or overly broad hidden geometry. Translating its lowest vertex onto the board moves already plausible surfaces. `complete-multiview` instead removes below-board completion and intersects the occupied volume with explicitly selected construction silhouettes and an assigned-square footprint.

This is a new **development-only construction contract**, not a threshold relaxation or an independent sensor accuracy result. The footprint and board plane are imposed physical priors. Checking them afterward verifies compliance with those priors; it is not independent measurement. The method cannot represent a piece whose real geometry overhangs its assigned square.

```json
{
  "schema": "multiview-completion-plan-v1",
  "objects": {
    "b8": {
      "construction_frames": [0, 7],
      "validation_frames": [3],
      "physical_priors": ["board_plane", "assigned_square"]
    }
  }
}
```

```sh
lidar-video complete-multiview runs/original/case.json \
  --plan plan.json --out runs/multiview
lidar-video audit-prepare runs/multiview/case.json --out runs/multiview-audit
```

Start from an original static `instance-multiview-v1` candidate, not a mesh already fitted or carved using the proposed reserved views. A manifest cannot prove an upstream model's training provenance: the caller must establish this source history. The output keeps the complete object inventory and all original observations; untouched objects retain their original contract. It clears stale scene exports.

Selected entries become `instance-multiview-v2`. Their construction and validation lists must exactly partition observations, include the original reference in construction, contain at least two construction views and at least one reserved view, and have no duplicate source viewpoints between construction and validation. V1 rejects explicit split metadata rather than silently mislabeling construction data as held out. A repair-loop builder may not change this metadata; an intentional split change starts a new development case.

The constructor checks trusted native depth in construction views, then uses only their indexed silhouettes, camera intrinsics and poses to carve the original metric volume. It uses 1 mm voxels by default, a one-RGB-pixel silhouette margin, and a half-voxel footprint inset. Other instances and out-of-frame projections remain unknown. The complete learned shape is neither resized nor translated during carving. No `pose.scale` is reapplied. Empty/nonwatertight results, invalid intrinsics and excessive voxel allocation are rejected.

Before import, no candidate is accepted. The unchanged depth, silhouette, support, dimensions, square bounds and association checks evaluate every observation. If pose refinement is needed, V2 fits only construction clouds; reserved clouds only validate. The original per-view regression guards remain active. Collision envelopes must also fit the square. Physics and export audits are still required after scene generation.

The output manifest and `construction.json` record source/candidate hashes, the plan, physical priors, occupied-volume changes and the frozen inputs. Scene graphs retain this completion provenance. Hidden surfaces remain inferred, even when checks pass.

## Current development evidence

The fresh frame-0 SAM 3D prior was selected using upstream reference-only ranking, then rejected by the original V1 checks. Two-view carving with frames 0/3 (about 14 degrees apart) also failed. A separate case uses the wider 0/7 pair (about 69 degrees apart), the original prior, and frame 3 excluded from construction. Explicitly imposing the existing board/square priors gives b8 silhouette IoUs of approximately 0.796, 0.661 and 0.511 in frames 0, 3 and 7 after support correction. Its reserved frame-3 median/p90 surface distances are 5.4/13.2 mm. The fixed f8 candidate also passes geometry.

All these views were inspected in earlier development experiments. Changing the construction split does **not** restore an untouched test set, and the 0.511 construction-view result has little margin above 0.50. Physical board orientation, absolute sensor accuracy, full-board reconstruction, chess-piece identity and robot manipulation remain unverified. Private captures, generated meshes and per-run evidence stay outside Git.
