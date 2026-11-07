# Reference silhouette refinement

A learned mesh can fit sparse LiDAR returns while projecting onto background in the camera image. `refine-silhouettes` creates a new candidate by carving that unsupported geometry using only the object's declared reference mask. Other observations remain inputs to the existing multiview validator.

```sh
lidar-video refine-silhouettes measured-case.json --out runs/reference-carving
lidar-video audit-prepare runs/reference-carving/case.json --out runs/carving-audit
```

The first command creates candidates, **not accepted meshes**. The second evaluates the same depth, silhouette, association and square-bound gates. A missing scene continues to block downstream audits. Every expected object must remain in the manifest; an individual improvement cannot approve an incomplete scene.

## Construction contract

- Use only the instance's original reference frame, its indexed mask, calibrated camera and learned mesh. Do not construct candidates from other-view scores or masks.
- Voxelize a finite watertight prior at a fixed metric pitch (default 1 mm), with a bounded cell budget. Project occupied cell centers into the reference image.
- Preserve the target mask with a one-RGB-pixel margin. Other segmented instances and out-of-frame cells are unknown and remain occupied; known background removes cells.
- Remesh retained cells. Voxelization is a discretized completion hypothesis, not an exact measured surface; it can change boundaries at voxel resolution. No pose scale is reapplied, and no dimensions are resized to satisfy a gate.
- Write new mesh/pose files and a complete candidate manifest. Preserve capture, calibration, reference selection, observations and expected inventory. Clear any old scene export path because it no longer describes the candidates.

`refinement.json` records construction frames, input/output hashes, retained voxel counts, resolution and limitations. The output directory must be new. Original inputs remain unchanged. No-op carving preserves the original geometry; complete removal or an invalid prior fails instead of producing an empty accepted object.

The command currently supports the development-only `creation-audit-case-v1` measured case with `instance-multiview-v1` objects. The installed package includes scikit-image for marching cubes. Larger or non-watertight learned priors must be repaired or regenerated before using this operation.

## Validation and limitations

Tests exercise actual voxelization/remeshing, occlusion preservation, unknown image regions, empty-result rejection, memory bounds and file immutability. A paired integration test changes every non-reference mask and its depth: candidate bytes remain identical, while the downstream validator rejects the changed evidence. Construction therefore cannot use those views to hide a validation failure.

This operation can remove unsupported completion; it cannot recover a missing feature or resolve a wrong learned shape. A rejected candidate remains rejected. Repeatedly inspected views are development evidence, even when the importer labels them `held_out` relative to the reference fit. A new recording or reserved evaluation set is needed for an independent accuracy claim.

## Recorded development result

On the existing two-object iPhone development case, 1 mm reference-only carving brings the f8 candidate through the unchanged depth, silhouette and assigned-square bounds checks. Its other-view silhouette IoU is 0.544262 (minimum 0.500), with median surface distance 4.285 mm and p90 16.607 mm.

The b8 candidate remains rejected: its failing-view IoU is 0.433846. The complete two-object case therefore has a failed numerical geometry audit; physics and export audits remain blocked without an accepted scene. These are numerical development results, not fresh independent reviewer approval or a final accuracy evaluation. The command does not remove b8 from the required inventory or promote f8 into a complete scene.

Validation: 161 tests pass, including the existing audit and importer regression suite; Ruff and whitespace checks pass. scikit-image emits NumPy 2.5 deprecation warnings during marching cubes. Private captures, masks, generated meshes and audit evidence are kept outside the source repository.
