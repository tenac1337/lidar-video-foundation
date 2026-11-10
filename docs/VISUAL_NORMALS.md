# Visual normals from retained carving fields

`visual-normals` writes a new triangle OBJ whose positions and connectivity are identical to the source, with normals derived from a Gaussian-filtered occupancy gradient. It changes shading only. It does not smooth geometry, repeat carving, refit poses, repair a mesh, or confer reconstruction acceptance. Defaults remain unchanged.

Save the retained field while constructing a candidate:

```sh
lidar-video complete-multiview case.json --plan plan.json --out runs/candidate --save-normal-field
# Or: lidar-video refine-silhouettes case.json --out runs/candidate --save-normal-field
```

Each selected object receives `normal_field.npz` containing its unchanged boolean occupancy matrix, grid origin and pitch in metres. The field uses the construction board frame, before subsequent alignment and object-local centering. Its hash is recorded in construction provenance. Saving the field does not alter mesh extraction, construction-view selection, validation roles or any quality gate.

After native mesh export and decimation, compute an explicit rigid transform from that final visual OBJ's local frame to the field's construction frame. Write the 4-by-4 matrix as a JSON array. Then:

```sh
lidar-video visual-normals scene/assets/piece/visual.obj \
  --field runs/candidate/objects/piece/normal_field.npz \
  --field-from-mesh field_from_mesh.json \
  --out runs/shading/visual.obj
```

If `scene_from_field` is the existing rigid board refinement and the local asset is centered at `position_m`, the required transform is `inverse(scene_from_field) @ translation(position_m)`. Include any additional rigid transforms actually applied by your pipeline. Do not use identity unless the two frames really coincide; scales and reflections are rejected. This command does not infer alignment.

The default Gaussian sigma is 0.75 voxels (`--sigma`, supported range greater than zero through 2). The field is padded by one empty cell, filtered with zero exterior occupancy, and differentiated. Trilinear interpolation supplies unit outward normals at existing vertices; normals are rotated back to the mesh frame. Undefined gradients, points outside the padded grid, invalid fields/transforms, and excessive grid sizes fail explicitly. Narrow structures can still shade poorly; this is an explicit visual option, not a universal surface estimator.

Outputs must be new. A sibling `.normals.json` records source/output hashes, field hash, transform, sigma and geometry identity. The exporter preserves XYZ vertex lines, triangle connectivity and texture coordinates, while replacing normal records and face normal references. It accepts XYZ-only triangle OBJ geometry with vertices before faces; polygon faces and colored/homogeneous vertex records are rejected.

Use the resulting file only as the visual mesh in a **new scene variant**, keeping original assets, collision paths, object poses, dimensions and physical parameters. Pass the OBJ directly to the native renderer after decimation: generic mesh load/re-export or simplification steps may recompute or discard supplied normals. Compare rendered views and native trajectories before selecting a variant. Improving shading does not improve geometric fit, physical stability, texture fidelity or task performance; all existing acceptance checks still apply.
