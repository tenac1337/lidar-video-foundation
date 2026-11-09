# Bounded collision padding removal

Convex decomposition can extend collision vertices slightly below the unchanged visual surface. `trim-collision-padding` creates an opt-in scene variant that removes only this lower extension for explicitly named objects. It does not add a pedestal, move or rotate an object, alter visual geometry, redistribute mass, regenerate convex decomposition, or pin a free body. Defaults remain unchanged.

```sh
lidar-video trim-collision-padding scene/scene_graph.json \
  --objects piece_b8 --out runs/trimmed --support-tolerance 0.001
lidar-video scene-program-assets runs/trimmed/scene_graph.json \
  --out runs/trimmed-native --samples 1 --support-tolerance 0.001 --simulate --frames 121
```

The scene must pass the existing asset/envelope/constraint checks. Each selected object must have a visual mesh, separate convex collision assets, and one `on` relation to a horizontal box support. Its visual minimum height must already meet that support within the explicit tolerance. Collision files shared with another object or with visual geometry are rejected. Output must be a new directory outside the source scene.

For each collision solid, the command intersects it with `z >= visual_min_z` in the object's existing local frame. It retains vertices above the plane and intersects crossing edges, then takes their convex hull. These points are inside the original convex solid, so the operation removes collision volume; it does not add support geometry. Each output must remain a finite, consistently oriented, watertight convex solid. Parts cannot be deleted or flattened. The default depth budget is 0.75 mm (`--max-depth` in metres, maximum 1 mm), and total collision-part volume loss must not exceed 10%. The volume sum can double-count overlapping parts; it is a guard on this edit, not a physical volume estimate.

The graph's nodes, poses, dimensions, relations and collision-path inventory remain unchanged. Its existing envelopes still bound the trimmed assets. Every visual asset and unselected collision asset is copied byte-for-byte. Only modified collision OBJ files are rewritten. `collision_padding.json` records the source graph hash, clipping planes, per-part depths/volumes and preservation checks. The serialized output is revalidated before the new graph is written.

`--support-tolerance` retains the existing explicit allowance for reconstructed envelope overlap. It does not authorize larger trimming or fix floating/incorrect visual geometry. In particular, an object whose visual base cannot support its modeled center of mass can still tip after this correction. Compare native passive-motion reports, support contacts and replay checks before selecting a variant. A successful export is not stability certification, and correcting one object does not validate the rest of the scene.
