# Native scene-relation audit

Native simulation reports include `scene_relations`, independently of replay
consistency and passive-motion summaries. The audit examines every saved pose;
it can catch a temporary relation loss even if the final pose recovers.

`audit_scene_relations(model, saved, graph)` accepts native MuJoCo `time` and
`qpos` arrays and a graph whose physical IDs name native bodies. `size_m` is a
body-local conservative envelope centered at the body's origin. The audit rotates
these envelopes with the actual native orientations. It does not step, repair,
snap, pin or rewrite the trajectory.

- `on`: child envelope contained in the parent's XY footprint, with the declared
  top-face gap or support penetration, checked in the parent's frame. Separate
  bodies with zero clearance also require a direct native contact candidate.
- `inside`: envelope containment in the parent's frame. This does not establish
  that a solid collider is an appropriate hollow container.
- Directional edges: box-face inequalities in world XYZ, with the declared gap.
- Without an asset-derived spatial contract, semantic `occupies` and `part_of`
  edges are listed explicitly in `unchecked_edges`.

Native contact candidates establish geometric proximity, not supporting force,
force closure or task success. Rigidly connected bodies do not require a dynamic
contact pair; their geometric relation is still checked. A positive declared
clearance also intentionally omits the contact requirement. Each `on` edge
records its `contact_check` mode. Contact generation depends on native collision
masks and margins: a loose distance tolerance does not create missing contacts.

The default geometric and contact-distance tolerances are 1 mm. The default
starts at the first saved sample, with no automatic settling exclusion. Explicit
`start_time_s` can produce a separate settled report; retain the full report so
startup failures remain visible. All extrema and failures are over saved samples,
not continuous-time guarantees. Conservative rotated envelopes may flag valid
non-box geometry; inspect those failures without silently widening tolerances.

Each checked edge records its maximum geometric residual, missing-contact count,
violated sample count and first violation time. `all_checked_relations_preserved`
does not imply complete coverage; consult `complete_edge_coverage` and
`unchecked_edges`. Empty coverage is never reported as success. Mocap and
deformable models are rejected because pose-only saved state is insufficient.

`simulate(..., relation_graph=...)` may use explicit relation declarations from
the constructive compiler, but physical body IDs, kinds and envelope dimensions
must match the export. `scene-program-assets` supplies the same physical support
allowances used by its compiler and retains all semantic edges for coverage
reporting. Its per-sample verification includes the relation audit. The audit is
diagnostic: it does not turn the existing engine/replay check into a stability
certificate or automatically discard failed samples.

## Semantic regions and frozen separation branches

`prepare_spatial_contract(graph, asset_root, support_tolerance=...)` in
`lidar_video.spatial_contract` validates the reference through the asset adapter
and constructive compiler. Pass its result as `spatial_contract` to the native
audit or `simulate`. `scene-program-assets --simulate` prepares it automatically
and saves `spatial_contract.json` beside its other outputs.

The contract binds physical envelopes, source asset hashes, calibrated regions,
object-local footprint points and the compiler's original separation witnesses.
Prepare it from the original reference, not a failed simulated pose. Do not edit
contract geometry or switch separation axes to make a rollout pass.

- `occupies` transforms the original envelope corners or clipped support-band
  points with the object's native rotation and translation, then checks their
  projection within the assigned square in the parent's frame. Band points
  include triangle/plane intersections from visual and collision meshes. This
  rotates the original material band; it does not select a different band at
  each pose or claim to identify an actual contact patch.
- `part_of` checks the calibrated region's containment and top-plane definition,
  plus its declared fixed-support anchoring. These are calibrated definitions,
  not independently observed or simulated square bodies. The evidence field
  makes that distinction explicit.
- `separation_audit` checks each frozen world-axis/sign witness against rotated
  envelopes, preserving explicit support-penetration allowances. It reports
  first failure, count and maximum residual. Losing that branch is not proof
  of mesh collision: another axis may still separate the objects. The audit
  never silently substitutes such an axis.

The same explicit position tolerance applies to semantic containment and
separation residuals. Contact-only startup failures remain failures, even when
all semantic and separation checks pass. Placement-domain bounds are not
audited dynamically. Complete declared-edge coverage does not establish
independent reconstruction accuracy, whole-scene mesh collision freedom,
continuous-time correctness or task success.
