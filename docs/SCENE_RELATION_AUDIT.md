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
- Other edges, including semantic `occupies` and `part_of`, are listed explicitly
  in `unchecked_edges`. They are not silently counted as preserved.

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

Only explicit supported edges are checked. The compiler's implicit pairwise
separation witnesses and placement-domain bounds are not audited dynamically;
this report is not a whole-scene collision-freedom certificate.
