# Multi-view geometry proposals

Run `python -m lidar_video.multiview_shape_runner manifest.json output-directory`
with the package installed. The output directory must be new. This is an explicit
experimental stage; it does not automatically overwrite native reconstruction assets.

The manifest contains `incumbent_mesh` (triangle mesh path), `support_from_world`
(rigid 4×4 transform into a support frame whose z=0 is the proposed contact plane),
`static_scene: true`, `construction_sources`, `validation_sources`, and `views`.
Each view specifies an indexed PNG `labels` path, integer positive target `label`,
integer `source_frame`, 3×3 `K` and 4×4 OpenCV `world_from_camera`. Paths are relative
to the manifest. All mesh vertices and translations use metres in the same world
frame. Optional `policy` fields correspond to `VolumePolicy`; optional
`selection_policy` provides `min_iou` and `max_regression`.

Freeze the split before producing proposals. Supply at least two construction
camera centers separated by 1 cm. A separate validation center is required for a
replacement to pass. The reserved camera may veto the construction-selected
candidate once; it cannot select another candidate after failure. Subsequent
method development on inspected views makes those views development data.

The bounded silhouette volume can add shape absent from a learned mesh. It uses
calibrated masks, a supplied support plane and a declared height cap. It abstains
on partial framing, unresolved bound contact, excessive voxel budgets, or
insufficient camera baseline. Other-instance pixels are unknown, not positive
object evidence. Disconnected components are retained and reported. Invisible
concavities and texture cannot be recovered by this method.

A passing result establishes only the configured per-view silhouette screen.
It is not proof of instance identity, measured surface accuracy, stable support,
collision-free placement, native simulation validity, scene completeness, an
independent-scene result, or an overall qualitative win. Full-scene replacement
requires those separate checks and an unambiguous one-to-one instance mapping.

Set `proposal_method: "support_pose"` to preserve the incumbent shape and scale
while searching only XY translation and yaw in the support frame. The default
bounds are ±8 mm on each horizontal axis and ±15 degrees of yaw. The fixed
coarse-to-fine search uses construction masks only, maximizes the worst per-view
improvement first, and retains contact height. Its `policy` accepts
`max_translation_m` and `max_yaw_degrees`. The same reserved-view veto applies;
footprint, collisions, dynamics and changes in occlusion remain separate checks.

## Explicit geometry priors

`proposal_method: "support_shape"` adds bounded width and height refinement to the
XY/yaw search. Width scales both support-horizontal axes equally; height scales
about the original mesh's lowest support-frame vertex, preserving its base height.
The optional `max_scale_fraction` defaults to 0.10 and cannot exceed 0.15. Camera
calibration and global metric scale are unchanged. This refines a supplied prior;
it does not recover unseen geometry from scratch. The returned
`world_from_original_world` is an affine transform, unlike the rigid transform
returned by pose-only refinement. A separate-view veto and full-scene checks are
still required. Rebuild native collision assets before using changed geometry in
simulation; a passing visual proposal is not a physics-ready replacement.

The Python `silhouette_volume` API optionally accepts `connected_landmarks`: four
or more distinct triangulated construction-only world points. This adds an
explicit single-connected-object assumption. All points must identify the same
unique six-connected voxel component within 3 mm. Component size never selects
the result. Ambiguous or inconsistent anchors cause abstention; discarded cells
and the prior are recorded. The selected component must still pass all bound,
finite mesh, watertightness and subsequent silhouette checks. Do not supply
landmarks derived from the reserved veto view.
