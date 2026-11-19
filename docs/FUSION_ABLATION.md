# Geometry modality ablations

`stereo_surface` now requires only RGB images, pinhole intrinsics, source frame
IDs and rigid camera poses. Depth/confidence fields are neither required nor
consulted. Metric scale still comes from supplied poses: RGB-derived geometry
with device poses is not camera-only localization or a sensor-independent scale
estimate. Keep that distinction explicit when comparing modalities.

Stereo retains left/right disparity checks, texture and uniqueness thresholds,
source-image bounds, metric baseline and field-of-view checks. Vertical
rectification rotates both images and camera coordinate systems by 90 degrees,
then restores source pixel coordinates and depth grids. World points remain in
the original world frame. Returned disparity/rectified-valid arrays stay in the
rotated rectified grid, as labelled in metadata. `rectification_alpha` defaults to
1; callers can explicitly freeze a value in [0, 1]. A change in rectification or
view selection is a new development condition, not a silent retry.

`fusion_gate.gate_depths(lidar, rgb, tolerance_m=0.015)` compares two registered
camera-axis-Z grids. It returns separate keep masks and agreement, conflict,
LiDAR-only and RGB-only masks. It never averages depth or changes inputs. When
both values exist and differ beyond the tolerance, both abstain. A single known
modality remains explicitly uncorroborated. Zero/NaN means unknown, not free space.

The caller must validate sensor confidence, RGB correspondence, native sample
identity, coordinate registration and occlusion assumptions. A nearby native ray
can intersect a different surface at a boundary; registration does not eliminate
that ambiguity. Repeated RGB-grid copies of a native depth pixel are not new
sensor measurements. Preserve native rays when constructing LiDAR point clouds.

For a controlled three-way comparison, retain LiDAR-derived, RGB-derived and
gated geometry separately. Share cameras, view-selection rules, range limits,
representation and rendering settings; retain every failed view and missing
surface. Record all settings before the run. Confidence or correspondence gates
must not be relaxed after seeing a preferred output. Report sensor-only and
model-assisted comparisons separately: a sparse stereo surface is not a test of
every learned RGB reconstruction method.

Per-view gating is not global geometric consistency. Pose drift and conflicting
observations in other views can remain in an accumulated cloud. Keeping
uncorroborated samples also means the union is not wholly cross-modally verified.
Point clouds/open surfaces are not complete meshes or simulation-ready assets.
Coverage counts, successful coordinate tests and a gate rejecting conflicts do
not establish reconstruction accuracy, qualitative superiority or generalization.

## Automatic stereo partners

`stereo_pairs.select_stereo_neighbors` takes calibrated poses, intrinsics, image
size, an anchor index, explicit candidate indices and excluded review indices.
It accepts no RGB or depth data. It ranks candidates by a rectification footprint
proxy and proximity to a target baseline, then selects at most two partners with
minimum spatial separation. The result includes every candidate's eligibility,
rejection reason, geometry and score. Incomplete selection is explicit; the API
never silently substitutes a neighbor or retries based on reconstruction output.

The footprint is projected coverage of a regular rectified ray grid in the
original images. It is not measured scene overlap and cannot resolve occlusion,
texture, motion or inaccurate poses. Horizontal and vertical stereo share the
same policy. Fix candidate windows, thresholds, exclusions and any separately
labelled fallback before running reconstruction. Changing partners also changes
LiDAR corroboration if both modalities share those views; interpret that as a
whole view-selection policy comparison, not an isolated RGB algorithm change.

## Metric anchoring and cross-view checks

`metric_fusion.fit_metric_scale` calibrates one bounded multiplicative scale from
registered native rays. The caller supplies a support mask and must establish
confidence, view consistency and registration before fitting. Sixteen spatial
tiles balance robust log-ratio estimates. Alternating 8x8 native pixel blocks
separate fitting from checking; validation never refits the scale. Insufficient
support, an out-of-bounds scale or failed residual checks return `scale=None`.
`candidate_scale` on a rejected result is diagnostic only. The checkerboard
samples share a recording and sensor errors, so passing is a calibration check,
not independent accuracy evidence. A calibrated learned prediction is an
RGB+LiDAR condition; retain the unscaled RGB-only control.

`sample_registered_depth` handles co-located pinhole grids with nearest samples;
out-of-image rays stay unknown rather than copying border pixels.
`smooth_support` removes depth jumps and incomplete neighborhoods. Smoothness and
sensor confidence do not prove that a depth belongs to the visible foreground
object. In particular, consistent background returns can still be wrong for a
small object's shape. This implementation never fits local shape or depth offsets.

`view_consistency` reprojects each positive camera-Z sample into supplied witness
views. By default, two distinct witness IDs with at least 2 cm baseline from the
query camera must agree within 15 mm, with no farther-depth contradiction. Nearer
returns may occlude the query point and are treated as unknown. Missing views and
small baselines provide no evidence. Duplicate witness IDs are rejected. These
are distinct views, not necessarily statistically independent observations; pose
bias and shared model errors can still pass. The function changes neither poses
nor depth and returns support, contradiction, keep masks and witness counts.

Freeze acceptance thresholds and witness selection before reconstruction. Save
abstentions, original predictions and intermediate aligned surfaces. Render all
review views with identical settings and report retained coverage alongside any
residual improvement: a strict filter can look less contradictory simply by
removing most of a scene. Pose refinement, correspondence verification and
complete mesh generation remain separate operations.

## RGB camera refinement and foreground hypotheses

`rgb_pose_refinement.refine_camera_component` accepts an explicitly static,
connected set of calibrated cameras and triangulated RGB tracks. It jointly fits
bounded camera corrections and landmark positions, fixing the first camera and
retaining device-pose priors (1 cm translation, 0.5 degree rotation). A 20 mm
landmark-displacement penalty regularizes weak depth directions. Optional native
camera-Z anchors must already have passed confidence, neighborhood, view and
RGB-triangulation agreement checks; a farther background return must not anchor
an apparent foreground point. In components without usable depth anchors, metric
scale comes from the supplied camera baselines and priors, not independent depth
confirmation.

Every fifth track is excluded from optimization. Acceptance requires convergence,
updates away from bounds, no increase in invalid check tracks, at least 5% lower
median check reprojection error, at most 5% worse p90, and absolute p90 <=2 pixels.
Rejected candidates remain separate and applied poses stay unchanged. The default
budget is 500 evaluations, with an explicit cap of 2000. A disjoint-track check
within one recording is not an independent evaluation or a guarantee of correct
correspondence. Repeated texture can still pass multiple-view checks. Save the
exact tracks, settings, original/candidate/applied poses and rejection reasons.

`foreground_surface.depth_ownership` takes same-camera registered sensor depth,
RGB depth and an explicit RGB corroboration mask. It separates agreement, possible
background returns behind RGB geometry, nearer sensor returns or mismatches, and
unknown evidence. It does not certify an object's identity or automatically
promote a geometry replacement. Report unique native ray IDs: repeated RGB pixels
must not count as additional sensor measurements.

`foreground_surface.open_depth_mesh` triangulates only fully observed adjacent
image cells, including intervening pixels when using a stride. Metric edge and
depth-jump limits prevent bridging large discontinuities. Missing samples remain
holes. The result consists of open observed-surface patches, without object
completion, hidden bases, semantic IDs, collision generation or physics claims.
A foreground-aware fusion proposal must remain separate from the accepted scene
until independent appearance, ownership and geometric checks support promotion.
