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
