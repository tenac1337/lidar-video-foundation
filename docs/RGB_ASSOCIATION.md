# RGB association when object depth is unresolved

`python -m lidar_video.rgb_association manifest.json output-directory` runs a
separate association proposal stage. It does not replace sensor depth evidence,
change the native scene, or claim a complete object inventory. The output
directory must be new.

The JSON manifest contains `static_scene: true`, `views`, an optional `policy`
matching `RGBPolicy`, and optional `extend_tracks` (default true). Each view
contains RGB and indexed-mask paths (`rgb`, `labels`), a unique integer
`source_frame`, calibrated 3x3 `K`, and a rigid OpenCV `world_from_camera` matrix.
Paths are relative to the manifest. Images must share a pixel grid. Camera
translations use metres. Empty detection images are allowed and remain empty.

The runner writes the input manifest and image hashes before association,
`associations.json`, and optional `extensions.json`. A failed validation is saved
as `failure.json`. Existing output directories are never overwritten.

## Evidence levels

1. Seed correspondences use RootSIFT descriptors, mutual ratio tests, and calibrated
   triangulation. Features lie inside masks, with an interior margin at least half
   their nominal keypoint size. Duplicate pixel locations do not count as distinct
   features. Positive depth, parallax and reprojection checks reject poor geometry.
2. Seed tracks require every pair of observations to have qualifying matches. Chains
   without all-pairs corroboration and multiple detections from one source view are
   rejected. Missing evidence is not silently filled by transitivity.
3. Optional extensions project those seed landmarks into additional instance masks.
   At least four unique pixels and 80% of the seed landmarks must support the same
   target. All sixteen corner perturbations of the four seed pixel coordinates,
   using the declared reprojection tolerance, must retain each counted landmark's
   label. Competing track claims to an observation are rejected.

Extensions are explicitly weaker evidence: projection inside a mask is not a new
RGB descriptor match. The corner perturbation screen is a sensitivity test, not
a confidence region or proof over continuous errors. Camera-pose uncertainty,
incorrect masks, repeated textures and specular reflections remain limitations.

The default descriptor ratio is 0.75; four unique matches, one-centimetre camera
baseline, one-degree parallax, two-pixel reprojection tolerance and three-pixel
spatial spread are required. Policies are saved with results.

The optional `geometry_guided: true` policy compares descriptors only within
calibrated epipolar bands. It additionally applies an absolute RootSIFT distance
cap (default 0.5); at least two geometrically eligible candidates are still needed
for a ratio test. The distance matrix is capped at four million feature pairs.
This is an explicitly separate ablation, not an automatic retry after failure.

Use the existing multi-view shape proposal stage only after freezing construction
and validation roles. Keep descriptor seed observations in construction when
using their triangulated landmarks. A mask used to test this method during
development does not become an independent evaluation image by reserving it for
one subsequent optimization. Do not claim extra independently reconstructed
objects from association counts or replace native depth validation with these
RGB-derived hypotheses.

Camera validation for this stage is independent of chess, reconstruction and
simulator imports. No object classes, support template, incumbent mesh geometry
or chess-square identities enter the association algorithm.
