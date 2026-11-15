# Generic scene development runner

This path removes chess-square assignments from static instance association and
mesh-to-scene compilation. It is a development adapter, not an evaluation-ready
end-to-end method or evidence of qualitative parity.

## Contracts

`general_scene.associate_instances` matches segmented instances using native
metric depth, mutual 3D proximity and bidirectional visibility-aware reprojection.
It requires distinct camera poses and rejects ambiguous associations. IDs are
spatially ordered object IDs, not chess identities. Rejections are retained; an
accepted subset does not certify that all objects were detected.

`build_mesh_scene` accepts world-frame visual meshes and convex collision parts.
It checks first-visible-surface ray depth, silhouette agreement, observed-point
distance, dimensions and visual/collision envelopes. It never rescales or fits a
prior to hide disagreement. A recorded vertical contact adjustment of at most
5 mm is allowed; evidence checks refer to the unadjusted prior.

The dominant horizontal background plane is a support hypothesis. Objects are
assembled bottom-up. A child needs full envelope containment and a unique
support. For object-on-object placement, both visual and collision geometry must
cover the parent's entire rectangular top. Coplanar triangle unions reject holes
and curved tops. This conservative domain excludes many real containers and
nonrectangular support surfaces. It certifies the supplied geometry, not hidden
real-world geometry.

The output preserves meshes and support edges through the existing NetworkX /
Scenic asset compiler. Replay reproduces the assembled graph exactly. Variation
uses the existing completion-preserving sampler with fixed orientations and a
fixed non-overlap branch. These are not unrestricted mesh-contact, articulation,
reachability, dynamics or task-success guarantees.

## Locked mesh-stage runs

Run `python -m lidar_video.general_run freeze recipe.json --out RUN`, retain the
returned SHA-256 externally, then run
`python -m lidar_video.general_run run RUN --lock-sha256 SHA256`.

The exact recipe fields are:

```json
{
  "schema": "general-mesh-recipe-v1",
  "scene_id": "development-example",
  "evaluation_scope": "development",
  "capture": "capture.npz",
  "detections": "detections.json",
  "mesh_manifest": "meshes.json",
  "static_scene": true,
  "policy": {},
  "jitter_m": 0.03,
  "samples": 100,
  "seed": 7,
  "simulate": false
}
```

Detection manifests use `generic-instance-masks-v1` and map **original source
frame IDs**, as strings, to `{"labels": "relative/indexed.png"}`. Zero is
background. Mesh manifests use `generic-world-mesh-priors-v1`, `units: "metres"`,
and an `objects` mapping keyed by associated object ID. Each object supplies
`visual`, a nonempty `collision` file list, and `provenance`. Locked input meshes
must be self-contained world-frame PLY files. Asset paths stay within their
manifest directory.

The lock covers recipe, capture, manifests, masks, meshes, package code and
runtime versions. Changes are checked before and after compilation/sampling.
A started run cannot be rerun in place, including after failure. These checks
detect accidental changes; they are not an OS sandbox or proof of blindness.
Only development scope is accepted. Model weights and detector inference are
outside this mesh-stage lock.

## Native learned development adapter

After provisioning the existing native SAM3D Colab environment:

```sh
python -m lidar_video.general_perception development-capture.npz \
  --out development-run --native-script scripts/colab_sam3d.py \
  --frames 0 1 2 --prompt 'tabletop object' --samples 100
```

SAM3 proposes generic instances. The same association checks choose reference
views. SAM3D receives measured scene context with only unreliable target depth
removed. Its metric mesh is transformed by `world_from_camera @ camera_from_object`
once; the reported scale is not applied again. CoACD supplies collision parts.
The adapter then freezes and executes the mesh stage. Any failure is retained;
there is no proxy substitution or manual completion in this path.

For irregular sensor recordings, the model's video is explicitly an **index-only
image container**. Original timestamps and source IDs remain in metadata. Its
1 FPS playback timing must never be used for temporal inference.

Development unit/integration fixtures include an analytic box scene and a
ray-generated platform, stacked cylinder, and rotated concave L. They use known
mesh priors and are explicitly not tests of learned reconstruction quality.
The tests check source-ID mapping, once-only transforms, native depth transport,
first-visible geometry, support holes/curves, 100 valid single-attempt variants,
asset preservation, and lock/failure handling.

Still required before reserved evaluation: a successful learned development run,
frozen model weights and runtime, complete baseline adapters with comparable
input/budget contracts, and an executable successor evaluation lock. Appearance
currently uses median observed color; texture transfer and complete inventory
recovery remain unvalidated. Do not use reserved evaluation data for development.
