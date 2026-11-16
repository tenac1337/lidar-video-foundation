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

SAM3 proposes generic instances. Raw masks and detector metadata are preserved.
A confidence-ordered mask-IoU threshold of 0.8 suppresses near duplicates. Pixels
claimed by multiple remaining masks are left unassigned to every instance; a
mask retaining less than 80% of its area is rejected. The per-frame audit retains
raw detection indices, exclusions and removed-pixel counts. This is conservative
abstention, not proof of correct instance boundaries or a complete inventory.
The same association checks choose reference views; pairwise evidence and
observation-to-source mappings are retained for diagnosis. SAM3D receives measured scene context with only unreliable target depth
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

## Discovery before reconstruction

`python -m lidar_video.general_discovery CAPTURE --out RUN --frames 0 1 2`
performs a development-only discovery diagnostic without generating meshes. It
uses the generic prompts `object` and `small object`, and retains both duplicate
IoU conditions (0.8 and 0.5). Neither condition is declared a winner by its count.
All raw masks, scores, cleaned label maps, raw-to-clean index mappings, pairwise
association evidence, source IDs and excluded observations are retained. Each
condition reports `inventory_complete: false`; corroborated tracks do not prove
that every object was found. The input must be development data, not a reserved
evaluation capture. This diagnostic does not replace a frozen evaluation recipe.

`run_checkpoint.pack` packages an explicit list of small artifact files into a
bounded, checksummed JSON envelope. A notebook can display this envelope as
`application/json`. A local caller retrieves the saved notebook, extracts the
envelope, and calls `run_checkpoint.restore` with its recorded SHA-256. Only then
is persistence verified. The transport rejects path escapes, member tampering,
oversized archives and overwriting an existing destination. Credentials and model
weights must not be included in the explicit artifact list.

Large live Colab outputs can be replaced with an `Output hidden` stub when the
notebook is saved. Use `run_checkpoint.notebook_chunks(payload)` and display each
returned JSON object in a **separate cell**. Recover with
`payload_from_notebook(notebook, expected_sha256=recorded_hash)`, then call
`restore` as above. Missing, duplicate or mixed pieces fail recovery. Live display
and the notebook's saved indicator still do not establish checkpoint durability.

The discovery API accepts a `checkpoint(output_directory, stage)` callback after
each completed view and after association. A runtime orchestrator can pause in
that callback until the local copy is verified. Saving a ZIP in `/content`,
displaying a download link, or reporting that a callback ran is insufficient proof
of durable storage. Preserve failed attempts and their raw proposals before any
new development condition; never silently rerun into the same output directory.

Each discovery condition also writes `support-depth-diagnostic.json`. Confident
sensor returns inside an RGB mask do not necessarily describe the object: they
may lie on a nearby support. The diagnostic fits a horizontal plane in an
8-native-pixel ring excluding all proposal masks and reports native sample counts
and height distributions. Missing background evidence stays unresolved. Plane
identity is uncertain, and thin objects may legitimately lie near it, so these
numbers neither reject objects nor certify their surfaces. They never alter
measured depth or relax the existing association and mesh validation checks.
