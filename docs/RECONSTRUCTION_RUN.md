# Locked development reconstruction

Use a single recipe when replaying a capture with known calibration, instance-assigned masks and optional learned candidates. `reconstruction-freeze` validates and records the recipe before building; `reconstruction-run` executes it without parameter overrides and checks the generated export.

```json
{
  "schema": "reconstruction-run-v1",
  "scene_id": "development-001",
  "evaluation_scope": "development",
  "capture": "capture.npz",
  "calibration": "calibration.json",
  "masks": "support-masks",
  "mesh_manifest": "meshes.json",
  "expected_objects": ["a1", "f6"],
  "fusion_source_frames": [100, 200, 300],
  "reference_source_frame": 100,
  "static_scene": true,
  "mesh_policy": "strict",
  "min_height_m": 0.006,
  "samples": 20,
  "seed": 7,
  "simulation": false
}
```

All fields are required; unknown fields are rejected. Paths resolve relative to the recipe. Set `mesh_manifest` explicitly to JSON `null` to use geometric completion proxies. When a manifest is supplied, its object inventory must match `expected_objects` exactly. This avoids silently filling missing learned candidates with proxies. This interface currently supports static, strict-policy, development runs, with no recognizer or game-history input.

`fusion_source_frames` and `reference_source_frame` are original recording IDs, not array positions. The capture must retain unique integer `source_frame` values. The freeze receipt records their mapping to capture indices, preserving the requested order. Mask filenames, instance-assignment keys, candidate observation frames and calibration `reference_frame` remain capture array indices according to their existing formats. If calibration includes both `source_frame` and `reference_frame`, they must agree. Candidate reference/observation indices and any declared calibration reference must be in the selected set.

The mask directory must contain `instances.json` with schema `chess-instance-assignments-v1`, for example:

```json
{
  "schema": "chess-instance-assignments-v1",
  "frames": {
    "0": {"1": "a1", "2": "f6"},
    "1": {"1": "a1", "2": "f6"},
    "2": {"1": "a1", "2": "f6"}
  }
}
```

Supply registered indexed integer PNGs named `00000.png`, etc. For each selected frame, every positive label must have exactly one present assignment, and each object may have only one label. Across the selected frames, the declared object inventory must be complete. Background is zero. Broad masks without instance assignments cannot be substituted. Preflight verifies registration dimensions and label contracts, not segmentation accuracy or object identity in the images.

```sh
lidar-video reconstruction-freeze recipe.json --out runs/replay-001
lidar-video reconstruction-run runs/replay-001 --lock-sha256 THE_RETURNED_HASH
```

Record the returned hash separately before execution. The lock binds the recipe, capture, calibration, complete mask directory, candidate manifest and referenced mesh/pose/observation files, local package source files, Python/platform identity and numerical/tool runtime versions. Learned meshes must be self-contained PLY files. Inputs must exist before freeze; output directories must be separate from input files and mask inventories.

The runner rechecks these bindings before and after building and export checking. Changed inputs, code or runtimes require a new freeze in a new output directory. Locks use resolved local paths; moving inputs or environments requires a new freeze. This is an integrity record, not a portable environment image, operating-system isolation, a guarantee of bitwise determinism or proof of independent observations.

`scene/` contains the reconstruction, `scene-seal.json` binds its files, and `comparison/` contains readability, geometry and consistency decisions, including fresh Scenic sampling and seeded replay. `result.json` reports `completed` only if all three policies accept, otherwise `rejected` or `blocked`. A runtime failure writes `failed` with the error and preserves partial outputs. Once `execution.json` exists the run cannot execute again, including after failure. The CLI returns 3 for rejected/blocked checks and 2 for command or execution errors. An unchanged lock cannot override a failed geometry or inventory check.

`simulation=false` explicitly skips dynamics. Completion does not establish physical accuracy, calibrated uncertainty or suitability for a final evaluation split. Use the separate [frozen evaluator](EVALUATION.md) for fixed prediction scoring. Scenic executes trusted project code; the worker timeout is not a security sandbox.
