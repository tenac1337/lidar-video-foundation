# Frozen scene evaluation

The development repair loop remains development-only. These separate commands measure fixed predictions without calling reconstruction, support fitting, candidate selection, repair workers, or scene acceptance gates. They do not certify independence or scientific validity.

## 1. Separate construction pixels from evaluation pixels

A data custodian chooses source-frame IDs before reconstruction. Frame IDs refer to the original recording, not indices in a selected array. The canonical capture must retain unique integer `source_frame` values. Example plan:

```json
{
  "schema": "scene-evaluation-split-v1",
  "construction_source_frames": [100, 200],
  "evaluation_source_frames": [300]
}
```

```sh
lidar-video evaluation-split capture.npz --plan split-plan.json --out split
```

The command preserves native intrinsics, sample IDs, timestamps, and camera poses. It writes separate `construction/capture.npz` and `evaluation/capture.npz` files, their hashes, and the omitted source IDs. Unsupported capture fields fail rather than being silently mis-sliced. Give the builder only the construction bundle and permitted calibration. Keep evaluation masks and independent measurements with the custodian. Separate files and hashes are not operating-system access controls.

Existing inspected captures cannot become confirmatory data by splitting them again. Per-scene calibration must use only permitted construction views; evaluation labels must not also fit the quantity being measured.

## 2. Record the fixed prediction

Export self-contained PLY meshes in metres with an explicit rigid `board_from_mesh` transform. A method may return an empty object list; that produces missing-object records, not perfect scores. Record whether the method accepted, rejected, or abstained before observing the evaluation results.

```json
{
  "schema": "frozen-scene-prediction-v1",
  "scene_id": "T01",
  "method_id": "geometric-checks",
  "method_decision": "accept",
  "units": "metres",
  "construction_source_frames": [100, 200],
  "calibration_source_frames": [100],
  "objects": [{
    "id": "piece_01",
    "mesh": "piece_01.ply",
    "board_from_mesh": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
  }]
}
```

The identity transform is appropriate only when mesh vertices are already in the board frame. Do not infer pose or rescale a mesh from evaluation data. If the method has no segmentation output, object association to reference IDs must follow a predeclared mapping procedure, with ambiguity reported; do not choose correspondences to minimize evaluation error.

## 3. Bind the evaluation inputs

```json
{
  "schema": "scene-evaluation-case-v1",
  "scene_id": "T01",
  "evaluation_scope": "confirmatory",
  "previously_inspected": false,
  "custodian_attestation": "Construction used only source frames 100/200; evaluation images and labels were withheld from selection and repairs.",
  "prediction": "prediction.json",
  "capture": "split/evaluation/capture.npz",
  "calibration": "calibration.json",
  "expected_objects": ["piece_01"],
  "evaluation_source_frames": [300],
  "observations": [{"object_id":"piece_01", "source_frame":300, "mask":"300.png", "label":1}]
}
```

Use `development_replay` for inspected or synthetic development data. Confirmatory scope requires an explicit custodian declaration; software cannot establish whether it is truthful. All expected objects need at least one declared observation. For each frame, one indexed integer mask must use distinct labels for distinct objects, with zero as background. Declare absent reference labels explicitly rather than silently dropping an expected object; they are reported as unscorable.

```sh
lidar-video evaluation-freeze case.json --out evaluation-run
```

Preserve the returned `lock_sha256` in the experiment ledger before scoring. The lock binds the prediction, meshes, capture, calibration, masks, source manifest, evaluator source, runtime versions, and metric settings. Paths resolve relative to their containing manifest. Input changes require a new freeze and a documented reason; the new run is not a substitute for the old run.

```sh
lidar-video evaluation-score evaluation-run --lock-sha256 THE_RECORDED_HASH
```

Scoring checks the lock and hashes before and after computation, and refuses to overwrite an existing result. It does not produce reviewer reports, accepted-scene certificates, or repair suggestions. Do not feed final results back into the method and then report the revised method on the same split as confirmatory.

## Metric interpretation

- Depth: all finite positive depth samples with confidence at least 1 inside the reference object mask, deduplicated by original native sample ID. No foreground-height, square-membership, erosion, or model-fit acceptance filter is applied. Camera-axis depth is backprojected with native intrinsics and the frozen board calibration.
- Surface agreement: one-way Euclidean distance from those observations to 60,000 points sampled from the fixed mesh surface with seed 0. Report median/p90 in metres. This approximates distance to the mesh; it is not exact triangle distance, symmetric shape accuracy, or an independent sensor calibration result.
- Silhouette: the existing projected triangle-mask calculation at maximum width 320 pixels, excluding pixels labeled as other instances. Meshes are not simplified or aligned during scoring. This is not a full z-buffer visibility renderer. Small silhouettes, near-plane crossings and incomplete occluder labels are limitations to inspect before the protocol freeze.
- Missing predictions receive an explicit record and zero silhouette IoU when a reference silhouette exists. Missing reference labels, invalid meshes, and absent usable depth remain explicit statuses; absent depth is not zero error. Unexpected object IDs are listed.
- Per-object means give each declared view equal weight and include scored-view counts. There is deliberately no pooled scene score that hides missing observations. Consumers should retain missing/unscorable counts and the full inventory alongside conditional accuracy.

This evaluator's all-valid-mask depth metric differs from the construction gates' eroded, foreground-filtered depth metric. Do not mix their numerical values in a before/after claim. Hold scoring settings and masks fixed across compared methods.

## Supported outputs

The runner covers geometric scoring, optional height-reference errors and integrity, not independent correctness labeling or full pose/reference-shape scoring. [Export comparisons](EXPORT_COMPARISON.md) provide fixed-policy decisions, elapsed times and aggregation with externally supplied labels. Do not derive correctness labels from the same checks being compared.

## Optional independent height measurements

Add `"height_references": "height.json"` to the evaluation case before freezing. The file is hash-bound with the other inputs and must use the same scene and object IDs:

```json
{
  "schema": "object-height-references-v1",
  "scene_id": "T01",
  "units": "metres",
  "axis": "board_z",
  "not_used_for_reconstruction": true,
  "objects": {
    "piece_01": {
      "height_m": 0.05,
      "resolution_m": 0.001,
      "measurement_method": "ruler"
    }
  }
}
```

This compares the frozen mesh's maximum minus minimum board-Z coordinate with an independently measured upright-object height. It does not compare the collider or shift, resize or fit the prediction. The result includes signed and absolute error in metres, the supplied instrument resolution and method, and an explicit scored-object count. Instrument resolution is not a statistical confidence interval or a pass/fail threshold.

Reference IDs may cover a subset of expected objects; absent references and missing/invalid predictions remain explicit statuses. Wrong units, a different scene, unknown object IDs, invalid numbers, and measurements declared as used for reconstruction are rejected. The independence flag is a human declaration, not something checksums can prove.

Do not supply base diameter as full mesh width: those are different physical quantities. Width and full pose reference scoring are not implemented. Existing evaluation locks must be regenerated after evaluator code changes; preserve earlier runs rather than overwriting them.
