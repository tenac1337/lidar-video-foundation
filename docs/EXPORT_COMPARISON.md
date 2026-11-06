# Fixed export comparison commands

These commands compare checks on existing scene exports. They do not reconstruct scenes, invoke repair agents, estimate physical accuracy, or replace the development audit's independent reviewer reports.

## Policies

Each command runs the following policies on the same input snapshot:

- `export`: require readable, nonempty graph and sample JSON records plus a nonempty Scenic file. This is a deliberately minimal export-readability policy, not a claim that the program compiles or the scene is correct.
- `geometry`: add existing graph constraints to the source graph and each saved sample: inventory, finite positions, positive dimensions, support relationships, support containment and non-overlapping object envelopes. This is envelope geometry, not RGB-D surface agreement.
- `consistency`: add a prior evidence seal, graph/sample metadata and asset checks, fresh Scenic compilation/sampling, and replay using the recorded seed/count. The scope excludes dynamics checks.

All policies use fixed inputs and no repairs. They have different checking costs; `elapsed_seconds` records time per policy. Do not describe them as matched-compute reconstruction methods. Missing required runtimes or seals produce `blocked`; malformed artifacts produce `reject`; completing the selected checks produces `accept` for that policy only.

Scenic worker dependency imports have a distinct unavailable-runtime exit status, so a missing dependency in the child process also produces `blocked`. A program compilation or replay error remains a rejection.

## Seal and compare an export

```sh
lidar-video export-seal runs/scene --out runs/scene-seal.json
```

The seal records all files under the scene directory using relative paths and SHA256. Write it outside that directory. It records file integrity, not quality or historical independence. Make the seal before handing artifacts to the comparison consumer; resealing a change does not prove the earlier evidence remained valid.

Example manifest, with paths relative to the manifest file:

```json
{
  "schema": "export-comparison-v1",
  "case_id": "case-001",
  "scene_id": "source-scene-001",
  "scene_dir": "scene",
  "expected_objects": ["left", "right"],
  "evidence_seal": "scene-seal.json"
}
```

`expected_objects` follows the graph validator's contract: use square IDs for nodes with a `square` field and node IDs otherwise.

`case_id` and `scene_id` are required and copied into the result. Aggregation rejects labels whose IDs differ from the selected result, rather than silently joining unrelated files. Regenerate older manifests/results without these IDs.

```sh
lidar-video export-compare runs/comparison-case.json --out runs/comparison
```

The result records decisions, reasons, completed checks, elapsed times, input hashes and policy scopes. If an input or seal changes while policies are running, no aggregate result is written. Outputs must be outside the input scene, and an existing output directory is never overwritten. Scenic files are executable project code; run these commands only on trusted artifacts. The subprocess timeout is not a security sandbox.

## Generated contract fixtures

```sh
lidar-video comparison-fixtures --out runs/contract-fixtures
```

This generates a clean two-box scene and controlled variants: overlapping objects, a missing object, a saved position outside its program range but inside board bounds, changed sample metadata, and a stale evidence seal. Non-integrity mutations receive matching seals so those cases isolate their intended contract violation. The stale-seal case changes graph bytes without changing geometry; its label concerns integrity only.

Each case has a `case.json` consumed by `export-compare`. For example:

```sh
lidar-video export-compare runs/contract-fixtures/clean/case.json \
  --out runs/contract-fixtures/clean/comparison
```

Run the same command for each case listed in the generated `labels.json` before aggregation. Generated labels follow the known mutations and exact fixture construction, not the checker's verdict. These are software regression fixtures, not new sensor data or unseen failure cases. Keep generated files under ignored `runs/` directories; no generated output is needed in source control.

## Aggregate independently labeled cases

```json
{
  "schema": "labeled-export-comparisons-v1",
  "cases": [{
    "case_id": "case-001",
    "scene_id": "source-scene-001",
    "label": "invalid",
    "label_reason": "An independently recorded required object is missing",
    "result": "case-001/comparison/comparison.json"
  }]
}
```

```sh
lidar-video comparison-summarize runs/labels.json --out runs/summary.json
```

The aggregation command joins labels after checking, rejects duplicate case IDs/result paths and requires a decision from every policy for every case. Labels are `valid`, `invalid` or `unknown`; unavailable ground truth should remain unknown.

- Invalid acceptance = accepted invalid cases / all invalid cases.
- Invalid rejection = rejected invalid cases / all invalid cases. Blocked cases are not rejections.
- Valid retention = accepted valid cases / all valid cases.
- Invalid fraction among known accepted = accepted invalid / accepted cases with valid or invalid labels. Accepted cases with unknown labels are counted separately.
- Decision coverage = cases with accept/reject / all cases, including unknown-label cases.

Every rate includes its underlying counts; absent denominators produce JSON null. Counts are also broken down by source scene. Multiple corruptions of one scene are dependent artifact cases, not additional independent scenes. In particular, zero invalid acceptances with zero decision coverage indicates unavailable checking, not a successful verifier.
