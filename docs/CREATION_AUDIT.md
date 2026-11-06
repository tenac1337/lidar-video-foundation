# Creation and audit workflow

This adapts Gym-Anything's creator → independent auditors → repair loop to the existing RGB-D/MuJoCo/Scenic pipeline. It does not import RoboWorld or replace measured reconstruction with generated scenes. No Isaac Sim or RTX renderer is needed for these local checks. A100 remains useful for new SAM inference, but saved mesh audits run locally.

## What is implemented

Four fresh reviewer roles consume explicit case and evidence packets:

- Geometry: original native-ray, mask, transform and silhouette gates, every expected object, square bounds and linkage to full exported learned meshes.
- Physics: a fresh bounded MuJoCo passive rollout, body inventory, free movable bodies, declared mass/friction, collision envelopes, support contact, solver warnings and penetration. Actuators, equality constraints and gravity compensation are rejected in this passive-only scope.
- Exports: graph structure and assets, saved Scenic samples, plus fresh Scenic compilation/sampling with stable per-ID metadata, asset references and dimensions; only object positions and documented sampling provenance may vary.
- Adversarial: concrete missing-object, nonfinite-coordinate, broken-identity and overlapping-object mutations against the same graph validator, supplemented by independent agent review.

The checks bind primitive collider position/orientation and compiled mesh assets to the graph. Scenic samples must preserve each ID’s declared identity, assets and metadata; matching only the overall inventory is insufficient. Exported relationships (including occupancy and board membership) and coordinate metadata must also match; only support-edge evidence may become `sampled_constraint`. Raw Scenic object IDs must be unique before conversion to a sampled graph, so extra objects cannot disappear behind duplicate IDs.

Agent reviews cannot override deterministic fail or blocked results. Acceptance requires all four machine checks and all four correctly bound reviewer reports to pass. A completed agent process is not acceptance. Missing reports remain `awaiting_reviews`; missing simulation evidence is `blocked` in its check, never silently skipped. Any changed input, policy source, packet or saved numerical check invalidates the round.

The current contracts cover development evidence and passive tabletop scenes. They do **not** certify robot actions, grasp feasibility, reward functions, chess-piece semantics, independent physical accuracy or a pristine final evaluation. Existing inspected views remain development data. Do not supply an untouched evaluation split to the repair loop. The case schema rejects `evaluation_scope: "final"`.

## Case files

Paths resolve relative to the case JSON. Use a new output directory for every run.

```json
{
  "schema": "creation-audit-case-v1",
  "evaluation_scope": "development",
  "geometry_scope": "measured",
  "capture": "capture.npz",
  "calibration": "calibration.json",
  "mesh_manifest": "meshes.json",
  "expected_objects": ["f8", "b8"]
}
```

The mesh manifest retains `roboskills-chess-meshes-v1` and per-entry `instance-multiview-v1` or explicit-split `instance-multiview-v2` contracts (see [multiview completion](MULTIVIEW_COMPLETION.md)). All observation masks, candidate meshes/poses, capture and calibration are hashed. Every expected object must be present; an empty or partial manifest cannot pass. The optional `scene_dir` points to the matching reconstruction output containing `scene_graph.json`, `scenario.scenic`, `scenic_samples.json`, assets and `simulation/scene.xml`. An absent scene blocks downstream checks.

For synthetic fixtures use `geometry_scope: "synthetic_fixture"`, `scene_dir`, and the exact expected object names. Do not supply capture, calibration or mesh manifest. A synthetic pass says nothing about real sensor reconstruction.

## Run with the existing coding-agent team

```sh
lidar-video audit-prepare case.json --out audit/round-01
# Give each independent reviewer only packets/<role>.json.
# Each writes reviews/<role>.json according to its report_contract.
lidar-video audit-finalize audit/round-01
```

`audit-prepare` intentionally exits 3 while awaiting reviews. All audit commands exit 0 only for `accepted`; inspect `decision.json` or `loop.json` for any other status. Packets include round IDs and SHA256 bindings to prevent stale reports. Supporting evidence references must resolve to existing files within that round. Reviews should be fresh sessions and must not read other reviewers' verdicts.

## Automated CLI workers

With an already authenticated Codex CLI, create a worker JSON containing an argv array (use actual absolute paths):

```json
["/path/to/venv/bin/python", "/path/to/repo/scripts/codex_audit_worker.py"]
```

```sh
lidar-video audit-loop case.json --out audit/run \
  --worker worker.json --rounds 1 --timeout 600
```

The bundled worker launches fresh `codex exec` sessions with a read-only sandbox, explicit output schema and final-report file. It inherits the configured model and authentication; no tokens belong in cases or worker JSON. It does not bypass approvals or sandboxing. `CODEX_BIN` may identify an existing CLI binary.

For optional repairs, a second argv JSON can point to `scripts/codex_builder_worker.py`. Pass it as `--builder builder.json --rounds 3`. The builder gets the case and audit decision and writes new candidates in a separate repair workspace. Its final response is a proposed case JSON. It cannot change expected inventory, measured evidence, validation metadata or policy; only candidate mesh/pose and scene export paths may change. The loop detects changes to prior round inputs, rejects contract changes, re-audits new candidates, and stops when accepted, unchanged, failed or the maximum of three rounds is reached. Each worker has a wall-time limit; nonzero exit, timeout or absent report is failure.

Other backends can implement the same interface: argv is executed without a shell, and packet/report paths are appended as the final two arguments. The builder uses the corresponding repair-packet/next-case paths. Custom workers run with the invoking account's permissions. Hash checks and prompt rules are **integrity checks, not a security sandbox**. Only use trusted case artifacts and worker commands; Scenic and MJCF loading are executable/native-code paths.

## Provenance

The workflow design follows [Gym-Anything for Robotics](https://arxiv.org/html/2610.40565v1) and its [creation/audit implementation](https://github.com/gym-anything/robotics/tree/main/src/gym_anything_robotics/creation_audit). This implementation is original and scoped to our existing pipeline; it does not copy their launchers or use their permission-bypass flags.

The Codex adapters follow [official non-interactive CLI documentation](https://learn.chatgpt.com/docs/non-interactive-mode). The automated checks and reports remain provider-independent.
