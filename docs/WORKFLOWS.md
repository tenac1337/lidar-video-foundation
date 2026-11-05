# Workflow commands

Run from the repository root in the installed environment. Replace paths with your own local files. Generated results belong under ignored `runs/` or outside the repository.

## Annotation-assisted reconstruction

```sh
python scripts/process_stray_chess.py runs/imported \
  --annotations /path/to/board_annotations.json --out runs/board-reconstruction
```

Annotations identify a capture frame and original source frame, the full-resolution image size, visible grid intersection pixels and their board-grid coordinates. The importer output must retain matching RGB files. The script verifies IDs, refines corners, fits the optional folding seam and accepts only supported static-view alignments. See `annotated_board.py` for the validated annotation contract.

## Learned geometry integration

```sh
python scripts/integrate_learned_chess.py /path/to/reconstruction /path/to/learned \
  --out runs/integrated
python scripts/verify_chess_result.py runs/integrated/result
```

The integration checks calibration, source-frame identity and camera pose before mapping the learned meshes into local views. The script explicitly selects geometric fallback and preserves rejection reasons. `--no-simulation` skips rendering for a geometry-only check.

## Colab packages

The synthetic package contains this source, the helper scripts and the small synthetic fixture:

```sh
python scripts/package_colab.py --out runs/chess-colab-input.zip
```

For the iPhone template, first prepare a registered selected-frame capture and matching board calibration, then package its three explicit input files:

```sh
python scripts/package_colab.py --iphone-input /path/to/prepared-input \
  --out runs/iphone-colab-input.zip
```

The directory must contain `capture.npz`, `board_calibration.json` and `input_manifest.json`. This helper packages those files; it does not select frames, resample the recording or infer calibration. Uploading a real-capture package sends that selected data to your Colab runtime. Do not add credentials to the package; supply `HF_TOKEN` through Colab Secrets.

## Geometry profiling

The run-specific diagnostics expect a recording-result directory containing `reconstruction/` and `gpu-run/learned-partial/` as produced by the earlier local/GPU handoff:

```sh
python scripts/profile_mesh_lidar.py /path/to/recording-result --out runs/profile
python scripts/profile_depth_controls.py /path/to/recording-result --out runs/profile \
  --source /path/to/original-stray-export
```

The summarizer also compares against the original `hybrid/verification.json` and therefore expects the profile under the recording-result directory:

```sh
python scripts/profile_mesh_lidar.py /path/to/recording-result \
  --out /path/to/recording-result/geometry-profile
python scripts/profile_depth_controls.py /path/to/recording-result \
  --out /path/to/recording-result/geometry-profile --source /path/to/original-stray-export
python scripts/summarize_geometry_profile.py /path/to/recording-result/geometry-profile
```

These routines read saved meshes and depth, run local fits for diagnosis, and write reports. They do not accept fitted meshes into the simulation or alter the original inputs. The complete summarizer requires all ten original-PNG checks. Full-resolution real data and learned assets are deliberately external to this repository.

## Reliability-gated learned reconstruction

The A100 chess command now requires a JSON list of candidate frame indices and an explicit stationary-scene assertion:

```sh
lidar-video chess-reconstruct-gpu /path/to/capture.npz \
  --calibration /path/to/board_calibration.json \
  --candidate-frames /path/to/candidate_frames.json --static-scene \
  --native-script scripts/colab_sam3d.py --ambiguous-policy skip \
  --workers 1 --mesh-policy fallback --out runs/learned
```

Use at least two different camera viewpoints of the same unchanged arrangement. SAM 3 segments every candidate view. A chess square proposes an association; mutual reprojection must corroborate it. The best supported view supplies each object's SAM 3D input. Only trusted eroded foreground depth conditions the model; the full mask supplies its silhouette. No depth filling or extra mesh scaling is performed.

Current engineering thresholds for upright tabletop pieces are at least 24 unique native core/foreground samples, 65% valid depth in both mask and core, and 70% of valid core samples between 15 and 200 mm above the board. At least 90% of trusted native foreground samples must also lie inside the proposed square, consistent with the existing one-piece-per-square geometry contract. These conservative policies can exclude small or partially visible pieces; they are not calibrated iPhone uncertainty estimates. Camera separation requires at least 5 mm translation or 1 degree rotation and different source-frame IDs when supplied.

A mesh must pass every associated view: median distance <= one quarter square width, 90th-percentile distance <= half square width, silhouette IoU >= 0.5, and no silhouette loss above 0.05. Held-out median/p90 distances may not worsen by more than 1/2 mm respectively. Pose fitting trains only on the reference, preserves scale, and constrains the base to the board. Existing square-boundary and size checks still apply. Alignment first tries support snapping, then reference-only XY translation with orientation fixed, then a bounded rigid fit with an orientation prior measured as approximate surface displacement. A large initial base offset alone no longer forces rotation if support snapping passes. Held-out points never enter either optimization objective; they only determine whether a candidate passes the unchanged gates. All attempted transforms and view metrics are retained on rejection as well as acceptance. A failed fit stays rejected. All views remain an engineering consistency check, not independent ground truth.

The output retains `segmentation_audit.json`, raw per-frame masks, selected reference frames, per-object input depth, and validation evidence in the scene graph. If no candidate has corroborating reliable depth, the run stops before SAM 3D. Inspect/download those diagnostics even when no result scene exists. Missing objects do not establish empty squares.

Replay the depth gates against an older single-reference run without using a GPU:

```sh
python scripts/audit_object_depth.py /path/to/learned-partial --out runs/depth-gate-replay.json
```

This replay neither generates replacement meshes nor accepts the old ones. The updated Colab templates automatically enumerate their prepared static frames and choose one worker on 40 GB A100 or two on 80 GB A100.

New GPU fallback masks include `trusted_masks/instances.json` (`chess-instance-assignments-v1`), mapping each frame’s nonzero labels to square identities. The collector retains that identity instead of splitting an instance into neighboring square objects. Contradictory cross-square geometry still fails the existing bounds check; it is not silently clipped. Legacy masks without this sidecar retain square-based collection.
