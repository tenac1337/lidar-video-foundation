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
