# Architecture

The package preserves measured camera-axis depth in metres and uses OpenCV camera coordinates. Camera poses transform those points into a right-handed z-up world; board calibration supplies the final scene frame. SAM 3D mesh scale is already applied by its upstream worker and must not be applied a second time.

## Modules

- `capture.py`: canonical input validation, coordinate conversion, intrinsic normalization and Record3D import.
- `stray.py`: Stray Scanner PNG/MP4/odometry import, explicit sampling and frame provenance.
- `board_calibration.py`, `annotated_board.py`: board fitting, orientation ambiguity and annotation-assisted static-view alignment.
- `reconstruction.py`: square-associated depth collections, learned mesh validation, explicit convex completion fallback and shared scene construction.
- `object_evidence.py`: unique native-depth samples, eroded-core reliability gates and mutual cross-view mask reprojection.
- `mesh_alignment.py`: reference-only rigid pose refinement constrained to board support, with per-view depth and silhouette acceptance.
- `gpu_reconstruction.py`: A100 enforcement, upstream SAM 3 detection, ambiguous-instance handling and SAM 3D jobs.
- `geometry.py`, `pipeline.py`: original tabletop reconstruction and measured-depth export to RoboSkills.
- `simulation.py`: pinned RoboSkills engine integration, collision geometry, rollouts and replay checks.
- `scenic_export.py`: probabilistic spatial variation and sample validation using the same object IDs and bounds.
- `chess_data.py`, `chess_recognition.py`, `chess_scene.py`: procedural fixtures, synthetic-style recognition and explicitly supplied game-state constraints.
- `manipulation.py`: separate known-scene synthetic manipulation experiment.
- `cli.py`: runnable stages and explicit input/output paths.

## Boundaries

Measured geometry, learned shape hypotheses, physical priors and simulation observations are separate evidence categories. Mass and friction are supplied priors. Scenic variation is a chosen test distribution, not a calibrated sensor posterior. A successful physics replay certifies simulator consistency, not sensor accuracy.

The default learned mesh policy is strict. Explicit fallback accepts valid input contracts but replaces meshes failing geometric quality checks with reported depth completions. Invalid transforms, corrupt files and infrastructure errors still fail. Static fusion is opt-in. Square names are provisional until physical a1 is resolved.

GPU runs use RoboSkills' published SAM 3D environment and unchanged worker. The adapter supplies measured metric depth directly and avoids default estimated-depth rescaling. A full hand/action reconstruction pipeline has not been demonstrated by this project.

Learned manifests tagged `instance-multiview-v1` carry registered indexed masks, instance labels, source frames and an explicit static-scene declaration. They require at least two distinct camera viewpoints. Untagged older manifests retain the previous validation behavior and are recorded as `legacy_square_cloud`; they do not constitute cross-view instance validation.
