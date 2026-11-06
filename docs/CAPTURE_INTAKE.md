# Stray capture intake

`inspect-stray` checks an export before import or reconstruction, reads all source images, and writes a separate JSON report. It never alters the recording or generates geometry.

```sh
lidar-video inspect-stray /path/to/stray-export --fps 2 --out runs/intake.json
```

Use the same FPS for the later `import-stray` command. The report lists selected original frame IDs and timestamps, the regular import timeline and its maximum time displacement. Hardware RGB/depth exposure skew is not independently measured.

## Checks and scope

- Required `odometry.csv`, `rgb.mp4`, `depth/` and `confidence/` data; PNG IDs must match odometry exactly.
- Finite, increasing timestamps, consecutive original frame IDs, valid camera quaternions, positive focal lengths and supported distortion metadata.
- Every depth/confidence PNG, including unsampled frames. Depth must be uint16 millimetres; confidence must use uint8 levels 0, 1 or 2 on the same fixed grid.
- RGB decoded-frame count and dimensions through FFprobe, ignoring MP4 edit lists as the importer does. RGB/depth aspect ratios must agree.
- Usable depth in selected frames. Empty-depth frames omitted by sampling remain explicit warnings, not silently removed data.
- SHA256 and file-inventory consistency before/after inspection for the odometry, RGB video and all depth/confidence PNGs consumed by the importer.

FFprobe must be available on PATH. Video inspection has a 60-second timeout. PNG validation and hashing cover the full export, so large recordings take longer than a selected-frame import.

Readiness does not establish sharpness, static objects, accurate calibration, absolute scale or independent evaluation data. A selected timeline with fewer than two frames receives a warning even though a single-frame capture is valid interchange data.

## Report and exit status

The report schema is `stray-intake-report-v1`. Status `ready` means the declared format and association checks completed. `invalid` means data, sampling, split-plan or integrity checks failed. `blocked` means required video inspection was unavailable, such as missing FFprobe or a timeout. Errors and blocked checks remain in the report together; invalidity takes precedence in the overall status.

Exit code is 0 for ready and 3 for a saved non-ready report. An invalid output location or existing report instead produces a command error. Reports must be outside the source directory and are never overwritten. Keep generated reports under ignored output directories.

## Optional split plan

Choose original source IDs retained in the report, then prepare:

```json
{
  "schema": "scene-evaluation-split-v1",
  "construction_source_frames": [0, 120],
  "evaluation_source_frames": [240]
}
```

```sh
lidar-video inspect-stray /path/to/stray-export --fps 2 \
  --plan split-plan.json --out runs/intake-with-plan.json
```

Both sets must be nonempty, unique, disjoint and retained by the selected import FPS. A frame present in the raw export but dropped by sampling is rejected. The plan hash is recorded and checked for changes during inspection. Missing or malformed plans produce saved errors.

This command validates the plan but does not split pixels. After a ready report, import with matching settings and run `evaluation-split` as described in [frozen evaluation](EVALUATION.md). Give reconstruction only its construction bundle. Neither a split plan nor an intake report provides access isolation or proves that evaluation images were never inspected.
