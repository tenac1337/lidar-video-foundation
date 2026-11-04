# LiDAR + video foundation

Metric RGB-D input → RoboSkills simulation + JSON scene graph + Scenic probabilistic program.

This is the core implementation for an iPhone LiDAR/video reconstruction project, with CPU geometry processing on a Mac and optional SAM 3 / SAM 3D inference on Google Colab A100. RoboSkills is pinned to commit `eca4691d2dfbb808dd58f3dd279fed154de6828a`.

## Current state

- Canonical RGB-D validation, Stray Scanner import, and a fixture-tested Record3D importer.
- Detected or annotation-assisted chessboard calibration, optional static-view fusion, and measured-depth geometry.
- SAM 3 segmentation across static views, native-depth reliability gates, corroborated per-object reference selection, and SAM 3D import with held-out geometry validation or explicit fallback.
- One shared scene description drives MuJoCo simulation, scene graphs, and executable Scenic samples.
- Geometry profiling, original-input checks, held-out pose experiments, and physics replay verification.

**Detailed real-piece reconstruction remains unresolved.** The real iPhone A100 run generated 14 meshes; none passed every geometry check. Its simulation used 30 coarse depth completions. Profiling found suspect foreground depth and inconsistent placement across views. A subsequent three-view A100 run generated one candidate, which also failed held-out checks. Its fallback exposed cross-square instance splitting; the follow-up preserves identity and rejects mixed-square depth before inference. The current fallback is a prototype, not validated physical geometry. See [status and limitations](docs/STATUS.md).

## Install

Python 3.11+ is declared; local checks use Python 3.13 on Apple Silicon. Install FFmpeg separately and ensure it is on PATH.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,diagnostics]'
python -m lidar_video.cli doctor
```

`requirements.lock` records the earlier working Mac environment; it is a version snapshot, not a cross-platform hash lock. MuJoCo video rendering requires a supported graphics context. Linux headless runs may require `MUJOCO_GL=egl`. Neural inference requires separately approved model access and the Colab runtime described below.

## Run a synthetic scene

The two small committed RGB-D fixtures are procedural synthetic data. No real recording is required.

```sh
lidar-video chess-reconstruct tests/fixtures/chess-small/capture.npz \
  --auto-calibrate --static-scene --out runs/chess-small
```

This exports a simulation, `scene_graph.json`, `scenario.scenic`, sampled scenes and verification records. Use `--no-simulation` to exercise geometry and Scenic without video rendering. To run the original tabletop demonstration:

```sh
lidar-video demo --out runs/tabletop
```

Commands refuse to overwrite existing result directories; choose a new output path for each run.

## Use an iPhone recording

```sh
lidar-video import-stray /path/to/stray-export --out runs/imported --fps 2
lidar-video validate runs/imported/capture.npz
lidar-video chess-reconstruct runs/imported/capture.npz \
  --auto-calibrate --out runs/reconstructed --no-simulation
```

Automatic calibration requires visible grid intersections. Crowded/folding boards may need the annotation-assisted script documented in [the workflow guide](docs/WORKFLOWS.md). Add static fusion only when the board and pieces remained stationary. Input units, timestamps and coordinate conventions are in [the capture contract](docs/CAPTURE_FORMAT.md).

## Optional Colab A100 stages

```sh
python scripts/package_colab.py --out runs/chess-colab-input.zip
```

Open `notebooks/chess_reconstruction_colab.ipynb` in Colab, select A100, add an approved Hugging Face read token as the `HF_TOKEN` secret, and upload the generated synthetic package. The notebook downloads model weights into the temporary runtime. Provisioning is substantial; the setup checks for at least 70 GiB of free disk.

`notebooks/iphone_reconstruction_colab.ipynb` accepts a separately prepared selected-frame package. `notebooks/roboskills_gpu_stages.ipynb` exercises the upstream video perception example. Every committed notebook is a clean template without stored execution outputs. The full hand-motion pipeline and OMAR integration remain deferred.

## Development

```sh
python -m pytest tests -q
ruff check src scripts tests
```

See [the architecture](docs/ARCHITECTURE.md) for module responsibilities and [workflow commands](docs/WORKFLOWS.md) for reconstruction, profiling and verification. Private captures, generated scenes, learned meshes, model weights, runtime logs and executed notebooks stay outside Git. `.gitignore` includes safeguards; review staged files before committing.

Upstream attribution and licensing notes are in [NOTICE.md](NOTICE.md). No project-wide open-source license has been selected; this repository is intended to remain private.
