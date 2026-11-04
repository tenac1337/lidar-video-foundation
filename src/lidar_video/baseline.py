"""Readiness checks and portable launcher for the pinned learned upstream pipeline."""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

from lidar_video.capture import write_json

COMMIT = "eca4691d2dfbb808dd58f3dd279fed154de6828a"
STAGES = "sam3 moge3 geocalib sam3d hawor align fit track coacd mujoco".split()


def preflight():
    weights = Path.home() / ".roboskills/weights"
    result = dict(
        system=platform.system(),
        architecture=platform.machine(),
        nvidia_smi=shutil.which("nvidia-smi") is not None,
        docker_cli=shutil.which("docker") is not None,
        uv=shutil.which("uv") is not None,
        sam3d_weights=(weights / "sam3d/checkpoints/pipeline.yaml").is_file(),
        hawor_weights=(weights / "hawor/hawor/checkpoints/hawor.ckpt").is_file(),
        mano_models=all(
            (Path.home() / ".robokit/body_models/mano" / name).is_file()
            for name in ("MANO_RIGHT.pkl", "MANO_LEFT.pkl")
        ),
        gated_model_access="not checked; having a token does not prove approved access",
    )
    blockers = []
    if result["system"] != "Linux":
        blockers.append(
            "Pinned SAM 3D/HaWoR GPU runtime requires Linux NVIDIA CUDA; Apple MPS is not supported"
        )
    for flag in ("nvidia_smi", "docker_cli", "uv", "sam3d_weights", "hawor_weights", "mano_models"):
        if not result[flag]:
            blockers.append(f"Missing prerequisite: {flag}")
    result.update(
        blockers=blockers,
        ready=not blockers,
        upstream_commit=COMMIT,
        status="blocked_preflight" if blockers else "prerequisites_present_runtime_unverified",
    )
    return result


def run_baseline(repo, out):
    repo, out = Path(repo).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != COMMIT:
        raise ValueError(f"Expected pinned RoboSkills commit {COMMIT}; got {head}")
    result = preflight()
    write_json(out / "preflight.json", result)
    video = repo / "assets/example_data/sugar-box.mp4"
    if not video.is_file():
        raise ValueError("Pinned upstream sample video is missing")
    if not result["ready"]:
        # Exercise the native stage's missing-weight guard, not a fake reconstruction.
        from roboskills.tools.sam3d.sam3d import WEIGHTS
        from roboskills.workflows.real2sim import run

        if not WEIGHTS.is_dir():
            try:
                run(
                    out / "native-guard-probe",
                    tools=["sam3d"],
                    video=video,
                    object="sugar box",
                    parallel=False,
                )
            except SystemExit as error:
                result["native_guard_probe"] = dict(stage="sam3d", outcome="blocked", reason=str(error))
        result["full_learned_pipeline_executed"] = False
        write_json(out / "result.json", result)
        return result
    command = [
        sys.executable,
        "-m",
        "roboskills.workflows.real2sim",
        "--workspace",
        str(out / "run"),
        "--video",
        str(video),
        "--object",
        "sugar box",
        "--tools",
        *STAGES,
        "--support",
        "pedestal",
        "--retry",
        "--no-parallel",
    ]
    # Sequential stages reduce simultaneous VRAM use. No nested agent/model CLI.
    with (out / "run.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    from roboskills.workflows.real2sim import folder, status

    statuses = {stage: status(out / "run" / folder(stage)) for stage in STAGES}
    result.update(
        full_learned_pipeline_executed=True,
        returncode=completed.returncode,
        stages=statuses,
        passed=completed.returncode == 0 and all(s.get("ok") for s in statuses.values()),
        scope="video reconstruction; robot manipulation is a separately validated known-scene baseline",
    )
    write_json(out / "result.json", result)
    return result
