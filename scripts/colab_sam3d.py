"""Run the pinned RoboSkills SAM 3D worker natively in an ephemeral Colab VM.

The official image supplies its prebuilt /opt Python environment. No Docker
service, privileged nested container, or recompilation of CUDA extensions is used.
This is a Colab adapter; RoboSkills and SAM 3D source remain unchanged.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

IMAGE = "ghcr.io/openrobokit/roboskills-sam3d:0.3.0"
BASE = Path("/content/sam3d-native")
REPO = Path("/content/roboskills")
WEIGHTS = BASE / "home/.roboskills/weights/sam3d"


def run(args, **kwargs):
    subprocess.run(list(map(str, args)), check=True, timeout=1800, **kwargs)


def link(source, target):
    target = Path(target)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"Refusing to replace existing {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)


def setup():
    if sys.platform != "linux" or not Path("/content").is_dir():
        raise RuntimeError("This installer is only for an ephemeral Linux Colab runtime")
    if shutil.disk_usage("/content").free < 70 * 2**30:
        raise RuntimeError(
            "At least 70 GiB free space is required for image + unpacked environment + weights"
        )
    BASE.mkdir(exist_ok=True)
    run(["apt-get", "update", "-qq"])
    run(
        [
            "apt-get",
            "install",
            "-y",
            "-qq",
            "skopeo",
            "umoci",
            "libegl1",
            "libgl1",
            "libgles2",
            "libglib2.0-0",
            "libgomp1",
            "git",
            "curl",
        ]
    )
    oci = BASE / "oci"
    bundle = BASE / "bundle"
    record_file = BASE / "image.json"
    if not record_file.exists():
        raw = subprocess.check_output(["skopeo", "inspect", "--raw", "docker://" + IMAGE], timeout=120)
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        record_file.write_text(json.dumps(dict(image=IMAGE, digest=digest), indent=2))
    record = json.loads(record_file.read_text())
    pinned = IMAGE.rsplit(":", 1)[0] + "@" + record["digest"]
    if not (BASE / "image-ready").exists():
        run(
            [
                "skopeo",
                "copy",
                "--override-os",
                "linux",
                "--override-arch",
                "amd64",
                "docker://" + pinned,
                "oci:" + str(oci) + ":sam3d",
            ]
        )
        (BASE / "image-ready").touch()
    if not bundle.exists():
        run(["umoci", "unpack", "--rootless", "--image", str(oci) + ":sam3d", bundle])
    for name in ("python", "venv", "sam3d"):
        link(bundle / "rootfs/opt" / name, "/opt/" + name)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    link(WEIGHTS, "/weights/sam3d")
    run(
        [
            "/opt/venv/bin/python",
            "-c",
            "import torch,pytorch3d,nvdiffrast,gsplat,kaolin; print(torch.__version__,torch.cuda.get_device_name(0))",
        ]
    )
    print("SAM3D_NATIVE_ENVIRONMENT_READY", flush=True)


def weights():
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("Load the HF_TOKEN Colab secret before downloading weights")
    # The original provisioner expects a token file. Keep it private and ephemeral.
    BASE.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hf-auth-", dir=BASE) as temp:
        token_path = Path(temp) / "token"
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(os.environ["HF_TOKEN"])
        env = dict(os.environ, HOME=str(BASE / "home"), HF_TOKEN_PATH=str(token_path))
        run(["bash", REPO / "docker/weights.sh", "sam3d"], env=env)
    print("SAM3D_WEIGHTS_READY", flush=True)


def infer(args):
    env = dict(
        os.environ,
        CONDA_PREFIX="/opt/venv",
        HF_HOME=str(WEIGHTS / "hf"),
        TORCH_HOME=str(WEIGHTS / "torch"),
        HF_HUB_OFFLINE="1",
        PYOPENGL_PLATFORM="egl",
        EGL_PLATFORM="device",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
    )
    env.pop("HF_TOKEN", None)  # Model inference is offline after provisioning.
    run(
        [
            "/opt/venv/bin/python",
            Path(__file__).with_name("sam3d_native_rays.py"),
            REPO / "src/roboskills/tools/sam3d/sam3d.py",
            args.video,
            args.masks,
            args.depth,
            args.out,
            args.frame,
            args.label,
            args.samples,
            0,
            1,
            0,
        ],
        env=env,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["setup", "weights", "infer"])
    parser.add_argument("--video")
    parser.add_argument("--masks")
    parser.add_argument("--depth")
    parser.add_argument("--out")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "setup":
        setup()
    elif args.stage == "weights":
        weights()
    else:
        if not all((args.video, args.masks, args.depth, args.out)):
            parser.error("infer requires --video, --masks, --depth, --out")
        infer(args)
