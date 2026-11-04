"""Synthetic sensor fixture rendered by MuJoCo, using RoboSkills' scene builder.

No chess recognition or real sensor accuracy is implied by this fixture.
"""

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from roboskills.tools.mujoco import engine

from lidar_video.capture import validate, write_json


def camera_axes(position, target):
    backward = np.asarray(position) - np.asarray(target)
    backward /= np.linalg.norm(backward)
    right = np.cross([0, 0, 1], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    return [*right, *up]


def encode_video(images, path, fps=30):
    n, h, w, _ = images.shape
    if not n or h % 2 or w % 2:
        raise ValueError("H264 output requires nonempty frames with even dimensions")
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    subprocess.run(command, input=np.ascontiguousarray(images).tobytes(), check=True)


def add_lighting(path, checker=False):
    root = ET.parse(path).getroot()
    ET.SubElement(root.find("worldbody"), "light", pos="0 -0.3 1.5", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.5 0.5 0.5")
    if checker:
        assets = ET.SubElement(root, "asset")
        ET.SubElement(
            assets,
            "texture",
            name="grid",
            type="2d",
            builtin="checker",
            width="512",
            height="512",
            rgb1="0.78 0.78 0.75",
            rgb2="0.3 0.32 0.33",
        )
        ET.SubElement(assets, "material", name="board_grid", texture="grid", texrepeat="4 4", reflectance="0")
        root.find("worldbody/body/geom").set("material", "board_grid")
    ET.ElementTree(root).write(path, encoding="unicode")


def generate(out, frames=24, seed=7, noisy=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    objects = [
        dict(
            id="red_block", position=[-0.12, -0.09, 0.035], size=[0.05, 0.05, 0.07], color=[0.78, 0.2, 0.15]
        ),
        dict(id="blue_block", position=[0.08, -0.07, 0.05], size=[0.06, 0.04, 0.1], color=[0.15, 0.35, 0.8]),
        dict(id="green_block", position=[0.09, 0.11, 0.025], size=[0.07, 0.06, 0.05], color=[0.2, 0.65, 0.3]),
    ]
    bodies = [
        dict(
            name="support",
            pos=[0, 0, -0.015],
            parts=[dict(name="support_geom", type="box", size=[0.24, 0.24, 0.015], rgba=[0.7, 0.7, 0.7, 1])],
        )
    ]
    bodies += [
        dict(
            name=o["id"],
            free=True,
            pos=o["position"],
            parts=[
                dict(
                    name=o["id"] + "_geom",
                    type="box",
                    size=np.array(o["size"]) / 2,
                    rgba=[*o["color"], 1],
                    mass=0.08,
                    friction=[0.8, 0.005, 0.0001],
                )
            ],
        )
        for o in objects
    ]
    cameras = []
    for i in range(frames):
        angle = 2 * np.pi * i / frames
        position = [0.08 * np.sin(angle), -0.56 + 0.03 * np.cos(angle), 0.62]
        cameras.append(
            dict(name=f"sensor{i}", pos=position, xyaxes=camera_axes(position, [0, 0, 0]), fovy=47)
        )
    scene = engine.build(
        out / "source.xml",
        bodies=bodies,
        cameras=cameras,
        option=dict(timestep=0.002, integrator="implicitfast"),
    )
    add_lighting(scene, checker=True)
    model, data = engine.load(scene)
    for _ in range(250):
        mujoco.mj_step(model, data)
    h, w = 240, 320
    f = h / (2 * np.tan(np.radians(47) / 2))
    k = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
    rgbs, depths, poses = [], [], []
    with mujoco.Renderer(model, height=h, width=w) as renderer:
        for i in range(frames):
            for _ in range(16):
                mujoco.mj_step(model, data)
            renderer.update_scene(data, camera=f"sensor{i}")
            rgbs.append(renderer.render().copy())
            renderer.enable_depth_rendering()
            z = renderer.render().copy()
            renderer.disable_depth_rendering()
            z[z > 3] = 0
            depths.append(z)
            pose = np.eye(4)
            pose[:3, :3] = data.cam_xmat[i].reshape(3, 3) @ np.diag([1, -1, -1])
            pose[:3, 3] = data.cam_xpos[i]
            poses.append(pose)
    depth = np.stack(depths).astype(np.float32)
    confidence = np.where(depth > 0, 2, 0).astype(np.uint8)
    if noisy:
        valid = depth > 0
        depth[valid] += rng.normal(0, 0.0005, valid.sum())
        holes = rng.random(depth.shape) < 0.015
        depth[holes] = 0
        confidence[holes] = 0
        low = (rng.random(depth.shape) < 0.015) & (depth > 0)
        confidence[low] = 0
        depth[low] += 0.03
    recording = dict(
        rgb=np.stack(rgbs),
        depth=depth,
        confidence=confidence,
        K=k,
        world_from_camera=np.stack(poses),
        rgb_time=np.arange(frames) / 30,
        depth_time=np.arange(frames) / 30,
    )
    report = validate(recording)
    np.savez_compressed(out / "capture.npz", **recording)
    encode_video(recording["rgb"], out / "source.mp4")
    Image.fromarray(recording["rgb"][0]).save(out / "source.png")
    write_json(
        out / "ground_truth.json",
        dict(
            source_kind="synthetic",
            objects=objects,
            board_size_m=[0.48, 0.48],
            noise_std_m=0.0005 if noisy else 0,
            note="Stationary boxes; camera moves. Truth is used only for evaluation, never reconstruction.",
        ),
    )
    write_json(
        out / "capture.json",
        dict(
            report,
            source_kind="synthetic MuJoCo rendering",
            seed=seed,
            renderer_note="macOS OpenGL may warn ARB_clip_control is unavailable; depth accuracy is fixture-tested",
        ),
    )
    return out / "capture.npz"
