"""Procedural chess RGB-D fixtures. Labels are evaluation data, never perception input."""

from pathlib import Path

import chess
import mujoco
import numpy as np
from PIL import Image
from roboskills.tools.mujoco import engine

from lidar_video.capture import validate, write_json
from lidar_video.synthetic import add_lighting, camera_axes

SQUARE = 0.045
SYMBOLS = ".PNBRQKpnbrqk"
HEIGHTS = dict(P=0.034, N=0.044, B=0.051, R=0.041, Q=0.060, K=0.069)


def square_xy(square, square_size=SQUARE):
    return (np.array([chess.square_file(square), chess.square_rank(square)]) - 3.5) * square_size


def piece_parts(symbol, rng):
    """A single procedural style, randomized proportions; not real chess meshes."""
    kind = symbol.upper()
    h = HEIGHTS[kind] * rng.uniform(0.97, 1.03)
    radius = 0.013 * rng.uniform(0.93, 1.07)
    color = rng.uniform(0.68, 0.84) if symbol.isupper() else rng.uniform(0.09, 0.19)
    rgba = [color, color * 0.96, color * 0.89, 1]
    parts = [
        dict(type="cylinder", pos=[0, 0, 0.003], size=[radius, 0.003]),
        dict(type="cylinder", pos=[0, 0, h * 0.3], size=[radius * 0.45, h * 0.22]),
    ]
    if kind == "P":
        parts.append(dict(type="sphere", pos=[0, 0, h - 0.008], size=[0.008]))
    elif kind == "N":
        parts += [
            dict(type="ellipsoid", pos=[0, 0, h * 0.65], size=[0.006, 0.008, h * 0.27]),
            dict(type="ellipsoid", pos=[0.004, 0, h * 0.85], size=[0.011, 0.006, h * 0.15]),
        ]
    elif kind == "B":
        parts.append(dict(type="ellipsoid", pos=[0, 0, h * 0.77], size=[0.007, 0.007, h * 0.23]))
    elif kind == "R":
        parts.append(dict(type="cylinder", pos=[0, 0, h * 0.77], size=[radius * 0.85, h * 0.12]))
        for x, y in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            parts.append(
                dict(type="box", pos=[x * 0.006, y * 0.006, h * 0.94], size=[0.003, 0.003, h * 0.06])
            )
    elif kind == "Q":
        parts += [
            dict(type="ellipsoid", pos=[0, 0, h * 0.69], size=[0.009, 0.009, h * 0.20]),
            dict(type="sphere", pos=[0, 0, h - 0.004], size=[0.004]),
        ]
    else:
        parts += [
            dict(type="ellipsoid", pos=[0, 0, h * 0.65], size=[0.009, 0.009, h * 0.18]),
            dict(type="box", pos=[0, 0, h * 0.90], size=[0.002, 0.002, h * 0.10]),
            dict(type="box", pos=[0, 0, h * 0.94], size=[0.007, 0.002, 0.002]),
        ]
    return [dict(p, rgba=rgba, mass=0.005) for p in parts]


def render_board(out, placement, seed, views=3, resolution=320):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    bodies = []
    for square in chess.SQUARES:
        x, y = square_xy(square)
        light = (chess.square_file(square) + chess.square_rank(square)) % 2
        color = [0.7, 0.62, 0.48, 1] if light else [0.27, 0.31, 0.28, 1]
        bodies.append(
            dict(
                name="square_" + chess.square_name(square),
                pos=[x, y, -0.0075],
                parts=[dict(type="box", size=[SQUARE / 2, SQUARE / 2, 0.0075], rgba=color)],
            )
        )
    for square, symbol in enumerate(placement):
        if symbol != ".":
            x, y = square_xy(square) + rng.uniform(-0.002, 0.002, 2)
            bodies.append(
                dict(
                    name="piece_" + chess.square_name(square),
                    pos=[x, y, 0],
                    euler=[0, 0, rng.uniform(-0.25, 0.25)],
                    parts=piece_parts(symbol, rng),
                )
            )
    cameras = []
    for i in range(views):
        angle = 2 * np.pi * (i / views + rng.uniform(-0.05, 0.05))
        pos = [0.20 * np.cos(angle), 0.20 * np.sin(angle), rng.uniform(0.57, 0.65)]
        cameras.append(dict(name=f"sensor{i}", pos=pos, xyaxes=camera_axes(pos, [0, 0, 0]), fovy=45))
    scene = engine.build(out / "source.xml", bodies=bodies, cameras=cameras)
    add_lighting(scene)
    model, data = engine.load(scene)
    mujoco.mj_forward(model, data)
    f = resolution / (2 * np.tan(np.radians(45) / 2))
    k = np.array([[f, 0, (resolution - 1) / 2], [0, f, (resolution - 1) / 2], [0, 0, 1]])
    rgb, depth, poses = [], [], []
    with mujoco.Renderer(model, height=resolution, width=resolution) as renderer:
        for i in range(views):
            renderer.update_scene(data, camera=f"sensor{i}")
            rgb.append(renderer.render().copy())
            renderer.enable_depth_rendering()
            z = renderer.render().copy()
            renderer.disable_depth_rendering()
            z[z > 2] = 0
            z[z > 0] += rng.normal(0, 0.0007, (z > 0).sum())
            z[rng.random(z.shape) < 0.02] = 0
            depth.append(z)
            pose = np.eye(4)
            pose[:3, :3] = data.cam_xmat[i].reshape(3, 3) @ np.diag([1, -1, -1])
            pose[:3, 3] = data.cam_xpos[i]
            poses.append(pose)
    rgb = np.array(rgb)
    rgb = np.clip(rgb.astype(float) * rng.uniform(0.90, 1.08) + rng.normal(0, 2, rgb.shape), 0, 255).astype(
        np.uint8
    )
    depth = np.array(depth, dtype=np.float32)
    capture = dict(
        rgb=rgb,
        depth=depth,
        confidence=np.where(depth > 0, 2, 0).astype(np.uint8),
        K=k,
        world_from_camera=np.array(poses),
        rgb_time=np.arange(views) / 30,
        depth_time=np.arange(views) / 30,
    )
    validate(capture)
    np.savez_compressed(out / "capture.npz", **capture)
    Image.fromarray(rgb[0]).save(out / "preview.png")
    write_json(
        out / "truth.json", dict(seed=seed, placement=list(placement), kind="procedural synthetic fixture")
    )
    write_json(
        out / "board_calibration.json",
        dict(
            square_size_m=SQUARE,
            board_from_world=np.eye(4).tolist(),
            origin="board center at top surface",
            axes="x toward h-file, y toward rank 8, z up",
            evidence="supplied synthetic calibration; board pose is NOT detected",
        ),
    )
    return capture


def random_placement(seed):
    rng = np.random.default_rng(seed)
    result = np.full(64, ".", dtype="U1")
    occupied = rng.choice(64, size=32, replace=False)
    result[occupied] = rng.permutation(np.resize(np.array(list(SYMBOLS[1:])), 32))
    return result.tolist()
