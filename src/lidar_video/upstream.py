"""Exercise the original Real2Sim launcher on measured depth and a supplied proxy.

Only the original mujoco stage runs here. Geometry supplies shape/mask inputs;
it does not pretend SAM 3, SAM 3D, or motion reconstruction have run.
"""

import numpy as np
import trimesh
from PIL import Image
from roboskills.tools.coacd.coacd import decompose
from roboskills.workflows.real2sim import run

from lidar_video.capture import load_capture, normalize, world_points, write_json
from lidar_video.geometry import scene_graph
from lidar_video.pipeline import prepare_upstream


def upstream_smoke(capture, out):
    data = normalize(load_capture(capture))
    graph, _, _ = scene_graph(data)
    out = prepare_upstream(data, out)
    for folder in ("shape", "masks", "collision", "gravity"):
        (out / folder).mkdir()
    target = next(n for n in graph["nodes"] if n["kind"] == "object")
    transform = np.array(graph["scene_from_world"])
    position = np.array(target["position_m"])
    half = np.array(target["size_m"]) / 2
    for i, z in enumerate(data["depth"]):
        cloud = world_points(z, data["K"], data["world_from_camera"][i])
        cloud = cloud @ transform[:3, :3].T + transform[:3, 3]
        valid = (z > 0) & (data["confidence"][i] >= 1)
        # Background is the support; other foreground is excluded from plane fitting.
        labels = np.where((cloud[..., 2] > 0.012) & valid, 2, 0).astype(np.uint8)
        selected = np.all(np.abs(cloud - position) <= half + 0.006, axis=-1) & valid & (cloud[..., 2] > 0.012)
        labels[selected] = 1
        Image.fromarray(labels).save(out / f"masks/{i:05d}.png")
    mesh = trimesh.creation.box(extents=target["size_m"])
    mesh.visual.vertex_colors = np.tile(
        [*np.array(target["color_rgb"]) * 255, 255], (len(mesh.vertices), 1)
    ).astype(np.uint8)
    mesh.export(out / "shape/mesh.ply")
    scene_from_object = np.eye(4)
    scene_from_object[:3, 3] = position
    scene_from_camera = transform @ data["world_from_camera"][0]
    pose = np.linalg.inv(scene_from_camera) @ scene_from_object
    write_json(out / "shape/pose.json", dict(camera_from_object=pose.tolist(), source="depth box proxy"))
    write_json(out / "gravity/gravity.json", dict(down=(-scene_from_camera[2, :3]).tolist()))
    decompose(out / "shape/mesh.ply", out / "collision", 0.05, 20000, 32)
    # Include geocalib as an already-present output so upstream inputs() forwards
    # gravity. Its status truthfully records a pose-derived value, not model inference.
    write_json(
        out / "gravity/status.json",
        dict(ok=True, reason="gravity derived from camera pose; GeoCalib not executed"),
    )
    run(out, tools=["geocalib", "mujoco"], parallel=False)
    result = dict(
        status="passed",
        launcher="roboskills.workflows.real2sim.run",
        executed=["CoACD decomposition", "mujoco scene assembly"],
        supplied=["metric RGB-D", "camera-pose gravity", "geometric mask", "box-proxy mesh"],
        target=target["id"],
        scope="one-object scene-assembly contract smoke test, not learned perception or a physics rollout",
    )
    write_json(out / "upstream_smoke.json", result)
    return result
