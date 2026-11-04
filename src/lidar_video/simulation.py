"""Reuse RoboSkills build/rollout/measure. Do not animate or pin movable objects."""

import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from roboskills.tools.mujoco import engine

from lidar_video.capture import write_json
from lidar_video.synthetic import add_lighting, camera_axes


def simulate(graph, out, frames=60, fps=30, asset_root=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    bodies, geom_to_node = [], {}
    physical = [n for n in graph["nodes"] if n["kind"] in ("support", "object")]
    template = ET.Element("mujoco")
    assets = ET.SubElement(template, "asset")
    ET.SubElement(template, "worldbody")
    has_meshes = False
    for node in physical:
        movable = node["kind"] == "object"
        geom = dict(
            name=node["id"] + "_geom",
            type="box",
            size=np.array(node["size_m"]) / 2,
            rgba=[*node["color_rgb"], 1],
            friction=[0.8, 0.005, 0.0001],
        )
        if movable:
            geom["mass"] = node.get("mass_kg", 0.08)  # declared prior, never estimated from LiDAR
        parts = [geom]
        if "mesh_path" in node:
            if asset_root is None:
                raise ValueError("asset_root is required for relative graph meshes")
            has_meshes = True
            paths = [node["mesh_path"], *node["collision_paths"]]
            parts = []
            for i, relative in enumerate(paths):
                source = (Path(asset_root) / relative).resolve()
                if not source.is_relative_to(Path(asset_root).resolve()):
                    raise ValueError("Mesh path escapes asset_root")
                destination = out / "assets" / f"{node['id']}_{i}.obj"
                destination.parent.mkdir(exist_ok=True)
                shutil.copyfile(source, destination)
                name = f"{node['id']}_mesh_{i}"
                ET.SubElement(assets, "mesh", name=name, file=str(destination.resolve()))
                part = dict(
                    name=f"{node['id']}_geom_{i}",
                    type="mesh",
                    mesh=name,
                    rgba=[*node["color_rgb"], 1],
                    friction=[0.8, 0.005, 0.0001],
                )
                if i == 0:
                    part.update(contype=0, conaffinity=0, mass=0, group=2)
                else:
                    part.update(mass=node.get("mass_kg", 0.08) / (len(paths) - 1), group=3, rgba=[0, 0, 0, 0])
                parts.append(part)
        if node.get("checkerboard"):
            # Texture the support itself: coplanar decorative tile geoms cause z-fighting.
            has_meshes = True  # The asset template also carries textures/materials.
            texture, material = node["id"] + "_checker", node["id"] + "_material"
            gap = node.get("center_gap_m", 0)
            if gap:
                from PIL import Image

                side = 1024
                yy, xx = np.indices((side, side))
                square = node["square_size_m"]
                y = (yy + 0.5) / side * (8 * square + gap) - (4 * square + gap / 2)
                ranks = np.floor((y - np.sign(y) * gap / 2) / square + 4).astype(int)
                files = xx * 8 // side
                pixels = np.where(((files + ranks) % 2)[..., None], [184, 163, 125], [56, 66, 59])
                pixels[np.abs(y) < gap / 2] = [90, 85, 75]
                destination = out / "assets" / f"{texture}.png"
                destination.parent.mkdir(exist_ok=True)
                Image.fromarray(pixels.astype(np.uint8)).save(destination)
                ET.SubElement(assets, "texture", name=texture, type="2d", file=str(destination.resolve()))
            else:
                ET.SubElement(
                    assets,
                    "texture",
                    name=texture,
                    type="2d",
                    builtin="checker",
                    rgb1="0.22 0.26 0.23",
                    rgb2="0.72 0.64 0.49",
                    width="512",
                    height="512",
                )
            ET.SubElement(
                assets,
                "material",
                name=material,
                texture=texture,
                texrepeat="1 1" if gap else "4 4",
                texuniform="false",
            )
            geom.update(material=material, rgba=[1, 1, 1, 1])
        for part in parts:
            geom_to_node[part["name"]] = node["id"]
        bodies.append(dict(name=node["id"], pos=node["position_m"], free=movable, parts=parts))
    support = next(n for n in graph["nodes"] if n["kind"] == "support")
    center = np.array(support["position_m"])
    span = max(support["size_m"][:2])
    camera_pos = center + [0.75 * span, -1.3 * span, 1.6 * span]
    actor_xml = None
    if has_meshes:
        actor_xml = out / "assets.xml"
        ET.ElementTree(template).write(actor_xml, encoding="unicode")
    scene = engine.build(
        out / "scene.xml",
        actor_xml=actor_xml,
        bodies=bodies,
        cameras=[dict(name="overview", pos=camera_pos, xyaxes=camera_axes(camera_pos, center), fovy=47)],
        option=dict(timestep=0.002, integrator="implicitfast"),
    )
    add_lighting(scene)
    # Upstream saves qpos/qvel/ctrl, not solver warm-start state. Make the model
    # explicitly stateless between steps so interval replay verifies those fields.
    root = ET.parse(scene).getroot()
    ET.SubElement(root.find("option"), "flag", warmstart="disable")
    # The upstream builder resolves template directories absolutely. Our assets
    # are copied into this run, so make the final MJCF relocatable with the bundle.
    if has_meshes:
        root.find("compiler").set("meshdir", ".")
        root.find("compiler").set("texturedir", ".")
        for mesh in root.findall("asset/mesh"):
            mesh.set("file", str(Path(mesh.get("file")).relative_to(out.resolve())))
        for texture in root.findall("asset/texture"):
            if texture.get("file"):
                texture.set("file", str(Path(texture.get("file")).relative_to(out.resolve())))
    ET.ElementTree(root).write(scene, encoding="unicode")
    model, _ = engine.load(scene)
    steps = round((frames - 1) / fps / model.opt.timestep)
    report = engine.rollout(
        scene,
        out / "rollout",
        np.zeros((steps, model.nu)),
        fps=fps,
        frames=frames,
        camera="overview",
        size=(640, 480),
        video=True,
        preview=4,
    )
    replay = replay_check(out / "rollout")
    if (
        report["warnings"]
        or report["max_penetration_m"] > 0.005
        or max(replay["max_qpos_error"], replay["max_qvel_error"]) > 1e-7
    ):
        raise RuntimeError(f"Physics validation failed: {report['warnings']}, {replay}")
    final = copy.deepcopy(graph)
    final["source"] = "RoboSkills CPU MuJoCo rollout"
    final["source_capture_time_s"] = graph["time_s"]
    final["frame"] = frames - 1
    final["time_s"] = report["physical_seconds"]
    final["tracks"] = []
    measured = engine.measure(out / "rollout", [-1], {n["id"]: ("body", n["id"]) for n in physical})[0]
    for node in final["nodes"]:
        if node["id"] not in measured:
            continue
        node["position_m"] = measured[node["id"]]["xyz"]
        node["rotation_matrix"] = measured[node["id"]]["rotation"]
        node["evidence"] = "simulator_state"
        if "mesh_path" in node:
            node["mesh_path"] = f"assets/{node['id']}_0.obj"
            node["collision_paths"] = [
                f"assets/{node['id']}_{i + 1}.obj" for i in range(len(node["collision_paths"]))
            ]
    final["edges"] = [e for e in final["edges"] if e["relation"] != "contact"]
    for contact in report["contact_frames"][-1]:
        if contact["normal_force_N"] > 0:
            a, b = [geom_to_node[g] for g in contact["geoms"]]
            final["edges"].append(
                dict(
                    source=a,
                    relation="contact",
                    target=b,
                    evidence="simulator_contact",
                    normal_force_N=contact["normal_force_N"],
                    time_s=report["physical_seconds"],
                )
            )
    write_json(out / "simulated_scene_graph.json", final)
    write_json(
        out / "verification.json",
        dict(
            replay,
            warnings=report["warnings"],
            max_penetration_m=report["max_penetration_m"],
            scope="passive reconstructed object/support stability; no hand or robot manipulation",
            physical_priors=dict(object_mass_kg=0.08, friction=[0.8, 0.005, 0.0001]),
            physics_backend="unmodified roboskills.tools.mujoco.engine",
        ),
    )
    return report, replay


def replay_check(out):
    model, data = engine.load(Path(out) / "scene.mjb")
    saved = np.load(Path(out) / "trajectory.npz")
    max_error, max_velocity_error = 0.0, 0.0
    for i in range(len(saved["time"]) - 1):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = saved["qpos"][i]
        data.qvel[:] = saved["qvel"][i]
        data.ctrl[:] = saved["ctrl"][i]
        data.time = saved["time"][i]
        mujoco.mj_forward(model, data)
        for _ in range(round((saved["time"][i + 1] - saved["time"][i]) / model.opt.timestep)):
            mujoco.mj_step(model, data)
        max_error = max(max_error, float(np.max(np.abs(data.qpos - saved["qpos"][i + 1]))))
        max_velocity_error = max(max_velocity_error, float(np.max(np.abs(data.qvel - saved["qvel"][i + 1]))))
    return dict(
        max_qpos_error=max_error,
        max_qvel_error=max_velocity_error,
        checked_intervals=len(saved["time"]) - 1,
        method="reset each saved state, restore qpos/qvel/ctrl, re-step to the next frame",
    )
