"""CPU Cartesian waypoint control of the upstream Panda/2F85, with contact-only props.

This is a known-scene manipulation baseline, not a policy learned from video.
The IK scratch state is used only for planning; rollout changes actuator controls.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from roboskills.tools.mujoco import engine
from scipy.spatial.transform import Rotation

from lidar_video.capture import write_json
from lidar_video.synthetic import camera_axes


def solve_ik(model, target, rotation, seed, tolerance=0.0002):
    data = mujoco.MjData(model)
    data.qpos[:7] = seed
    site = model.site("pinch").id
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(250):
        mujoco.mj_forward(model, data)
        pos_error = np.asarray(target) - data.site_xpos[site]
        rot_error = Rotation.from_matrix(rotation @ data.site_xmat[site].reshape(3, 3).T).as_rotvec()
        if np.linalg.norm(pos_error) < tolerance and np.linalg.norm(rot_error) < 0.003:
            return data.qpos[:7].copy()
        mujoco.mj_jacSite(model, data, jp, jr, site)
        jac = np.vstack([jp[:, :7], jr[:, :7] * 0.2])
        error = np.r_[pos_error, rot_error * 0.2]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(6), error)
        dq *= min(1, 0.15 / max(np.linalg.norm(dq), 1e-10))
        data.qpos[:7] = np.clip(data.qpos[:7] + dq, model.jnt_range[:7, 0], model.jnt_range[:7, 1])
    raise ValueError(f"IK did not converge at {target}: position error {np.linalg.norm(pos_error):.6f} m")


def build_scene(out, start, target, size):
    actor = Path(engine.__file__).parent / "franka/panda_2f85.xml"
    center = (np.asarray(start) + np.asarray(target)) / 2
    camera = [1.05, -1.1, 0.9]
    scene = engine.build(
        Path(out) / "scene.xml",
        actor_xml=actor,
        bodies=[
            dict(
                name="table",
                pos=[*center, -0.025],
                parts=[
                    dict(name="table_geom", type="box", size=[0.24, 0.24, 0.025], rgba=[0.47, 0.38, 0.25, 1])
                ],
            ),
            dict(
                name="prop",
                free=True,
                pos=[*start, size[2] / 2 + 0.001],
                parts=[
                    dict(
                        name="prop_geom",
                        type="box",
                        size=np.asarray(size) / 2,
                        rgba=[0.85, 0.3, 0.1, 1],
                        mass=0.04,
                        friction=[0.8, 0.005, 0.0001],
                    )
                ],
            ),
        ],
        cameras=[dict(name="overview", pos=camera, xyaxes=camera_axes(camera, [0.35, 0, 0.25]), fovy=48)],
        option=dict(timestep=0.002, integrator="implicitfast"),
    )
    root = ET.parse(scene).getroot()
    ET.SubElement(root.find("option"), "flag", warmstart="disable")
    ET.ElementTree(root).write(scene, encoding="unicode")
    return scene


def plan_controls(model, start, target, size, fps=20):
    # Each interval uses a constant setpoint so the saved actuator controls can
    # reproduce every interval exactly. Smooth interpolation occurs across frames.
    z, lift = size[2] / 2, max(0.12, size[2] * 2)
    above_start, grasp = [*start, z + lift], [*start, z]
    above_target, place = [*target, z + lift], [*target, z + 0.002]
    orientation = np.diag([1.0, -1.0, -1.0])
    seed = np.array([0, -0.5, 0, -2.0, 0, 1.5, 0.7854])
    q = solve_ik(model, above_start, orientation, seed)
    initial = model.qpos0.copy()
    initial[:7] = q
    keyframes = [
        (1.0, above_start, 0),
        (2.0, grasp, 0),
        (1.2, grasp, 255),
        (2.0, above_start, 255),
        (2.0, above_target, 255),
        (2.0, place, 255),
        (1.2, place, 0),
        (2.0, above_target, 0),
        (1.0, above_target, 0),
    ]
    rows, phases, last_xyz, last_grip = [], [], np.array(above_start), 0.0
    for index, (seconds, xyz, grip) in enumerate(keyframes):
        for j in range(round(seconds * fps)):
            t = (j + 1) / round(seconds * fps)
            blend = t * t * (3 - 2 * t)
            position = last_xyz * (1 - blend) + np.array(xyz) * blend
            q = solve_ik(model, position, orientation, q)
            rows.append(np.r_[q, last_grip * (1 - blend) + grip * blend])
            phases.append(index)
        last_xyz, last_grip = np.array(xyz), grip
    stride = round(1 / fps / model.opt.timestep)
    if abs(stride * model.opt.timestep * fps - 1) > 1e-8:
        raise ValueError("Choose fps that evenly divides the physics step rate")
    return initial, np.repeat(np.array(rows), stride, axis=0), np.array(phases), len(rows) + 1


def verify_rollout(out, controls, initial, target, size):
    model, data = engine.load(Path(out) / "scene.mjb")
    saved = np.load(Path(out) / "trajectory.npz")
    data.qpos[:] = initial
    data.ctrl[:] = controls[0]
    mujoco.mj_forward(model, data)
    positions, speeds, errors, velocity_errors = [], [], [], []
    times = np.rint(saved["time"] / model.opt.timestep).astype(int)
    previous = 0
    for frame, step in enumerate(times):
        for control in controls[previous:step]:
            data.ctrl[:] = control
            mujoco.mj_step(model, data)
        previous = step
        errors.append(float(np.max(np.abs(data.qpos - saved["qpos"][frame]))))
        velocity_errors.append(float(np.max(np.abs(data.qvel - saved["qvel"][frame]))))
        positions.append(data.body("prop").xpos.copy())
        speeds.append(float(np.linalg.norm(data.qvel[-6:-3])))
    positions = np.array(positions)
    horizontal_error = float(np.linalg.norm(positions[-1, :2] - target))
    height_error = float(abs(positions[-1, 2] - size[2] / 2))
    result = dict(
        max_replay_qpos_error=max(errors),
        max_lift_m=float(positions[:, 2].max() - positions[0, 2]),
        max_replay_qvel_error=max(velocity_errors),
        final_xy_error_m=horizontal_error,
        final_height_error_m=height_error,
        final_speed_m_s=speeds[-1],
        final_position_m=positions[-1].tolist(),
        checked_frames=len(times),
        mocap_bodies=int(model.nmocap),
        prop_has_free_joint=bool(model.jnt_type[model.body("prop").jntadr[0]] == mujoco.mjtJoint.mjJNT_FREE),
        replay_method="native physics from initial state and every saved per-step actuator control",
    )
    result["motion_success"] = bool(
        result["max_lift_m"] > 0.06
        and horizontal_error < 0.015
        and height_error < 0.004
        and speeds[-1] < 0.01
        and max(errors + velocity_errors) < 1e-7
    )
    np.savez_compressed(Path(out).parent / "object_track.npz", time=saved["time"], position=positions)
    return result


def run_manipulation(out, start=(0.5, -0.08), target=(0.5, 0.08), size=(0.035, 0.035, 0.05)):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    scene = build_scene(out, start, target, size)
    model, _ = engine.load(scene)
    initial, controls, phases, frames = plan_controls(model, start, target, size)
    np.savez_compressed(out / "controls.npz", ctrl=controls, initial_qpos=initial, phase_per_interval=phases)
    report = engine.rollout(
        scene,
        out / "rollout",
        controls,
        initial={"qpos": initial},
        fps=20,
        frames=frames,
        camera="overview",
        size=(640, 480),
        video=True,
        preview=6,
    )
    result = verify_rollout(out / "rollout", controls, initial, np.asarray(target), size)
    pad_frames = sum(
        any(
            "prop_geom" in c["geoms"] and any("pad" in g for g in c["geoms"]) and c["normal_force_N"] > 0
            for c in frame
        )
        for frame in report["contact_frames"]
    )
    positions = np.load(out / "object_track.npz")["position"]
    bilateral_lift_frames = 0
    for position, contacts in zip(positions, report["contact_frames"]):
        touching = {
            g for c in contacts if "prop_geom" in c["geoms"] and c["normal_force_N"] > 0 for g in c["geoms"]
        }
        bilateral_lift_frames += int(
            position[2] > size[2] / 2 + 0.06
            and any(g.startswith("left_pad") for g in touching)
            and any(g.startswith("right_pad") for g in touching)
        )
    final_contacts = report["contact_frames"][-1]
    released_on_table = any(
        set(c["geoms"]) == {"prop_geom", "table_geom"} and c["normal_force_N"] > 0 for c in final_contacts
    ) and not any("prop_geom" in c["geoms"] and any("pad" in g for g in c["geoms"]) for c in final_contacts)
    result.update(
        pad_contact_frames=pad_frames,
        warnings=report["warnings"],
        bilateral_lift_contact_frames=bilateral_lift_frames,
        released_on_table=released_on_table,
        max_penetration_m=report["max_penetration_m"],
        scope="known-scene box pick/place with upstream Panda/Robotiq; not video-to-policy or chess grasping",
        physical_priors=dict(mass_kg=0.04, object_friction=0.8),
        prop_attachments=False,
        implementation="CPU IK targets + upstream native actuator rollout",
    )
    result["passed"] = bool(
        result["motion_success"]
        and bilateral_lift_frames > 0
        and released_on_table
        and not report["warnings"]
        and report["max_penetration_m"] < 0.005
    )
    write_json(out / "verification.json", result)
    return result
