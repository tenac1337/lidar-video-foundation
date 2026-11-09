"""Measure saved free-body motion independently of deterministic replay checks."""

import mujoco
import numpy as np


def measure_passive_motion(model, saved, object_names):
    """Report movement from each body's initial pose; do not certify stability.

    Peaks are over saved samples, not continuous-time extrema. Quaternion signs
    are equivalent. A stationary object can still be floating or unsupported.
    """
    time = np.asarray(saved["time"], float)
    position, velocity = np.asarray(saved["qpos"], float), np.asarray(saved["qvel"], float)
    if (
        time.ndim != 1
        or len(time) < 2
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
        or position.shape != (len(time), model.nq)
        or velocity.shape != (len(time), model.nv)
        or not np.isfinite(position).all()
        or not np.isfinite(velocity).all()
    ):
        raise ValueError("Passive-motion audit requires finite, increasing saved states")
    if len(set(object_names)) != len(object_names):
        raise ValueError("Duplicate object names")
    objects = {}
    for name in object_names:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            raise ValueError(f"Unknown body: {name}")
        joint = model.body_jntadr[body]
        if model.body_jntnum[body] != 1 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Passive-motion audit requires a free body: {name}")
        start, dof = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        poses, speeds = position[:, start : start + 7], velocity[:, dof : dof + 6]
        norm = np.linalg.norm(poses[:, 3:7], axis=1)
        if np.any(norm < 1e-12):
            raise ValueError("Invalid free-body quaternion")
        quaternion = poses[:, 3:7] / norm[:, None]
        angle = np.rad2deg(2 * np.arccos(np.clip(np.abs(quaternion @ quaternion[0]), 0, 1)))
        displacement = poses[:, :3] - poses[0, :3]
        distances = np.linalg.norm(displacement, axis=1)
        objects[name] = dict(
            final_displacement_xyz_m=displacement[-1].tolist(),
            final_translation_m=float(distances[-1]),
            max_sampled_translation_m=float(distances.max()),
            final_rotation_deg=float(angle[-1]),
            max_sampled_rotation_deg=float(angle.max()),
            final_linear_speed_m_s=float(np.linalg.norm(speeds[-1, :3])),
            final_angular_speed_deg_s=float(np.rad2deg(np.linalg.norm(speeds[-1, 3:]))),
        )
    return dict(
        duration_s=float(time[-1] - time[0]),
        sampled_frames=len(time),
        objects=objects,
        scope="Saved free-body motion; no stability, support-contact or task-success certification",
    )
