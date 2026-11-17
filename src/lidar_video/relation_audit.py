"""Sampled geometric relation checks on native poses, separate from replay.

Rotated declared body-local envelopes are conservative geometry proxies. Native
contact candidates establish proximity, not load bearing or force closure.
Unknown semantic edges are explicitly unchecked, never silently certified.
"""

from itertools import product

import mujoco
import numpy as np

from lidar_video.scene_program import RELATIONS, as_networkx


def audit_scene_relations(
    model,
    saved,
    graph,
    *,
    position_tolerance_m=0.001,
    contact_tolerance_m=0.001,
    start_time_s=0.0,
    spatial_contract=None,
):
    """Check every selected saved pose without stepping or changing the source.

    `on` checks containment/contact in the parent's frame; directional edges use
    world axes. `inside` checks envelope containment, not a hollow collision
    model. Envelope centers must coincide with their named native body origins.
    Contact checks inspect direct named-body geom pairs, not articulated trees.
    Times are absolute trajectory times, with an explicit optional lower bound.
    """
    for value in (position_tolerance_m, contact_tolerance_m, start_time_s):
        if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError("Finite nonnegative tolerances and start time required")
    if model.nmocap or model.nflex:
        raise ValueError("Pose-only relation audit does not restore mocap or deformable state")
    time, qpos = np.asarray(saved["time"], float), np.asarray(saved["qpos"], float)
    if (
        time.ndim != 1
        or not len(time)
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
        or qpos.shape != (len(time), model.nq)
        or not np.isfinite(qpos).all()
    ):
        raise ValueError("Finite increasing native pose samples required")
    selected = np.flatnonzero(time >= start_time_s)
    if not len(selected):
        raise ValueError("No samples at or after requested start time")
    for joint in range(model.njnt):
        kind, address = model.jnt_type[joint], model.jnt_qposadr[joint]
        offset = 3 if kind == mujoco.mjtJoint.mjJNT_FREE else 0
        if kind in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            if not np.allclose(
                np.linalg.norm(qpos[:, address + offset : address + offset + 4], axis=1), 1, atol=1e-6
            ):
                raise ValueError("Native pose quaternions must be normalized")
    network = as_networkx(graph)
    physical = {n["id"]: n for n in graph["nodes"] if n.get("kind") in {"object", "support"}}
    bodies, corners = {}, {}
    signs = np.array(list(product([-1, 1], repeat=3)))
    for name, node in physical.items():
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        size = np.asarray(node["size_m"], float)
        if body <= 0 or size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError(f"Named native body and positive envelope required: {name}")
        bodies[name], corners[name] = body, signs * size / 2
    regions, footprints, separation = {}, {}, []
    if spatial_contract is not None:
        if spatial_contract.get("version") != 1 or spatial_contract.get("physical_envelopes") != {
            n: list(node["size_m"]) for n, node in physical.items()
        }:
            raise ValueError("Spatial contract must match the physical body envelopes")
        regions = spatial_contract["regions"]
        footprints = spatial_contract["footprints"]
        for name, region in regions.items():
            if name not in network or region["parent"] not in bodies:
                raise ValueError("Invalid region binding")
            bounds = np.asarray(region["local_bounds_xy_m"], float)
            if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
                raise ValueError("Invalid region bounds")
            for field, shape in [
                ("reference_parent_position_m", (3,)),
                ("reference_parent_rotation", (3, 3)),
            ]:
                value = np.asarray(region[field], float)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError("Invalid region reference transform")
            if not np.isfinite(region["local_z_m"]):
                raise ValueError("Invalid region height")
        for key, footprint in footprints.items():
            index = int(key)
            if not 0 <= index < len(graph["edges"]):
                raise ValueError("Invalid footprint edge index")
            edge = graph["edges"][index]
            if (
                edge["relation"] != "occupies"
                or footprint["source"] != edge["source"]
                or footprint["target"] != edge["target"]
                or edge["source"] not in bodies
                or edge["target"] not in regions
            ):
                raise ValueError("Invalid footprint binding")
            points = np.asarray(footprint["local_points_m"], float)
            if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
                raise ValueError("Finite 3D footprint points required")
        for witness in spatial_contract["separation_witnesses"]:
            if (
                witness["source"] not in bodies
                or witness["target"] not in bodies
                or witness["source"] == witness["target"]
                or witness["axis"] not in ("x", "y", "z")
                or witness["sign"] not in (-1, 1)
            ):
                raise ValueError("Invalid separation witness")
            penetration = witness.get("support_penetration_m", 0)
            if not np.isfinite(penetration) or penetration < 0:
                raise ValueError("Invalid separation overlap allowance")
            separation.append(
                dict(witness, violated_samples=0, first_violation_time_s=None, max_geometric_residual_m=0.0)
            )
    checked, unchecked = [], []
    for index, edge in enumerate(graph["edges"]):
        a, b, relation = edge["source"], edge["target"], edge["relation"]
        semantic = (
            relation == "occupies"
            and str(index) in footprints
            or relation == "part_of"
            and a in regions
            and regions[a]["parent"] == b
        )
        if not semantic and (relation not in RELATIONS or a not in bodies or b not in bodies):
            unchecked.append(
                dict(
                    edge_index=index,
                    source=a,
                    target=b,
                    relation=relation,
                    reason="Not a supported physical-envelope relation",
                )
            )
            continue
        if a == b:
            raise ValueError("Self spatial relations are unsupported")
        for field in ("gap_m", "clearance_m", "support_penetration_m"):
            value = edge.get(field, 0)
            if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value < 0:
                raise ValueError("Relation offsets must be finite and nonnegative")
        if edge.get("clearance_m", 0) and edge.get("support_penetration_m", 0):
            raise ValueError("Support clearance and penetration cannot both be positive")
        checked.append(
            dict(
                edge_index=index,
                source=a,
                target=b,
                relation=relation,
                violated_samples=0,
                first_violation_time_s=None,
                max_geometric_residual_m=0.0,
                missing_contact_samples=0,
            )
        )
        if semantic:
            checked[-1]["evidence"] = (
                "Asset-derived band or envelope in calibrated region"
                if relation == "occupies"
                else regions[a]["evidence"]
            )
        if relation == "on":
            same_rigid_body = model.body_weldid[bodies[a]] == model.body_weldid[bodies[b]]
            checked[-1]["contact_check"] = (
                "not_applicable_declared_clearance"
                if edge.get("clearance_m", 0) > 0
                else "not_applicable_rigidly_connected"
                if same_rigid_body
                else "native_contact_candidate_required"
            )
    data = mujoco.MjData(model)
    inverse_bodies = {bid: name for name, bid in bodies.items()}
    directions = dict(
        left_of=(0, -1), right_of=(0, 1), behind=(1, -1), in_front_of=(1, 1), below=(2, -1), above=(2, 1)
    )
    for frame in selected:
        data.qpos[:] = qpos[frame]
        data.time = time[frame]
        # Only geometry is read. No saved velocity/control/force is reconstructed.
        mujoco.mj_fwdPosition(model, data)
        world = {n: corners[n] @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid] for n, bid in bodies.items()}
        nearby = set()
        for contact in data.contact:
            pair = [inverse_bodies.get(int(model.geom_bodyid[g])) for g in (contact.geom1, contact.geom2)]
            if None not in pair and contact.dist <= contact_tolerance_m:
                nearby.add(frozenset(pair))
        for record in checked:
            edge = graph["edges"][record["edge_index"]]
            a, b, relation = record["source"], record["target"], record["relation"]
            missing = False
            if relation == "part_of":
                region = regions[a]
                bounds = np.asarray(region["local_bounds_xy_m"])
                half = np.asarray(physical[b]["size_m"]) / 2
                local = np.array([[x, y, region["local_z_m"]] for x in bounds[:, 0] for y in bounds[:, 1]])
                current = local @ data.xmat[bodies[b]].reshape(3, 3).T + data.xpos[bodies[b]]
                reference = (
                    local @ np.asarray(region["reference_parent_rotation"]).T
                    + region["reference_parent_position_m"]
                )
                residual = max(
                    float(np.max(-half[:2] - bounds[0])),
                    float(np.max(bounds[1] - half[:2])),
                    abs(region["local_z_m"] - half[2]),
                    float(np.max(np.abs(current - reference))),
                    0,
                )
            elif relation == "occupies":
                region = regions[b]
                parent = bodies[region["parent"]]
                points = np.asarray(footprints[str(record["edge_index"])]["local_points_m"])
                points = points @ data.xmat[bodies[a]].reshape(3, 3).T + data.xpos[bodies[a]]
                local = (points - data.xpos[parent]) @ data.xmat[parent].reshape(3, 3)
                bounds = np.asarray(region["local_bounds_xy_m"])
                residual = max(
                    float(np.max(bounds[0] - local[:, :2].min(axis=0))),
                    float(np.max(local[:, :2].max(axis=0) - bounds[1])),
                    0,
                )
            elif relation in {"on", "inside"}:
                parent = bodies[b]
                local = (world[a] - data.xpos[parent]) @ data.xmat[parent].reshape(3, 3)
                low, high = local.min(axis=0), local.max(axis=0)
                half = np.asarray(physical[b]["size_m"]) / 2
                if relation == "inside":
                    gap = edge.get("gap_m", 0)
                    residual = max(float(np.max(-half + gap - low)), float(np.max(high - half + gap)), 0)
                else:
                    expected = edge.get("clearance_m", 0) - edge.get("support_penetration_m", 0)
                    residual = max(
                        float(np.max(-half[:2] - low[:2])),
                        float(np.max(high[:2] - half[:2])),
                        abs(float(low[2] - half[2]) - expected),
                        0,
                    )
                    # Same-weld bodies are structurally connected, not a dynamic
                    # contact pair. Positive clearance also does not promise contact.
                    missing = (
                        record["contact_check"] == "native_contact_candidate_required"
                        and frozenset((a, b)) not in nearby
                    )
            else:
                axis, sign = directions[relation]
                # Positive sign means source's low face lies beyond target's high face.
                low_side, high_side = (b, a) if sign > 0 else (a, b)
                residual = max(
                    float(
                        world[low_side][:, axis].max()
                        + edge.get("gap_m", 0)
                        - world[high_side][:, axis].min()
                    ),
                    0,
                )
            record["max_geometric_residual_m"] = max(record["max_geometric_residual_m"], residual)
            record["missing_contact_samples"] += int(missing)
            if residual > position_tolerance_m or missing:
                record["violated_samples"] += 1
                if record["first_violation_time_s"] is None:
                    record["first_violation_time_s"] = float(time[frame])
        extents = {n: (v.min(axis=0), v.max(axis=0)) for n, v in world.items()} if separation else {}
        for record in separation:
            a, b = record["source"], record["target"]
            axis = "xyz".index(record["axis"])
            low_side, high_side = (a, b) if record["sign"] == 1 else (b, a)
            residual = max(
                float(extents[low_side][1][axis] - extents[high_side][0][axis])
                - record.get("support_penetration_m", 0),
                0,
            )
            record["max_geometric_residual_m"] = max(record["max_geometric_residual_m"], residual)
            if residual > position_tolerance_m:
                record["violated_samples"] += 1
                if record["first_violation_time_s"] is None:
                    record["first_violation_time_s"] = float(time[frame])
    for record in checked:
        record["preserved_at_all_checked_samples"] = record["violated_samples"] == 0
    for record in separation:
        record["preserved_at_all_checked_samples"] = record["violated_samples"] == 0
    return dict(
        sampled_frames=len(selected),
        start_time_s=float(time[selected[0]]),
        end_time_s=float(time[selected[-1]]),
        position_tolerance_m=position_tolerance_m,
        contact_tolerance_m=contact_tolerance_m,
        checked_edges=checked,
        unchecked_edges=unchecked,
        graph_edges=network.number_of_edges(),
        all_checked_relations_preserved=bool(checked)
        and all(r["preserved_at_all_checked_samples"] for r in checked),
        complete_edge_coverage=bool(checked) and not unchecked,
        separation_audit=dict(
            supplied=spatial_contract is not None,
            witnesses=separation,
            all_witnesses_preserved=all(r["preserved_at_all_checked_samples"] for r in separation)
            if spatial_contract is not None
            else None,
            scope="Frozen world-axis separation branches of rotated envelopes; branch failure is not proof of mesh collision",
        ),
        scope="Sampled rotated-envelope geometry and direct-body native contact candidates; no load-bearing, force-closure, task-success or continuous-time guarantee",
    )
