import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import trimesh

from lidar_video.relation_audit import audit_scene_relations
from lidar_video.spatial_contract import prepare_spatial_contract


def fixture(tmp_path):
    mesh = trimesh.creation.cone(radius=0.09, height=0.1, sections=4)
    mesh.vertices[:, 2] = 0.05 - mesh.vertices[:, 2]
    mesh.export(tmp_path / "pyramid.obj")
    graph = dict(
        nodes=[
            dict(
                id="board", kind="support", position_m=[0, 0, -0.05], size_m=[1, 1, 0.1], square_size_m=0.08
            ),
            dict(id="square", kind="square", position_m=[0, 0, 0]),
            dict(
                id="piece",
                kind="object",
                position_m=[0, 0, 0.05],
                size_m=[0.18, 0.18, 0.1],
                mesh_path="pyramid.obj",
                collision_paths=["pyramid.obj"],
            ),
            dict(id="other", kind="object", position_m=[0.3, 0, 0.05], size_m=[0.08, 0.08, 0.1]),
        ],
        edges=[
            dict(source="square", target="board", relation="part_of"),
            dict(
                source="piece",
                target="square",
                relation="occupies",
                footprint_mode="support_band",
                support_band_height_m=0.03,
            ),
            dict(source="piece", target="board", relation="on"),
            dict(source="other", target="board", relation="on"),
        ],
    )
    root = ET.fromstring("""<mujoco><asset><mesh name="pyramid"/></asset><worldbody>
      <body name="board" pos="0 0 -.05"><geom type="box" size=".5 .5 .05"/></body>
      <body name="piece" pos="0 0 .05"><freejoint/><geom type="mesh" mesh="pyramid" mass=".1"/></body>
      <body name="other" pos=".3 0 .05"><freejoint/><geom type="box" size=".04 .04 .05" mass=".1"/></body>
    </worldbody></mujoco>""")
    root.find("asset/mesh").set("file", str(tmp_path / "pyramid.obj"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    saved = dict(time=[0.0, 1.0, 2.0], qpos=np.tile(model.qpos0, (3, 1)))
    return model, saved, graph, prepare_spatial_contract(graph, tmp_path)


def test_band_uses_clipped_triangle_edges_and_excludes_upper_overhang(tmp_path):
    m, s, g, c = fixture(tmp_path)
    points = np.asarray(c["footprints"]["1"]["local_points_m"])
    assert np.max(points[:, 2]) == pytest.approx(-0.02)
    assert np.max(points[:, 0]) == pytest.approx(0.027)
    r = audit_scene_relations(m, s, g, spatial_contract=c)
    assert r["complete_edge_coverage"]
    assert r["checked_edges"][1]["preserved_at_all_checked_samples"]
    assert r["separation_audit"]["all_witnesses_preserved"]


def test_rotation_and_transient_membership_loss_are_detected(tmp_path):
    m, s, g, c = fixture(tmp_path)
    s["qpos"][1, 3:7] = [np.sqrt(0.5), 0, np.sqrt(0.5), 0]
    r = audit_scene_relations(m, s, g, spatial_contract=c)
    edge = r["checked_edges"][1]
    assert edge["violated_samples"] == 1 and edge["first_violation_time_s"] == 1
    assert edge["max_geometric_residual_m"] == pytest.approx(0.01)
    assert edge["preserved_at_all_checked_samples"] is False


def test_frozen_separation_branch_does_not_silently_switch_axes(tmp_path):
    m, s, g, c = fixture(tmp_path)
    s["qpos"][1, 7:10] = [0, 0.3, 0.05]  # Still separated in Y; reference X branch is lost.
    r = audit_scene_relations(m, s, g, spatial_contract=c)
    witnesses = r["separation_audit"]["witnesses"]
    pair = next(w for w in witnesses if {w["source"], w["target"]} == {"piece", "other"})
    assert pair["violated_samples"] == 1 and pair["max_geometric_residual_m"] > 0
    assert not r["separation_audit"]["all_witnesses_preserved"]


def test_fixed_calibrated_region_drift_and_bounds_are_checked(tmp_path):
    m, s, g, c = fixture(tmp_path)
    m.body_pos[m.body("board").id, 0] = 0.02
    r = audit_scene_relations(m, s, g, spatial_contract=c)
    assert r["checked_edges"][0]["violated_samples"] == 3
    assert r["checked_edges"][0]["max_geometric_residual_m"] == pytest.approx(0.02)


@pytest.mark.parametrize("change", ["size", "points", "binding", "witness"])
def test_invalid_contract_rejected(tmp_path, change):
    m, s, g, c = fixture(tmp_path)
    c = copy.deepcopy(c)
    if change == "size":
        c["physical_envelopes"]["piece"] = [1, 1, 1]
    if change == "points":
        c["footprints"]["1"]["local_points_m"] = [[np.nan, 0, 0]]
    if change == "binding":
        c["footprints"]["1"]["target"] = "missing"
    if change == "witness":
        c["separation_witnesses"][0]["source"] = "missing"
    with pytest.raises(ValueError):
        audit_scene_relations(m, s, g, spatial_contract=c)
