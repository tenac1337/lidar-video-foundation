import copy

import mujoco
import numpy as np
import pytest

from lidar_video.relation_audit import audit_scene_relations


def example():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="table" pos="0 0 -.05"><geom type="box" size=".5 .5 .05"/></body>
      <body name="block" pos="0 0 .05"><freejoint/><geom type="box" size=".1 .05 .05" mass=".1"/></body>
      <body name="other" pos=".3 0 .05"><freejoint/><geom type="box" size=".05 .05 .05" mass=".1"/></body>
    </worldbody></mujoco>""")
    graph = dict(
        nodes=[
            dict(id="table", kind="support", size_m=[1, 1, 0.1]),
            dict(id="block", kind="object", size_m=[0.2, 0.1, 0.1]),
            dict(id="other", kind="object", size_m=[0.1, 0.1, 0.1]),
        ],
        edges=[
            dict(source="block", target="table", relation="on"),
            dict(source="block", target="other", relation="left_of", gap_m=0.1),
        ],
    )
    saved = dict(time=np.array([0, 1.0, 2.0]), qpos=np.tile(model.qpos0, (3, 1)))
    return model, saved, graph


def test_supported_scene_and_input_immutability():
    m, s, g = example()
    original = copy.deepcopy(s)
    r = audit_scene_relations(m, s, g)
    assert r["all_checked_relations_preserved"] and r["complete_edge_coverage"]
    assert len(r["checked_edges"]) == 2
    np.testing.assert_array_equal(original["qpos"], s["qpos"])


def test_transient_relation_loss_not_hidden_by_final_recovery():
    m, s, g = example()
    s["qpos"][1, 0] = 0.6
    r = audit_scene_relations(m, s, g)
    assert not r["all_checked_relations_preserved"]
    for edge in r["checked_edges"]:
        assert edge["violated_samples"] == 1 and edge["first_violation_time_s"] == 1
    assert r["checked_edges"][0]["missing_contact_samples"] == 1


def test_geometrically_near_but_floating_fails_contact_check():
    m, s, g = example()
    s["qpos"][:, 2] += 0.0005
    r = audit_scene_relations(m, s, g, position_tolerance_m=0.001)
    edge = r["checked_edges"][0]
    assert edge["max_geometric_residual_m"] < 0.001
    assert edge["missing_contact_samples"] == 3
    assert not edge["preserved_at_all_checked_samples"]


def test_clearance_explicitly_does_not_require_contact():
    m, s, g = example()
    s["qpos"][:, 2] += 0.01
    g["edges"][0]["clearance_m"] = 0.01
    assert audit_scene_relations(m, s, g)["all_checked_relations_preserved"]


def test_rotated_envelope_detects_edge_overhang():
    m, s, g = example()
    s["qpos"][:, 1] = 0.43
    s["qpos"][1, 3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    r = audit_scene_relations(m, s, g)
    edge = r["checked_edges"][0]
    assert edge["violated_samples"] == 1
    assert edge["max_geometric_residual_m"] == pytest.approx(0.03)


def test_unknown_semantic_relations_are_unchecked_not_passed():
    m, s, g = example()
    g["edges"].append(dict(source="block", target="other", relation="near"))
    r = audit_scene_relations(m, s, g)
    assert r["all_checked_relations_preserved"] and not r["complete_edge_coverage"]
    assert r["unchecked_edges"][0]["relation"] == "near"


@pytest.mark.parametrize(
    "relation,axis,sign",
    [
        ("left_of", 0, -1),
        ("right_of", 0, 1),
        ("behind", 1, -1),
        ("in_front_of", 1, 1),
        ("below", 2, -1),
        ("above", 2, 1),
    ],
)
def test_all_directional_conventions(relation, axis, sign):
    model, saved, graph = example()
    graph["edges"] = [dict(source="block", target="other", relation=relation, gap_m=0.01)]
    for i in range(3):
        saved["qpos"][i, :3] = saved["qpos"][i, 7:10]
        saved["qpos"][i, axis] += sign * 0.4
    assert audit_scene_relations(model, saved, graph)["all_checked_relations_preserved"]
    saved["qpos"][1, axis] -= sign * 0.8
    assert audit_scene_relations(model, saved, graph)["checked_edges"][0]["violated_samples"] == 1


def test_inside_is_geometry_only_and_direction_uses_world_axes():
    m, s, g = example()
    g["nodes"][0]["size_m"] = [1, 1, 1]
    g["edges"][0]["relation"] = "inside"
    assert audit_scene_relations(m, s, g)["all_checked_relations_preserved"]
    s["qpos"][1, 2] = 1
    assert audit_scene_relations(m, s, g)["checked_edges"][0]["violated_samples"] == 1


def test_rigidly_connected_bodies_check_geometry_without_demanding_dynamic_contact():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="table" pos="0 0 -.05"><geom type="box" size=".5 .5 .05"/></body>
      <body name="mount" pos="0 0 .05"><geom type="box" size=".05 .05 .05"/></body>
    </worldbody></mujoco>""")
    graph = dict(
        nodes=[
            dict(id="table", kind="support", size_m=[1, 1, 0.1]),
            dict(id="mount", kind="support", size_m=[0.1, 0.1, 0.1]),
        ],
        edges=[dict(source="mount", target="table", relation="on")],
    )
    saved = dict(time=[0.0, 1.0], qpos=np.empty((2, 0)))
    report = audit_scene_relations(model, saved, graph)
    assert report["all_checked_relations_preserved"]
    assert report["checked_edges"][0]["contact_check"] == "not_applicable_rigidly_connected"
    model.body_pos[model.body("mount").id, 0] = 1
    assert not audit_scene_relations(model, saved, graph)["all_checked_relations_preserved"]


@pytest.mark.parametrize("issue", ["nan", "quaternion", "time", "missing", "empty", "tolerance"])
def test_bad_inputs_fail_explicitly(issue):
    m, s, g = example()
    options = {}
    if issue == "nan":
        s["qpos"][0, 0] = np.nan
    if issue == "quaternion":
        s["qpos"][0, 3:7] = 0
    if issue == "time":
        s["time"][1] = 0
    if issue == "missing":
        g["nodes"][0]["id"] = "absent"
        g["edges"][0]["target"] = "absent"
    if issue == "empty":
        options["start_time_s"] = 3
    if issue == "tolerance":
        options["position_tolerance_m"] = -1
    with pytest.raises(ValueError):
        audit_scene_relations(m, s, g, **options)
