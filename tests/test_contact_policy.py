import copy
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from lidar_video.contact_policy import configure_mesh_contacts


def example():
    return ET.fromstring(
        """<mujoco><option timestep=".002"><flag warmstart="disable"/></option><asset><mesh name="cube" vertex="-.01 -.01 -.01 .01 -.01 -.01 -.01 .01 -.01 .01 .01 -.01 -.01 -.01 .01 .01 -.01 .01 -.01 .01 .01 .01 .01 .01"/></asset><worldbody><geom name="table" type="box" pos="0 0 -.005" size="1 1 .005" friction=".8 .005 .0001"/><body name="object" pos="0 0 .01"><freejoint/><geom name="visual" type="mesh" mesh="cube" contype="0" conaffinity="0" mass="0"/><geom name="collision" type="mesh" mesh="cube" mass=".08" friction=".8 .005 .0001"/></body></worldbody></mujoco>"""
    )


def test_explicit_profile_preserves_geometry_mass_friction_and_joint_state():
    mujoco = pytest.importorskip("mujoco")
    root = example()
    before = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    original = copy.deepcopy(root)
    record = configure_mesh_contacts(root)
    after = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for attribute in [
        "body_mass",
        "body_inertia",
        "body_pos",
        "body_ipos",
        "mesh_vert",
        "mesh_face",
        "geom_friction",
        "jnt_type",
        "qpos0",
    ]:
        np.testing.assert_array_equal(getattr(before, attribute), getattr(after, attribute))
    assert after.opt.enableflags & mujoco.mjtEnableBit.mjENBL_MULTICCD
    assert after.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD
    assert after.opt.noslip_iterations == 3
    assert record["active_collision_geoms"] == 2
    assert root.find('.//geom[@name="visual"]').attrib == original.find('.//geom[@name="visual"]').attrib
    assert after.geom_condim[after.geom("collision").id] == 6


def test_passive_mesh_stays_supported_but_moves_under_a_real_force():
    mujoco = pytest.importorskip("mujoco")
    root = example()
    configure_mesh_contacts(root)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = model.body("object").id
    initial = data.xpos[bid].copy()
    for _ in range(1000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    assert np.linalg.norm(data.xpos[bid] - initial) < 0.001
    for _ in range(50):
        data.xfrc_applied[bid, 0] = 1.0
        mujoco.mj_step(model, data)
    data.xfrc_applied[:] = 0
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    assert data.xpos[bid, 0] - initial[0] > 0.005
    assert (
        model.body_jntnum[bid] == 1 and model.jnt_type[model.body_jntadr[bid]] == mujoco.mjtJoint.mjJNT_FREE
    )


@pytest.mark.parametrize("tag", ["default", "include", "contact/pair"])
def test_implicit_models_reject_before_mutation(tag):
    root = example()
    parent = root
    for name in tag.split("/"):
        parent = ET.SubElement(parent, name)
    before = ET.tostring(root)
    with pytest.raises(ValueError):
        configure_mesh_contacts(root)
    assert ET.tostring(root) == before
