import copy

import mujoco
import numpy as np
import pytest

from lidar_video.passive_motion import measure_passive_motion


@pytest.fixture
def states():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="fixed"><geom type="box" size="1 1 .1"/></body>
      <body name="piece" pos="0 0 2"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
    </worldbody></mujoco>""")
    saved = dict(
        time=np.array([0.0, 0.5, 1.0]), qpos=np.tile(model.qpos0, (3, 1)), qvel=np.zeros((3, model.nv))
    )
    return model, saved


def test_reports_transient_peak_separately_from_final_pose(states):
    model, saved = states
    saved["qpos"][1, :3] += [0.003, 0.004, 0]
    saved["qpos"][2, :3] += [0.001, 0, 0]
    saved["qpos"][1, 3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    saved["qpos"][2, 3:7] *= -1  # Same orientation, opposite quaternion sign.
    saved["qvel"][-1, :3] = [0, 0.03, 0.04]
    saved["qvel"][-1, 3:] = [0, 0, np.pi / 2]
    report = measure_passive_motion(model, saved, ["piece"])
    obj = report["objects"]["piece"]
    assert obj["max_sampled_translation_m"] == pytest.approx(0.005)
    assert obj["final_translation_m"] == pytest.approx(0.001)
    assert obj["max_sampled_rotation_deg"] == pytest.approx(90)
    assert obj["final_rotation_deg"] == pytest.approx(0)
    assert obj["final_linear_speed_m_s"] == pytest.approx(0.05)
    assert obj["final_angular_speed_deg_s"] == pytest.approx(90)
    assert report["sampled_frames"] == 3 and report["duration_s"] == 1


def test_no_stability_claim_for_motionless_floating_object(states):
    model, saved = states
    report = measure_passive_motion(model, saved, ["piece"])
    assert report["objects"]["piece"]["max_sampled_translation_m"] == 0
    assert "no stability" in report["scope"]
    assert "passed" not in report and "stable" not in report


@pytest.mark.parametrize(
    "problem", ["time", "nonfinite", "shape", "quaternion", "unknown", "fixed", "duplicate"]
)
def test_invalid_motion_inputs_fail_explicitly(states, problem):
    model, original = states
    saved = copy.deepcopy(original)
    names = ["piece"]
    if problem == "time":
        saved["time"][1] = saved["time"][0]
    if problem == "nonfinite":
        saved["qpos"][1, 0] = np.nan
    if problem == "shape":
        saved["qvel"] = saved["qvel"][:-1]
    if problem == "quaternion":
        saved["qpos"][1, 3:7] = 0
    if problem == "unknown":
        names = ["missing"]
    if problem == "fixed":
        names = ["fixed"]
    if problem == "duplicate":
        names = ["piece", "piece"]
    with pytest.raises(ValueError):
        measure_passive_motion(model, saved, names)
