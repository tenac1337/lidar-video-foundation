"""Board pose/scale recovery without supplied board geometry or piece labels."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lidar_video.board_calibration import calibrate, detect_frame, project
from lidar_video.capture import load_capture
from lidar_video.chess_scene import board_from_observation
from lidar_video.reconstruction import reconstruct

ROOT = Path(__file__).resolve().parent / "fixtures"


def capture(name="chess-small"):
    return load_capture(ROOT / name / "capture.npz")


@pytest.mark.parametrize("name", ["chess-small", "chess-demo"])
def test_grid_pose_and_scale_recovered_from_images_and_depth(name):
    result = calibrate(capture(name))
    world = np.linalg.inv(result["board_from_world"])
    assert abs(result["square_size_m"] - 0.045) < 0.0003
    assert np.linalg.norm(world[:3, 3]) < 0.001
    assert np.dot(world[:3, 2], [0, 0, 1]) > 0.999
    assert result["checked_frames"] == [0, 1, 2]
    assert not result["orientation"]["resolved"]


def test_nontrivial_world_transform_and_explicit_a1_annotation():
    data = capture()
    # Only a corner-square annotation is supplied, not a board pose or size.
    pixel = project([[-3.5 * 0.045, -3.5 * 0.045, 0]], data["world_from_camera"][0], data["K"])[0]
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.3, 0.7]).as_matrix()
    world[:3, 3] = [0.3, -0.8, 0.15]
    data["world_from_camera"] = world @ data["world_from_camera"]
    result = calibrate(data, frame=0, a1_pixel=pixel)
    recovered = np.linalg.inv(result["board_from_world"])
    np.testing.assert_allclose(recovered[:3, :3], world[:3, :3], atol=0.003)
    np.testing.assert_allclose(recovered[:3, 3], world[:3, 3], atol=0.001)
    assert result["orientation"]["resolved"]
    with pytest.raises(ValueError, match="requires --frame"):
        calibrate(data, a1_pixel=pixel)
    with pytest.raises(ValueError, match="corner square"):
        calibrate(data, frame=0, a1_pixel=[159.5, 159.5])


def test_scale_is_measured_and_supplied_size_does_not_rescale_sensor_depth():
    data = capture()
    data["depth"] *= 1.4
    data["world_from_camera"][:, :3, 3] *= 1.4
    before = data["depth"].copy()
    result = calibrate(data)
    assert abs(result["square_size_m"] - 0.063) < 0.0004
    with pytest.raises(ValueError, match="supplied size"):
        calibrate(data, square_size=0.045)
    np.testing.assert_equal(data["depth"], before)


def test_depth_outliers_are_rejected_while_board_fit_remains_accurate():
    data = capture()
    corners = detect_frame(data, 0)["corners"]
    for u, v in corners[np.arange(0, 49, 5)]:
        x, y = np.rint([u, v]).astype(int)
        data["depth"][0, y - 3 : y + 4, x - 3 : x + 4] += 0.1
    result = calibrate(data, frame=0)
    assert 35 <= result["quality"]["inlier_corners"] < 49
    assert abs(result["square_size_m"] - 0.045) < 0.0003


def test_missing_grid_depth_and_camera_pose_drift_fail():
    original = capture()
    data = copy.deepcopy(original)
    data["rgb"][:] = 127
    with pytest.raises(ValueError, match="grid not visible"):
        calibrate(data)
    data = copy.deepcopy(original)
    data["confidence"][:] = 0
    data["confidence"][:, 150:170, 150:170] = 2
    with pytest.raises(ValueError, match="usable depth"):
        calibrate(data)
    data = copy.deepcopy(original)
    data["world_from_camera"][1, 0, 3] += 0.03
    with pytest.raises(ValueError, match="inconsistent across camera poses"):
        calibrate(data)


def test_auto_calibration_flows_through_reconstruction_and_preserves_ambiguity(tmp_path):
    report = reconstruct(
        ROOT / "chess-small/capture.npz", None, tmp_path / "scene", samples=3, simulation=False
    )
    assert report["objects"] == 3
    graph = json.loads((tmp_path / "scene/scene_graph.json").read_text())
    assert not graph["board_orientation"]["resolved"]
    assert (tmp_path / "scene/calibration/overlay.png").is_file()
    with pytest.raises(ValueError, match="physical a1"):
        board_from_observation(dict(symbols=["."] * 64, board_orientation=graph["board_orientation"]), {})
    with pytest.raises(ValueError, match="physical a1"):
        reconstruct(
            ROOT / "chess-small/capture.npz",
            None,
            tmp_path / "game",
            context="unused.json",
            simulation=False,
        )


def test_later_reference_frame_and_static_fusion_exclude_the_lead_in(tmp_path):
    source = ROOT / "chess-small/capture.npz"
    reconstruct(
        source, None, tmp_path / "later", reference_frame=1, static_scene=True, samples=2, simulation=False
    )
    graph = json.loads((tmp_path / "later/scene_graph.json").read_text())
    assert graph["frame"] == 1
    assert graph["time_s"] == pytest.approx(1 / 30)
    assert graph["assumptions"]["fused_frames"] == [1, 2]
    reference = json.loads((tmp_path / "later/upstream_workspace/input/reference.json").read_text())
    assert reference["frame"] == 1
    with pytest.raises(ValueError, match="Reference frame outside"):
        reconstruct(source, None, tmp_path / "bad", reference_frame=3, simulation=False)
