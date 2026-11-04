import csv
import subprocess

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lidar_video.annotated_board import fit_annotated_grid, metric_grid
from lidar_video.capture import load_capture
from lidar_video.scenic_export import export_scenic, sample_scenic
from lidar_video.stray import import_stray, read_odometry, uniform_selection


def fixture_directory(path):
    path.mkdir()
    (path / "depth").mkdir()
    (path / "confidence").mkdir()
    fields = "timestamp frame x y z qx qy qz qw fx fy cx cy distortion_center_x distortion_center_y".split()
    with (path / "odometry.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i in range(6):
            writer.writerow(
                dict(
                    timestamp=100 + i / 60,
                    frame=i,
                    x=1,
                    y=2,
                    z=3,
                    qx=0,
                    qy=0,
                    qz=0,
                    qw=1,
                    fx=40 + i,
                    fy=40 + i,
                    cx=32,
                    cy=24,
                )
            )
            cv2.imwrite(str(path / "depth" / f"{i:06d}.png"), np.full((6, 8), 500 + i, np.uint16))
            cv2.imwrite(str(path / "confidence" / f"{i:06d}.png"), np.full((6, 8), 2, np.uint8))
    rgb = np.stack([np.full((48, 64, 3), 30 * i, np.uint8) for i in range(6)])
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "64x48",
            "-r",
            "60",
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path / "rgb.mp4"),
        ],
        input=rgb.tobytes(),
        check=True,
    )
    return path


def test_stray_import_units_axes_varying_intrinsics_and_association(tmp_path):
    source = fixture_directory(tmp_path / "source")
    report = import_stray(source, tmp_path / "out", fps=30)
    data = load_capture(tmp_path / "out/capture.npz")
    assert report["decoded_video_frames"] == 6
    assert report["video_edit_list_ignored"]
    assert not report["independent_depth_timestamps"]
    assert np.array_equal(data["source_frame"], [0, 2, 4])
    assert np.allclose(data["depth"][:, 0, 0], [0.5, 0.502, 0.504])
    assert np.allclose(data["K"][:, 0, 0], [5, 5.25, 5.5])
    assert np.allclose(data["world_from_camera"][:, :3, 3], [1, -3, 2])
    assert np.allclose(data["world_from_camera"][0, :3, :3] @ [0, 0, 1], [0, -1, 0])
    assert np.allclose(data["rgb"][:, 0, 0, 0], [0, 60, 120], atol=3)


def test_stray_refuses_missing_frames(tmp_path):
    source = fixture_directory(tmp_path / "source")
    (source / "depth/000002.png").unlink()
    with pytest.raises(ValueError, match="frame IDs disagree"):
        import_stray(source, tmp_path / "out")


def test_mp4_leading_edit_does_not_shift_rgb_depth_pairs(tmp_path):
    source = fixture_directory(tmp_path / "source")
    shifted = tmp_path / "shifted.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-itsoffset",
            "-0.016667",
            "-i",
            str(source / "rgb.mp4"),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            str(shifted),
        ],
        check=True,
    )
    shifted.replace(source / "rgb.mp4")
    reader = cv2.VideoCapture(str(source / "rgb.mp4"))
    visible = 0
    while reader.read()[0]:
        visible += 1
    reader.release()
    assert visible == 5  # The first physical sample is hidden by the edit list.
    import_stray(source, tmp_path / "out", fps=30)
    data = load_capture(tmp_path / "out/capture.npz")
    assert np.allclose(data["rgb"][:, 0, 0, 0], [0, 60, 120], atol=3)


def test_stray_refuses_bad_quaternion_and_distortion(tmp_path):
    source = fixture_directory(tmp_path / "source")
    path = source / "odometry.csv"
    original = path.read_text()
    rows = list(csv.DictReader(original.splitlines()))
    rows[0]["qw"] = "2"
    with path.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="quaternion"):
        read_odometry(path)
    rows[0]["qw"] = "1"
    rows[0]["distortion_center_x"] = "32"
    with path.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="distortion"):
        read_odometry(path)


def test_explicit_resampling_keeps_original_tuple_indices():
    times = np.array([10, 10.018, 10.035, 10.068, 10.084, 10.101])
    indices, target = uniform_selection(times, 20)
    assert indices.tolist() == [0, 2, 5]
    assert np.allclose(target, [10, 10.05, 10.1])
    with pytest.raises(ValueError, match="repeats"):
        uniform_selection(times, 200)


def test_folded_board_metric_fit_without_scale_or_pose_prior():
    grid = np.array([[x, y, 0] for y in (-1, 1) for x in range(-3, 4)], float)
    rotation = Rotation.from_euler("xyz", [12, -24, 76], degrees=True).as_matrix()
    points = metric_grid(grid, 0.045, 0.009) @ rotation.T + [1.2, -0.6, 0.8]
    size, gap, pose, rms = fit_annotated_grid(grid, points, center_gap=True)
    assert np.isclose(size, 0.045) and np.isclose(gap, 0.009)
    assert np.allclose(pose[:3, :3], rotation) and np.allclose(pose[:3, 3], [1.2, -0.6, 0.8])
    assert rms < 1e-8
    with pytest.raises(ValueError, match="metric fit"):
        fit_annotated_grid(grid, points, center_gap=False)


def test_dense_scenic_conditions_ranges_on_square_containment(tmp_path):
    nodes = [dict(id="board", kind="support", position_m=[0, 0, -0.01], size_m=[0.4, 0.4, 0.02])]
    for i in range(8):
        for j in range(4):
            x, y = (i - 3.5) * 0.05, (j - 1.5) * 0.05
            nodes.append(
                dict(
                    id=f"piece_{i}_{j}",
                    kind="object",
                    position_m=[x, y, 0.025],
                    size_m=[0.0499, 0.0499, 0.05],
                    square_bounds_m=[[x - 0.025, y - 0.025], [x + 0.025, y + 0.025]],
                )
            )
    graph = dict(nodes=nodes, edges=[], tracks=[])
    export_scenic(graph, tmp_path / "scene.scenic", jitter=0.002)
    samples = sample_scenic(tmp_path / "scene.scenic", graph, tmp_path / "samples.json", count=3)
    assert len(samples) == 3
