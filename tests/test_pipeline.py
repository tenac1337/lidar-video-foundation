import io
import json
import zipfile

import lzfse
import numpy as np
import pytest
from PIL import Image

from lidar_video.capture import camera_points, import_r3d, load_capture, normalize, validate, world_points
from lidar_video.geometry import fit_support, scene_graph
from lidar_video.pipeline import prepare_upstream
from lidar_video.scenic_export import export_scenic, sample_scenic, validate_sample


@pytest.fixture
def capture():
    n, h, w = 3, 80, 80
    depth = np.ones((n, h, w), np.float32)
    depth[:, 20:32, 15:27] = 0.93
    depth[:, 44:56, 55:67] = 0.90
    pose = np.repeat(np.diag([1.0, -1, -1, 1])[None], n, axis=0)
    pose[:, 2, 3] = 1
    return dict(
        rgb=np.full((n, h, w, 3), 120, np.uint8),
        depth=depth,
        confidence=np.full(depth.shape, 2, np.uint8),
        K=np.array([[100.0, 0, 39.5], [0, 100, 39.5], [0, 0, 1]]),
        world_from_camera=pose,
        rgb_time=np.arange(n) / 30,
        depth_time=np.arange(n) / 30,
    )


def test_validation_and_metric_identity(capture):
    assert validate(capture)["frames"] == 3
    out = normalize(capture)
    np.testing.assert_array_equal(out["depth"], capture["depth"])
    np.testing.assert_array_equal(out["K"], capture["K"])


@pytest.mark.parametrize(
    "field, mutation, message",
    [
        ("depth_time", lambda a: a + 0.1, "skew"),
        ("rgb_time", lambda a: np.zeros_like(a), "strictly increasing"),
        ("world_from_camera", lambda a: a @ np.diag([-1, 1, 1, 1]), "proper rigid"),
        ("depth", lambda a: -a, "Negative depth"),
        ("confidence", lambda a: a * 0, "no usable depth"),
        ("K", lambda a: np.zeros_like(a), "Invalid pinhole"),
    ],
)
def test_bad_capture_rejected(capture, field, mutation, message):
    capture[field] = mutation(capture[field])
    with pytest.raises(ValueError, match=message):
        validate(capture)


def test_invalid_depth_is_not_filled(capture):
    capture["depth"][0, 2, 3] = np.nan
    capture["confidence"][0, 4, 5] = 0
    out = normalize(capture)
    assert out["depth"][0, 2, 3] == out["depth"][0, 4, 5] == 0
    assert np.isfinite(out["depth"]).all()


def test_varying_intrinsics_remaps_both_streams(capture):
    capture["K"] = np.repeat(capture["K"][None], 3, axis=0)
    capture["K"][1, 0, 2] += 2
    capture["depth"][1] = 1
    capture["depth"][1, 30, 42] = 0.5
    capture["rgb"][1, 30, 42] = [250, 0, 0]
    out = normalize(capture)
    assert out["depth"][1, 30, 40] == 0.5
    np.testing.assert_array_equal(out["rgb"][1, 30, 40], [250, 0, 0])
    assert out["depth"][1, 30, -1] == 0


def test_projection_roundtrip_and_world_axes(capture):
    z, k = capture["depth"][0], capture["K"]
    points = camera_points(z, k)
    uv = points @ k.T
    uv = uv[..., :2] / uv[..., 2:]
    v, u = np.indices(z.shape)
    np.testing.assert_allclose(uv, np.stack([u, v], axis=-1), atol=1e-12)
    world = world_points(z, k, capture["world_from_camera"][0])
    assert abs(world[0, 0, 2]) < 1e-8
    assert world[25, 20, 2] == pytest.approx(0.07, abs=1e-6)


def test_graph_from_depth_not_truth(capture):
    graph, cloud, _ = scene_graph(capture)
    objects = [n for n in graph["nodes"] if n["kind"] == "object"]
    assert len(objects) == 2
    assert len(cloud) == 6400
    assert sorted(round(n["size_m"][2], 2) for n in objects) == [0.07, 0.1]
    assert all(e["evidence"] == "geometric_hypothesis" for e in graph["edges"])
    assert [n["id"] for n in graph["tracks"][0]["nodes"]] == [n["id"] for n in graph["tracks"][-1]["nodes"]]


def test_plane_rejects_vertical_wall():
    rng = np.random.default_rng(3)
    points = np.column_stack([np.ones(1000), rng.uniform(-1, 1, (1000, 2))])
    with pytest.raises(ValueError, match="support plane"):
        fit_support(points)


def test_upstream_npz_preserves_metric_depth(capture, tmp_path):
    out = prepare_upstream(normalize(capture), tmp_path / "upstream")
    with np.load(out / "depth/depth.npz") as data:
        assert set(data.files) == {"depth", "valid", "K", "frames"}
        np.testing.assert_array_equal(data["depth"], capture["depth"])
        assert data["valid"].all()
    assert (out / "input/video.mp4").stat().st_size > 0
    with pytest.raises(FileExistsError):
        prepare_upstream(capture, out)


def test_upstream_rejects_hidden_dropped_frames(capture, tmp_path):
    capture["rgb_time"][-1] += 0.02
    capture["depth_time"][-1] += 0.02
    with pytest.raises(ValueError, match="dropped frames"):
        prepare_upstream(capture, tmp_path / "bad")


def test_scenic_compiles_samples_and_preserves_ids(capture, tmp_path):
    graph, _, _ = scene_graph(capture)
    path = tmp_path / "scene.scenic"
    export_scenic(graph, path)
    samples = sample_scenic(path, graph, tmp_path / "samples.json", count=20)
    assert len(samples) == 20
    for sample in samples:
        validate_sample(sample)
        assert {n["id"] for n in sample["nodes"]} == {n["id"] for n in graph["nodes"]}
    assert samples[0]["nodes"][1]["position_m"] != samples[1]["nodes"][1]["position_m"]
    assert "np.float" not in path.read_text()


def make_r3d(path, varying=False, timestamps=True):
    meta = dict(
        cameraType=1,
        w=8,
        h=6,
        dw=4,
        dh=3,
        K=[10, 0, 0, 0, 10, 0, 3.5, 2.5, 1],
        poses=[[0, 0, 0, 1, 0, 0, 0]] * 2,
    )
    if timestamps:
        meta["frameTimestamps"] = [0, 1 / 30]
    if varying:
        meta["perFrameIntrinsicCoeffs"] = [[10, 10, 3.5, 2.5]] * 2
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata", json.dumps(meta))
        for i in range(2):
            image = io.BytesIO()
            Image.new("RGB", (8, 6), (100, 120, 140)).save(image, format="JPEG")
            archive.writestr(f"rgbd/{i}.jpg", image.getvalue())
            archive.writestr(f"rgbd/{i}.depth", lzfse.compress(np.ones((3, 4), dtype="<f4").tobytes()))
            archive.writestr(f"rgbd/{i}.conf", lzfse.compress(np.full((3, 4), 2, np.uint8).tobytes()))


def test_r3d_fixture_converts_calibration_and_axes(tmp_path):
    path = tmp_path / "test.r3d"
    make_r3d(path)
    import_r3d(path, tmp_path / "capture.npz")
    data = load_capture(tmp_path / "capture.npz")
    np.testing.assert_allclose(data["K"], [[5, 0, 1.5], [0, 5, 1], [0, 0, 1]])
    np.testing.assert_allclose(data["world_from_camera"][0, :3, 2], [0, 1, 0])
    assert data["rgb"].shape == (2, 3, 4, 3)
    assert data["depth"].min() == 1


def test_unknown_r3d_calibration_is_not_silently_ignored(tmp_path):
    path = tmp_path / "varying.r3d"
    make_r3d(path, varying=True)
    with pytest.raises(ValueError, match="schema adapter"):
        import_r3d(path, tmp_path / "capture.npz")


def test_r3d_does_not_invent_timestamps(tmp_path):
    path = tmp_path / "no-times.r3d"
    make_r3d(path, timestamps=False)
    with pytest.raises(ValueError, match="frameTimestamps"):
        import_r3d(path, tmp_path / "capture.npz")


def test_integer_sensor_units_are_not_assumed_to_be_metres(capture):
    capture["depth"] = (capture["depth"] * 1000).astype(np.uint16)
    with pytest.raises(ValueError, match="floating-point metres"):
        validate(capture)
