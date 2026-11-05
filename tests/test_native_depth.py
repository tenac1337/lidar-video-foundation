"""Sensor geometry survives RGB resizing, rectification, and the model handoff."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from lidar_video.capture import camera_points, normalize, validate
from lidar_video.native_depth import sensor_camera_points
from lidar_video.pipeline import prepare_upstream
from lidar_video.reconstruction import calibrated_points


def resized_capture():
    n, h, w = 2, 8, 10
    y, x = np.indices((h, w))
    native_z = np.stack([0.6 + x * 0.004 + y * 0.003 + f * 0.01 for f in range(n)]).astype(np.float32)
    ks = np.array([[[14.0, 0, 4.2], [0, 14.0, 3.4], [0, 0, 1]], [[14.7, 0, 4.5], [0, 14.7, 3.2], [0, 0, 1]]])
    yy, xx = np.indices((h * 4, w * 4), dtype=np.float32)
    ids = cv2.remap(
        np.arange(h * w, dtype=np.float32).reshape(h, w),
        xx / 4,
        yy / 4,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.int32)
    data = dict(
        rgb=np.zeros((n, h * 4, w * 4, 3), np.uint8),
        depth=native_z.reshape(n, -1)[:, ids],
        confidence=np.full((n, h * 4, w * 4), 2, np.uint8),
        K=np.diag([4, 4, 1]) @ ks,
        world_from_camera=np.repeat(np.eye(4)[None], n, axis=0),
        rgb_time=np.arange(n, dtype=float),
        depth_time=np.arange(n, dtype=float),
        depth_sample_id=np.repeat(ids[None], n, axis=0),
        native_K=ks,
        native_depth_shape=np.array([h, w]),
    )
    return data, native_z, ks


def test_original_xyz_survives_upsampling_changing_intrinsics_and_repeated_normalization():
    data, native_z, ks = resized_capture()
    normalized = normalize(data)
    twice = normalize(normalized)
    cal = dict(board_from_world=np.eye(4))
    for candidate in (data, normalized, twice):
        np.testing.assert_array_equal(candidate["native_K"], ks)
        for f in range(2):
            ids = candidate["depth_sample_id"][f]
            expected = camera_points(native_z[f], ks[f]).reshape(-1, 3)[np.maximum(ids, 0)]
            actual = calibrated_points(candidate, cal, f)
            np.testing.assert_allclose(actual[ids >= 0], expected[ids >= 0], atol=1e-12)
            assert np.isnan(actual[ids < 0]).all()
    # Former model-pixel backprojection spreads one sensor sample over many rays.
    first = data["depth_sample_id"][0] == 24
    assert len(np.unique(sensor_camera_points(data, 0)[first], axis=0)) == 1
    assert len(np.unique(camera_points(data["depth"][0], data["K"][0])[first], axis=0)) > 1


def test_legacy_origin_aligned_input_recovers_native_intrinsics_before_normalization():
    explicit, _, ks = resized_capture()
    legacy = {k: v for k, v in explicit.items() if k not in ("depth_sample_id", "native_K")}
    normalized = normalize(legacy)
    np.testing.assert_allclose(normalized["native_K"], ks)
    for f in range(2):
        np.testing.assert_allclose(
            sensor_camera_points(normalized, f), sensor_camera_points(normalize(explicit), f), equal_nan=True
        )


def test_explicit_ids_without_original_intrinsics_are_not_guessed():
    data, _, _ = resized_capture()
    del data["native_K"]
    with pytest.raises(ValueError, match="original native_K"):
        normalize(data)
    with pytest.raises(ValueError, match="original native_K"):
        calibrated_points(data, dict(board_from_world=np.eye(4)), 0)


@pytest.mark.parametrize(
    "key,value",
    [
        ("native_K", np.eye(2)),
        ("native_K", np.full((3, 3), np.nan)),
        ("native_K", np.diag([-1, 1, 1])),
        ("native_depth_shape", np.array([8.0, 10.0])),
        ("native_depth_shape", np.array([8, 0])),
    ],
)
def test_bad_native_geometry_metadata_fails_validation(key, value):
    data, _, _ = resized_capture()
    data[key] = value
    with pytest.raises(ValueError, match="native_"):
        validate(data)


def test_colab_worker_uses_sensor_rays_for_conditioning_and_scoring(tmp_path):
    data, _, _ = resized_capture()
    data = normalize(data)
    out = prepare_upstream(data, tmp_path / "upstream")
    worker = tmp_path / "worker.py"
    worker.write_text("""import numpy as np

def points(depth,k):
    raise AssertionError("The incorrect resized-grid rays must not be used")

def run(video,masks,depth,out,frame,*args):
    with np.load(depth) as data:
        z=data['depth'][int(frame)].astype(np.float32)
        conditioning=points(z,data['K'])
        scoring=points(z,data['K'])
        np.savez(out,conditioning=conditioning,scoring=scoring)
""")
    adapter = Path(__file__).parents[1] / "scripts/sam3d_native_rays.py"
    spec = importlib.util.spec_from_file_location("native_adapter", adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = tmp_path / "model_points.npz"
    module.run(worker, ["unused", "unused", str(out / "depth/depth.npz"), str(result), "1"])
    with np.load(result) as saved:
        for key in ("conditioning", "scoring"):
            np.testing.assert_allclose(saved[key], sensor_camera_points(data, 1), equal_nan=True)
