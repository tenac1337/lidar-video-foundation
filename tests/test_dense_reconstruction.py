import json
from dataclasses import replace

import numpy as np
import pytest
from test_dense_surface import POLICY, scene

from lidar_video.dense_reconstruction import reconstruct_dense, select_neighbors


def capture():
    rgb, _, k, poses = scene()
    return dict(
        rgb=rgb, K=k, world_from_camera=poses, source_frame=np.array([0, 1, 2]), depth="must not be read"
    )


def test_capture_pipeline_saves_checked_colored_surfaces(tmp_path):
    data = capture()
    out = tmp_path / "dense"
    result = reconstruct_dense(
        data, out, reference_ids=[1], excluded_ids=[30], static_scene=True, policy=POLICY, max_width=144
    )
    assert result["after_voxel"] > 500
    frame = np.load(out / "frames/1.npz")
    z = frame["depth"]
    assert (z > 0).sum() == result["frames"][0]["triangle"]["accepted_pixels"]
    assert result["frames"][0]["mesh"]["faces"] > 100
    assert np.isfinite(np.load(out / "clouds/dense_rgb.npy")).all()
    assert (out / "meshes/1-observed.ply").is_file()
    assert len(json.loads((out / "pair-ledger.json").read_text())) == 6
    with pytest.raises(FileExistsError):
        reconstruct_dense(data, out, reference_ids=[1], static_scene=True, policy=POLICY, max_width=144)


def test_excluded_capture_is_rejected_before_any_output(tmp_path):
    out = tmp_path / "forbidden"
    with pytest.raises(ValueError, match="Excluded"):
        reconstruct_dense(capture(), out, reference_ids=[1], excluded_ids=[0], static_scene=True)
    assert not out.exists()


def test_no_baseline_produces_explicit_empty_output(tmp_path):
    data = capture()
    data["world_from_camera"][:] = np.eye(4)
    result = reconstruct_dense(
        data,
        tmp_path / "empty",
        reference_ids=[1],
        static_scene=True,
        policy=POLICY,
        max_width=144,
        z_up_support_fragments=True,
    )
    assert result["after_voxel"] == 0
    assert result["frames"][0]["reason"] == "insufficient eligible witnesses"
    assert not result["support_fragments"]["accepted"]
    assert (tmp_path / "empty/fragment-hypotheses.json").is_file()


def test_resize_intrinsics_preserves_pixel_center_convention(tmp_path):
    data = capture()
    data["world_from_camera"][:] = np.eye(4)
    reconstruct_dense(
        data, tmp_path / "scaled", reference_ids=[1], static_scene=True, policy=POLICY, max_width=72
    )
    scaled = np.load(tmp_path / "scaled/cameras.npz")["K"][0]
    assert scaled[0, 0] == data["K"][0, 0] * 0.5
    assert scaled[0, 2] == (data["K"][0, 2] + 0.5) * 0.5 - 0.5


def test_selection_is_deterministic_and_uses_policy_baseline():
    data = capture()
    p = data["world_from_camera"]
    a, _ = select_neighbors(p, [0, 1, 2], [1])
    b, _ = select_neighbors(p, [0, 1, 2], [1])
    assert a == b == {1: [0, 2]}
    strict = replace(POLICY, min_baseline_m=0.2)
    c, _ = select_neighbors(p, [0, 1, 2], [1], minimum_baseline=strict.min_baseline_m)
    assert c == {1: []}
