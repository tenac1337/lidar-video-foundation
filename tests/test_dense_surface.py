from dataclasses import replace

import cv2
import numpy as np
import pytest

from lidar_video.dense_surface import (
    DensePolicy,
    close_view_triangles,
    corroborate_pairs,
    plane_sweep_pair,
    reciprocal_check,
    verify_depth_proposal,
)


def scene():
    """Analytic two-depth scene with occlusion; texture does not encode depth."""
    h, w = 96, 144
    k = np.array([[140.0, 0, 72], [0, 140.0, 48], [0, 0, 1]])
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 0, 3] = [-0.06, 0, 0.06]
    rng = np.random.default_rng(7)
    texture = rng.integers(15, 240, (640, 640, 3), dtype=np.uint8)
    # Band-limit the world texture before ray sampling to avoid synthetic aliasing.
    texture = cv2.GaussianBlur(texture, (11, 11), 1.6)
    yy, xx = np.indices((h, w))
    rays = np.stack([(xx - 72) / 140, (yy - 48) / 140, np.ones((h, w))], -1)
    images, depths = [], []
    for pose in poses:
        near = rays * 0.6 + pose[:3, 3]
        foreground = (abs(near[..., 0]) < 0.09) & (abs(near[..., 1]) < 0.11)
        z = np.where(foreground, 0.6, 1.0)
        world = rays * z[..., None] + pose[:3, 3]
        uv = (world[..., :2] * 400 + 320).astype(np.float32)
        images.append(cv2.remap(texture, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR))
        depths.append(z)
    return np.array(images), np.array(depths), k, poses


POLICY = DensePolicy(near_m=0.4, far_m=1.3, planes=160)


def test_metric_plane_sweep_recovers_both_depths_without_sensor_or_shape_prior():
    rgb, truth, k, poses = scene()
    result = plane_sweep_pair(
        rgb[1], rgb[2], k, k, poses[1], poses[2], source_ids=[11, 22], policy=POLICY, static_scene=True
    )
    z = result["depth"]
    for value in (0.6, 1.0):
        core = cv2.erode((truth[1] == value).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        valid = core & (z > 0)
        assert valid.sum() > 150
        assert np.median(abs(z[valid] - value)) < 0.006
        assert np.quantile(abs(z[valid] - value), 0.95) < 0.015
    assert not result["metadata"]["geometry_checked"]


def test_three_view_reciprocal_surface_rejects_missing_witness_and_wrong_depth():
    rgb, truth, k, poses = scene()
    proposals = []
    for i in (0, 2):
        forward = plane_sweep_pair(
            rgb[1], rgb[i], k, k, poses[1], poses[i], source_ids=[1, i], policy=POLICY, static_scene=True
        )
        reverse = plane_sweep_pair(
            rgb[i], rgb[1], k, k, poses[i], poses[1], source_ids=[i, 1], policy=POLICY, static_scene=True
        )
        proposals.append(
            dict(id=i, depth=forward["depth"], reverse_depth=reverse["depth"], K=k, pose=poses[i])
        )
    result = corroborate_pairs(proposals, k, poses[1], reference_id=1, policy=POLICY)
    keep = result["valid"]
    assert keep.sum() > 500
    assert np.quantile(abs(result["depth"][keep] - truth[1][keep]), 0.95) < 0.015
    proposals[0]["reverse_depth"] = np.zeros_like(truth[0])
    assert not corroborate_pairs(proposals, k, poses[1], reference_id=1, policy=POLICY)["valid"].any()
    proposals[0]["reverse_depth"] = np.full_like(truth[0], 1.2)
    assert not corroborate_pairs(proposals, k, poses[1], reference_id=1, policy=POLICY)["valid"].any()


def test_no_texture_or_zero_baseline_never_creates_surface():
    rgb, _, k, poses = scene()
    rgb[:] = 127
    result = plane_sweep_pair(
        rgb[0], rgb[1], k, k, poses[0], poses[1], source_ids=[0, 1], policy=POLICY, static_scene=True
    )
    assert not result["valid"].any()
    with pytest.raises(ValueError, match="baseline"):
        plane_sweep_pair(
            rgb[0], rgb[1], k, k, poses[0], poses[0], source_ids=[0, 1], policy=POLICY, static_scene=True
        )
    with pytest.raises(ValueError, match="Static"):
        plane_sweep_pair(rgb[0], rgb[1], k, k, poses[0], poses[1], source_ids=[0, 1])


def test_depth_occlusion_is_not_counted_as_geometric_support():
    _, _, k, poses = scene()
    query = np.ones((96, 144))
    for value, state in [(0.6, "nearer_or_occluded"), (1.3, "farther_conflict")]:
        check = reciprocal_check(query, k, poses[1], query * value, k, poses[2], policy=POLICY)
        assert check[state].sum() > 1000
        assert not check["keep"].any()


def test_duplicate_witnesses_and_invalid_policy_abstain():
    _, depth, k, poses = scene()
    p = dict(id=0, depth=depth[1], reverse_depth=depth[0], K=k, pose=poses[0])
    with pytest.raises(ValueError, match="Distinct"):
        corroborate_pairs([p, p], k, poses[1], reference_id=1)
    with pytest.raises(ValueError, match="Finite"):
        replace(POLICY, depth_tolerance_m=np.nan).validate()


def test_repeated_pattern_does_not_pass_nonlocal_uniqueness():
    rgb, _, k, poses = scene()
    x = np.arange(rgb.shape[2])
    periodic = np.tile(((x % 8) < 4).astype(np.uint8) * 200, (rgb.shape[1], 1))
    left = np.repeat(periodic[..., None], 3, -1)
    right = np.roll(left, -8, axis=1)
    r = plane_sweep_pair(
        left, right, k, k, poses[0], poses[1], source_ids=[0, 1], policy=POLICY, static_scene=True
    )
    # Both repeated candidate matches must be in the common field of view.
    assert r["valid"][10:-10, 32:-10].mean() < 0.01


def test_third_edge_is_required_even_when_anchor_pair_edges_agree():
    _, truth, k, poses = scene()
    witnesses = [dict(id=i, K=k, pose=poses[i]) for i in (0, 2)]
    selected = np.zeros((*truth[1].shape, 2), np.int16)
    selected[..., 1] = 1
    edges = {(0, 2): truth[0], (2, 0): truth[2]}
    checked = close_view_triangles(truth[1], selected, k, poses[1], witnesses, edges, policy=POLICY)
    assert checked["keep"].sum() > 5000
    assert not close_view_triangles(truth[1], selected, k, poses[1], witnesses, {}, policy=POLICY)[
        "keep"
    ].any()
    edges[0, 2] = truth[0] + 0.1
    assert not close_view_triangles(truth[1], selected, k, poses[1], witnesses, edges, policy=POLICY)[
        "keep"
    ].any()


def test_rotated_camera_warp_has_correct_metric_depth():
    # Independently warp a constant-depth plane through a rotated target camera.
    rgb, _, k, poses = scene()
    angle = np.deg2rad(4)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    pose = poses[2].copy()
    pose[:3, :3] = rotation
    yy, xx = np.indices(rgb[1].shape[:2])
    rays = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(k).T @ rotation.T
    points = rays * (0.8 / rays[..., 2:]) + pose[:3, 3]
    pixels = points @ k.T
    uv = pixels[..., :2] / pixels[..., 2:]
    right = cv2.remap(rgb[1], uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    result = plane_sweep_pair(
        rgb[1], right, k, k, poses[1], pose, source_ids=[1, 2], policy=POLICY, static_scene=True
    )
    z = result["depth"]
    assert (z > 0).sum() > 1000
    assert np.median(abs(z[z > 0] - 0.8)) < 0.01


def test_external_proposal_cannot_bypass_third_edge_checks():
    _, truth, k, poses = scene()
    witnesses = [dict(id=i, K=k, pose=poses[i], reverse_depth=truth[i]) for i in (0, 2)]
    edges = {(0, 2): truth[0], (2, 0): truth[2]}
    r = verify_depth_proposal(truth[1], k, poses[1], witnesses, edges, reference_id=1, policy=POLICY)
    assert r["keep"].sum() > 5000
    np.testing.assert_allclose(r["depth"][r["keep"]], truth[1][r["keep"]], atol=1e-7)
    bad = verify_depth_proposal(truth[1] + 0.1, k, poses[1], witnesses, edges, reference_id=1, policy=POLICY)
    assert not bad["keep"].any()
    assert not verify_depth_proposal(truth[1], k, poses[1], witnesses, {}, reference_id=1, policy=POLICY)[
        "keep"
    ].any()
