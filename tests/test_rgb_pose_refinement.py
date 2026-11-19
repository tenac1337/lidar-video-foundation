import copy

import numpy as np
import pytest

from lidar_video.landmark_tracks import triangulate_landmark
from lidar_video.rgb_association import RGBPolicy
from lidar_video.rgb_pose_refinement import refine_camera_component


def fixture():
    k = np.array([[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]])
    true = np.tile(np.eye(4), (3, 1, 1))
    true[:, 0, 3] = [0, 0.08, -0.08]
    initial = true.copy()
    initial[1, 1, 3] = 0.004
    initial[2, 1, 3] = -0.003
    views = [dict(K=k, world_from_camera=p, source_frame=i) for i, p in enumerate(initial)]
    rng = np.random.default_rng(9)
    points = rng.uniform([-0.18, -0.15, 0.7], [0.18, 0.15, 1.1], (80, 3))
    tracks = []
    for p in points:
        obs = []
        for i, pose in enumerate(true):
            cp = (p - pose[:3, 3]) @ pose[:3, :3]
            uv = cp @ k.T
            obs.append(dict(view=i, pixel=(uv[:2] / uv[2]).tolist()))
        tri = triangulate_landmark(obs, views, RGBPolicy(max_reprojection_px=6))
        assert tri["passed"]
        tracks.append(
            dict(
                world_point=tri["world_point"], observations=obs, depth_anchors=[dict(view=0, z=float(p[2]))]
            )
        )
    return views, tracks, true


def test_refinement_preserves_metric_gauge_and_checks_unused_tracks():
    views, tracks, true = fixture()
    result = refine_camera_component(views, tracks, static_scene=True, max_nfev=100)
    assert result["report"]["accepted"], result["report"]
    assert np.array_equal(result["poses"][0], true[0])
    assert result["report"]["after"]["median_px"] < result["report"]["before"]["median_px"] * 0.5
    assert np.linalg.norm(result["poses"][1, :3, 3] - true[1, :3, 3]) < 0.004
    corrupted = copy.deepcopy(tracks)
    for i, t in enumerate(corrupted):
        if i % 5 == 0:
            t["observations"][1]["pixel"][1] += 30
    check = refine_camera_component(views, corrupted, static_scene=True, max_nfev=100)
    assert np.allclose(check["candidate_poses"], result["candidate_poses"])
    assert not check["report"]["accepted"]
    assert np.array_equal(check["poses"], np.array([v["world_from_camera"] for v in views]))


def test_insufficient_tracks_abstain_and_invalid_observations_fail():
    views, tracks, _ = fixture()
    result = refine_camera_component(views, tracks[:5], static_scene=True)
    assert not result["report"]["accepted"]
    with pytest.raises(ValueError):
        refine_camera_component(views, tracks)
    tracks[0]["observations"][1]["view"] = 0
    with pytest.raises(ValueError):
        refine_camera_component(views, tracks, static_scene=True)


def test_disconnected_fit_graph_abstains():
    views, tracks, _ = fixture()
    extra_views = [dict(v, source_frame=v["source_frame"] + 3) for v in views]
    extra_tracks = copy.deepcopy(tracks)
    for t in extra_tracks:
        for o in t["observations"]:
            o["view"] += 3
        for a in t["depth_anchors"]:
            a["view"] += 3
    result = refine_camera_component(views + extra_views, tracks + extra_tracks, static_scene=True)
    assert not result["report"]["accepted"]
    assert result["report"]["reason"] == "fit camera graph is disconnected"
