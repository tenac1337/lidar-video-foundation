import numpy as np
import pytest

from lidar_video.landmark_tracks import corroborate_chains, triangulate_landmark


def fixture():
    k = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    points = np.array([[x, y, 0.8] for x in [-0.04, 0.04] for y in [-0.04, 0.04]])
    views = []
    pixels = []
    for i, x in enumerate([-0.1, 0, 0.1]):
        pose = np.eye(4)
        pose[0, 3] = x
        views.append(dict(K=k, world_from_camera=pose, source_frame=i, labels=np.ones((240, 320), np.uint16)))
        projected = (points - pose[:3, 3]) @ k.T
        pixels.append(projected[:, :2] / projected[:, 2:])
    pairs = [
        dict(
            left_view=i,
            right_view=j,
            left_label=1,
            right_label=1,
            passed=True,
            left_pixels=pixels[i].tolist(),
            right_pixels=pixels[j].tolist(),
        )
        for i, j in [(0, 1), (1, 2)]
    ]
    return views, pixels, pairs, points


def test_three_view_landmarks_can_validate_chain_without_direct_instance_edge():
    views, pixels, pairs, points = fixture()
    result = corroborate_chains(views, pairs, static_scene=True)
    assert len(result["tracks"]) == 1
    track = result["tracks"]["landmark_object_000"]
    assert track["direct_landmarks_per_view"] == {0: 4, 1: 4, 2: 4}
    actual = np.array([p["world_point"] for p in track["landmarks"]])
    distances = np.linalg.norm(actual[:, None] - points[None, :], axis=2)
    assert np.max(distances.min(axis=1)) < 1e-10
    assert len(set(distances.argmin(axis=1))) == len(points)
    assert not result["native_depth_validated"]


def test_wrong_third_view_cannot_pass_via_instance_transitivity():
    views, pixels, pairs, _ = fixture()
    pairs[1]["right_pixels"] = (pixels[2] + [0, 12]).tolist()
    result = corroborate_chains(views, pairs, static_scene=True)
    assert not result["tracks"]
    assert any(
        r["reason"] == "insufficient directly corroborated three-view landmarks" for r in result["rejected"]
    )


def test_two_view_landmarks_are_not_counted_as_three_view_support():
    views, pixels, pairs, _ = fixture()
    pairs[1]["left_pixels"] = (pixels[1] + [0.5, 0]).tolist()
    result = corroborate_chains(views, pairs, static_scene=True)
    assert not result["tracks"]
    assert all(sum(r["direct_landmarks_per_view"].values()) == 0 for r in result["rejected"])


def test_no_silent_relaxation_or_duplicate_camera_landmarks():
    views, pixels, pairs, _ = fixture()
    with pytest.raises(ValueError, match="conservative"):
        corroborate_chains(views, pairs, static_scene=True, min_landmarks=1)
    with pytest.raises(ValueError, match="static scene"):
        corroborate_chains(views, pairs)
    with pytest.raises(ValueError, match="three unique"):
        triangulate_landmark([dict(view=0, pixel=pixels[0][0])] * 3, views)
