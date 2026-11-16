import numpy as np
import pytest

from lidar_video.rgb_association import RGBPolicy, associate_rgb, features, pair_evidence, triangulate_matches


def setup_pair():
    k = np.array([[300.0, 0, 160], [0, 300, 120], [0, 0, 1]])
    left = dict(K=k, world_from_camera=np.eye(4), source_frame=3)
    pose = np.eye(4)
    pose[0, 3] = 0.1
    right = dict(K=k, world_from_camera=pose, source_frame=17)
    points = np.array(
        [
            [-0.04, -0.04, 0.8],
            [0.04, -0.04, 0.9],
            [-0.04, 0.04, 0.8],
            [0.04, 0.04, 0.9],
            [0, -0.02, 0.85],
            [0, 0.02, 0.85],
        ]
    )

    def project(view):
        p = points - view["world_from_camera"][:3, 3]
        uv = p @ k.T
        return uv[:, :2] / uv[:, 2:]

    return left, right, points, project(left), project(right)


def test_calibrated_triangulation_recovers_metric_points_without_depth():
    left, right, truth, a, b = setup_pair()
    actual, valid = triangulate_matches(a, b, left, right)
    assert valid.all()
    np.testing.assert_allclose(actual, truth, atol=1e-10)
    with pytest.raises(ValueError, match="baseline"):
        triangulate_matches(a, a, left, left)


def test_off_epipolar_matches_and_negative_depth_are_rejected():
    left, right, _, a, b = setup_pair()
    wrong = b.copy()
    wrong[:, 1] += 20
    _, valid = triangulate_matches(a, wrong, left, right)
    assert not valid.any()
    _, valid = triangulate_matches(b, a, left, right)
    assert not valid.any()


def test_mutual_features_form_pair_and_repeated_descriptors_abstain():
    left, right, _, a, b = setup_pair()
    desc = np.eye(6, 128, dtype=np.float32)
    lf = dict(pixels=a, labels=np.ones(6, int), descriptors=desc)
    rf = dict(pixels=b, labels=np.full(6, 7), descriptors=desc)
    result = pair_evidence(lf, rf, left, right)
    assert len(result) == 1 and result[0]["passed"]
    assert result[0]["left_label"] == 1 and result[0]["right_label"] == 7
    ambiguous = dict(rf, descriptors=np.ones((6, 128), np.float32))
    assert pair_evidence(lf, ambiguous, left, right) == []


def test_feature_extractor_excludes_background_and_mask_boundaries():
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    labels = np.zeros((200, 200), np.uint16)
    labels[35:165, 35:165] = 4
    result = features(rgb, labels)
    assert len(result["pixels"]) > 0
    pixels = np.rint(result["pixels"]).astype(int)
    assert np.all(result["labels"] == 4)
    assert np.all((pixels > 35) & (pixels < 164))
    assert len(set(map(tuple, pixels))) == len(pixels)
    empty = features(np.zeros_like(rgb), labels)
    assert empty["descriptors"].shape == (0, 128)
    with pytest.raises(ValueError, match="Registered RGB"):
        features(rgb.astype(float), labels)


def mock_views():
    result = []
    for i in range(3):
        labels = np.ones((20, 20), np.uint16)
        labels[:, 10:] = 2
        pose = np.eye(4)
        pose[0, 3] = i * 0.1
        result.append(
            dict(
                rgb=np.zeros((20, 20, 3), np.uint8),
                labels=labels,
                K=np.eye(3),
                world_from_camera=pose,
                source_frame=i,
                frame=i,
            )
        )
    return result


def test_association_rejects_transitive_chain_and_many_to_one(monkeypatch):
    from lidar_video import rgb_association

    monkeypatch.setattr(rgb_association, "features", lambda *args: dict(pixels=np.zeros((6, 2))))

    def pair(lf, rf, left, right, policy):
        if right["source_frame"] - left["source_frame"] == 1:
            return [dict(left_label=1, right_label=1, passed=True)]
        return []

    monkeypatch.setattr(rgb_association, "pair_evidence", pair)
    result = associate_rgb(mock_views(), static_scene=True)
    assert not result["tracks"]
    assert any(r["reason"] == "RGB chain lacks all-pairs corroboration" for r in result["rejected"])

    def collision(*args):
        return [
            dict(left_label=1, right_label=1, passed=True),
            dict(left_label=2, right_label=1, passed=True),
        ]

    monkeypatch.setattr(rgb_association, "pair_evidence", collision)
    result = associate_rgb(mock_views(), static_scene=True)
    assert not result["tracks"]
    assert any(r["reason"] == "ambiguous many-to-one RGB association" for r in result["rejected"])


def test_association_preserves_labels_and_marks_hypothesis_scope(monkeypatch):
    from lidar_video import rgb_association

    monkeypatch.setattr(rgb_association, "features", lambda *args: dict(pixels=np.zeros((6, 2))))
    monkeypatch.setattr(
        rgb_association, "pair_evidence", lambda *args: [dict(left_label=2, right_label=2, passed=True)]
    )
    result = associate_rgb(mock_views(), static_scene=True)
    assert len(result["tracks"]) == 1
    assert [o["label"] for o in result["tracks"]["rgb_object_000"]["observations"]] == [2, 2, 2]
    assert not result["inventory_complete"] and not result["native_depth_validated"]
    with pytest.raises(ValueError, match="static_scene"):
        associate_rgb(mock_views())
    views = mock_views()
    views[1]["source_frame"] = 0
    with pytest.raises(ValueError, match="Duplicate source"):
        associate_rgb(views, static_scene=True)
    with pytest.raises(ValueError, match="policy"):
        RGBPolicy(ratio=1.0)


def test_empty_detection_views_are_preserved_without_inventing_tracks():
    views = mock_views()
    for view in views:
        view["labels"][:] = 0
    result = associate_rgb(views, static_scene=True)
    assert result["feature_counts"] == [0, 0, 0]
    assert not result["tracks"] and not result["pair_checks"]
    views[0]["world_from_camera"][0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        associate_rgb(views, static_scene=True)


def extension_fixture():
    left, right, points, a, b = setup_pair()
    third = dict(K=left["K"], world_from_camera=np.eye(4), source_frame=29)
    third["world_from_camera"][0, 3] = -0.1
    views = []
    for v in [left, right, third]:
        labels = np.zeros((240, 320), np.uint16)
        p = (points - v["world_from_camera"][:3, 3]) @ v["K"].T
        uv = np.rint(p[:, :2] / p[:, 2:]).astype(int)
        for x, y in uv:
            labels[y - 10 : y + 11, x - 10 : x + 11] = 1
        views.append(dict(v, labels=labels))
    seeds = [dict(view=i, frame=i, source_frame=v["source_frame"], label=1) for i, v in enumerate(views[:2])]
    result = dict(
        tracks={"seed": dict(observations=seeds)},
        pair_checks=[
            dict(
                passed=True,
                left_view=0,
                right_view=1,
                left_label=1,
                right_label=1,
                world_points=points.tolist(),
                left_pixels=a.tolist(),
                right_pixels=b.tolist(),
            )
        ],
    )
    return views, result


def test_landmark_projection_extends_with_weaker_evidence_label():
    from lidar_video.rgb_association import extend_tracks

    views, result = extension_fixture()
    expanded = extend_tracks(views, result)
    track = expanded["tracks"]["seed"]
    assert track["accepted"] and len(track["observations"]) == 3
    assert "no new descriptor corroboration" in track["observations"][2]["evidence"]
    assert len(track["seed_observations"]) == 2
    assert not expanded["native_depth_validated"]
    views[2]["labels"][:] = 0
    assert len(extend_tracks(views, result)["tracks"]["seed"]["observations"]) == 2


def test_landmark_projection_rejects_competing_tracks_and_wrong_source():
    import copy

    from lidar_video.rgb_association import extend_tracks

    views, result = extension_fixture()
    result["tracks"]["duplicate"] = copy.deepcopy(result["tracks"]["seed"])
    expanded = extend_tracks(views, result)
    assert all(not t["accepted"] for t in expanded["tracks"].values())
    result["tracks"]["seed"]["observations"][0]["source_frame"] = 999
    with pytest.raises(ValueError, match="Seed observation"):
        extend_tracks(views, result)


def test_epipolar_guidance_disambiguates_off_ray_descriptor_duplicates():
    left, right, _, a, b = setup_pair()
    desc = np.eye(6, 128, dtype=np.float32)
    lf = dict(pixels=a, labels=np.ones(6, int), descriptors=desc)
    rf = dict(
        pixels=np.concatenate([b, b + [0, 60]]),
        labels=np.full(12, 7),
        descriptors=np.concatenate([desc, desc]),
    )
    assert pair_evidence(lf, rf, left, right) == []
    found = pair_evidence(lf, rf, left, right, RGBPolicy(geometry_guided=True))
    assert len(found) == 1 and found[0]["passed"] and found[0]["matches"] == 4


def test_landmark_uncertainty_rejects_nominal_only_mask_overlap():
    from lidar_video.rgb_association import extend_tracks

    views, result = extension_fixture()
    labels = views[2]["labels"]
    labels[:] = 0
    p = np.array(result["pair_checks"][0]["world_points"])
    camera = (p - views[2]["world_from_camera"][:3, 3]) @ views[2]["K"].T
    uv = np.rint(camera[:, :2] / camera[:, 2:]).astype(int)
    for x, y in uv:
        labels[y - 1 : y + 2, x - 1 : x + 2] = 1
    track = extend_tracks(views, result)["tracks"]["seed"]
    assert len(track["observations"]) == 2
    assert track["perturbation_corners"] == 16
