import numpy as np
import pytest

from lidar_video.learned_association import geometric_pairs, interior_labels
from lidar_video.rgb_association import associate_rgb


def fixture():
    k = np.array([[300.0, 0, 160], [0, 300, 120], [0, 0, 1]])
    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = 0.1
    points = np.array([[x, y, 0.8] for y in [-0.04, 0, 0.04] for x in [-0.06, -0.02, 0.02, 0.06]])
    views = [dict(K=k, world_from_camera=p, source_frame=i) for i, p in enumerate(poses)]
    pixels = []
    for v in views:
        camera = points - v["world_from_camera"][:3, 3]
        projected = camera @ k.T
        pixels.append(projected[:, :2] / projected[:, 2:])
    features = [dict(pixels=p, labels=np.ones(len(p), int)) for p in pixels]
    matches = np.column_stack([np.arange(len(points)), np.arange(len(points))])
    return features, views, matches, np.full(len(points), 0.9), points


def test_keypoints_outside_instance_core_never_supply_object_evidence():
    labels = np.zeros((20, 20), np.uint16)
    labels[4:16, 4:16] = 1
    labels[8:12, 8:12] = 2
    pixels = np.array([[0, 0], [4, 4], [6, 6], [9, 9], [30, 30]])
    np.testing.assert_array_equal(interior_labels(pixels, labels), [0, 0, 1, 2, 0])
    with pytest.raises(ValueError, match="Finite"):
        interior_labels(np.array([[np.nan, 0]]), labels)


def test_learned_matches_still_require_metric_geometry():
    f, v, m, s, truth = fixture()
    result = geometric_pairs(*f, m, s, *v)
    assert len(result) == 1 and result[0]["passed"]
    np.testing.assert_allclose(result[0]["world_points"], truth, atol=1e-10)
    f[1]["pixels"][:, 1] += 30
    assert geometric_pairs(*f, m, s, *v) == []


def test_instance_dominance_rejects_split_identity_even_with_many_matches():
    f, v, m, s, _ = fixture()
    f[1]["labels"][6:] = 2
    result = geometric_pairs(*f, m, s, *v)
    assert len(result) == 2 and all(not r["passed"] for r in result)
    assert all(r["matches"] == 6 for r in result)


def test_duplicate_feature_indices_and_duplicate_locations_are_not_extra_support():
    f, v, m, s, _ = fixture()
    with pytest.raises(ValueError, match="one-to-one"):
        geometric_pairs(*f, np.concatenate([m, m[:1]]), np.append(s, 0.9), *v)
    f[0]["pixels"][:] = f[0]["pixels"][0]
    assert all(not r["passed"] for r in geometric_pairs(*f, m, s, *v))
    with pytest.raises(ValueError, match="outside feature"):
        geometric_pairs(*f, m + 100, s, *v)


def test_injected_matcher_keeps_existing_all_pairs_safeguards():
    f, v, m, s, _ = fixture()
    labels = np.ones((240, 320), np.uint16)
    views = [dict(x, rgb=np.zeros((240, 320, 3), np.uint8), labels=labels) for x in v]

    def extract(rgb, labels, policy):
        return dict(pixels=np.zeros((12, 2)))

    def match(lf, rf, left, right, policy):
        return geometric_pairs(*f, m, s, left, right, policy)

    result = associate_rgb(views, static_scene=True, feature_provider=extract, pair_provider=match)
    assert len(result["tracks"]) == 1
    assert not result["native_depth_validated"] and not result["inventory_complete"]


def test_low_confidence_correspondences_cannot_supply_geometric_support():
    f, v, m, scores, _ = fixture()
    assert geometric_pairs(*f, m, np.zeros_like(scores), *v) == []
