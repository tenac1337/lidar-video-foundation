import numpy as np
import pytest

from lidar_video.surface_fragments import support_fragment_hypotheses


def test_separates_raised_observations_without_synthesizing_object_bases():
    rng = np.random.default_rng(3)
    plane = np.column_stack([rng.uniform(-0.3, 0.3, (4000, 2)), rng.normal(0, 0.0005, 4000)])
    patches = []
    for x in [-0.12, 0.12]:
        # Irregular observed patches, no category or complete-object model.
        p = rng.uniform([x - 0.02, -0.02, 0.025], [x + 0.02, 0.02, 0.065], (600, 3))
        patches.append(p)
    points = np.concatenate([plane, *patches])
    original = points.copy()
    r = support_fragment_hypotheses(points, z_up=True)
    assert r["report"]["accepted"]
    assert r["report"]["fragment_count"] == 2
    assert np.all(r["labels"][:4000] == -1)
    assert np.all(r["labels"][4000:] >= 0)
    for f in r["fragments"]:
        assert np.min(points[f["point_indices"], 2]) >= 0.025
        assert not f["complete_object"]
    np.testing.assert_array_equal(points, original)


def test_unknown_gravity_or_insufficient_plane_abstains():
    rng = np.random.default_rng(5)
    points = rng.uniform(-1, 1, (2000, 3))
    with pytest.raises(ValueError, match="z-up"):
        support_fragment_hypotheses(points)
    r = support_fragment_hypotheses(points, z_up=True)
    assert not r["report"]["accepted"]
    assert not r["fragments"]
    assert (r["labels"] == -1).all()
