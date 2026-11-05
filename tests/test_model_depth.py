"""Scene context survives object filtering; rejected object depth never leaks back."""

import numpy as np
import pytest

from lidar_video.model_depth import object_conditioning_depth


def inputs():
    depth = np.full((2, 3, 4), 0.8, np.float32)
    depth[1, 1, 1:3] = 0.5
    archive = dict(
        frames=np.array([10, 42]),
        depth=depth,
        valid=np.ones(depth.shape, bool),
        native_K=np.eye(3),
        depth_sample_id=np.arange(24).reshape(depth.shape),
    )
    mask = np.zeros((3, 4), bool)
    mask[1, 1:3] = True
    trusted = mask.copy()
    trusted[1, 2] = False
    return archive, mask, trusted


def test_context_survives_while_object_scoring_keeps_exact_trusted_evidence():
    archive, mask, trusted = inputs()
    archive["valid"][1, 0, 0] = False
    before = {k: v.copy() for k, v in archive.items()}
    result = object_conditioning_depth(archive, 42, mask, trusted)
    np.testing.assert_array_equal(result["valid"][1] & mask, archive["valid"][1] & trusted)
    np.testing.assert_array_equal(result["valid"][1][~mask], archive["valid"][1][~mask])
    assert result["depth"][1, 1, 2] == 0
    assert result["depth"][1, 0, 0] == 0
    assert result["depth"][1, 0, 1] == archive["depth"][1, 0, 1]
    for k in archive:
        np.testing.assert_array_equal(archive[k], before[k])
    for k in ("native_K", "depth_sample_id", "frames"):
        np.testing.assert_array_equal(result[k], archive[k])
    np.testing.assert_array_equal(result["depth"][0], archive["depth"][0])
    np.testing.assert_array_equal(result["valid"][0], archive["valid"][0])


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_bad_context_depth_is_not_promoted(value):
    archive, mask, trusted = inputs()
    archive["depth"][1, 0, 0] = value
    result = object_conditioning_depth(archive, 42, mask, trusted)
    assert not result["valid"][1, 0, 0]
    assert result["depth"][1, 0, 0] == 0


def test_untrusted_mask_or_wrong_frame_fails():
    archive, mask, trusted = inputs()
    with pytest.raises(ValueError, match="uniquely"):
        object_conditioning_depth(archive, 1, mask, trusted)
    trusted[0, 0] = True
    with pytest.raises(ValueError, match="inside"):
        object_conditioning_depth(archive, 42, mask, trusted)
    with pytest.raises(ValueError, match="registered"):
        object_conditioning_depth(archive, 42, mask[:1], trusted[:1])
    with pytest.raises(ValueError, match="no valid trusted"):
        object_conditioning_depth(archive, 42, mask, np.zeros_like(mask))
