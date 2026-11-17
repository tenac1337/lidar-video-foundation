import numpy as np
import pytest
from test_multiview_shape import box, cameras

from lidar_video.landmark_refinement import refine_landmark_mesh
from lidar_video.multiview_shape import silhouette_score


def surface_points(height=0.08):
    top = [[x, y, height] for x in [-0.02, 0, 0.02] for y in [-0.025, 0, 0.025]]
    sides = [[x, y, z] for x in [-0.025, 0.025] for y in [-0.02, 0, 0.02] for z in [0.02, 0.04, 0.06]]
    return np.array(top + sides)


def test_rgb_landmarks_refine_full_prior_without_losing_silhouette_or_base():
    truth, incumbent = box(), box(0.072)
    views = cameras(truth)
    before_v, before_f = incumbent.vertices.copy(), incumbent.faces.copy()
    mesh, report = refine_landmark_mesh(incumbent, surface_points(), views[:2], np.eye(4), static_scene=True)
    assert report["candidate"]["landmark_huber_m2"] < 0.8 * report["incumbent"]["landmark_huber_m2"]
    assert mesh.extents[2] > incumbent.extents[2]
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.bounds[0, 2] == pytest.approx(incumbent.bounds[0, 2])
    np.testing.assert_array_equal(mesh.faces, before_f)
    np.testing.assert_array_equal(incumbent.vertices, before_v)
    np.testing.assert_array_equal(incumbent.faces, before_f)
    for view in views[:2]:
        assert silhouette_score(mesh, view)["iou"] >= silhouette_score(incumbent, view)["iou"]
    transform = np.array(report["world_from_original_world"])
    np.testing.assert_allclose(mesh.vertices, before_v @ transform[:3, :3].T + transform[:3, 3])
    assert not report["accepted"]


def test_exact_surface_and_reordered_clustered_landmarks_do_not_change_prior():
    prior = box()
    views = cameras(prior)
    p = surface_points()
    original, a = refine_landmark_mesh(prior, p, views[:2], np.eye(4), static_scene=True)
    reordered = np.concatenate([p[::-1], np.repeat(p[:1], 20, axis=0)])
    second, b = refine_landmark_mesh(prior, reordered, views[:2], np.eye(4), static_scene=True)
    np.testing.assert_allclose(original.vertices, prior.vertices, atol=1e-12)
    np.testing.assert_allclose(second.vertices, original.vertices, atol=1e-12)
    assert a["occupied_landmark_cells"] == b["occupied_landmark_cells"]


def test_sparse_collinear_or_nonfinite_landmarks_cannot_generate_a_shape():
    prior = box()
    views = cameras(prior)
    for points in [
        np.zeros((2, 3)),
        np.zeros((5, 3)),
        np.array([[i * 0.01, 0, 0.04] for i in range(5)]),
        np.full((5, 3), np.nan),
    ]:
        with pytest.raises(ValueError, match="landmarks"):
            refine_landmark_mesh(prior, points, views[:2], np.eye(4), static_scene=True)


def test_static_scene_and_camera_evidence_are_required():
    prior = box()
    views = cameras(prior)
    p = surface_points()
    with pytest.raises(ValueError, match="static scene"):
        refine_landmark_mesh(prior, p, views[:2], np.eye(4))
    with pytest.raises(ValueError, match="Two distinct"):
        refine_landmark_mesh(prior, p, views[:1], np.eye(4), static_scene=True)
    with pytest.raises(ValueError, match="bounded"):
        refine_landmark_mesh(prior, p, views[:2], np.eye(4), static_scene=True, max_scale_fraction=0.2)
