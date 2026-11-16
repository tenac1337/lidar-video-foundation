import copy

import numpy as np
import pytest
import trimesh

from lidar_video.multiview_shape import (
    VolumePolicy,
    select_candidate,
    silhouette_score,
    silhouette_volume,
)


def cameras(mesh, origins=None):
    """Independent ray-traced silhouettes of a known metric solid."""
    k = np.array([[160.0, 0, 64], [0, 160, 64], [0, 0, 1]])
    yy, xx = np.indices((128, 128))
    local = np.stack([(xx - 64) / 160, (yy - 64) / 160, np.ones_like(xx)], -1).reshape(-1, 3)
    result = []
    origins = origins or [[-0.16, -0.13, 0.40], [0.16, -0.13, 0.40], [0.0, 0.18, 0.40]]
    for i, origin in enumerate(origins):
        origin = np.asarray(origin)
        forward = np.array([0, 0, 0.04]) - origin
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        pose = np.eye(4)
        pose[:3, :3] = np.stack([right, np.cross(forward, right), forward], 1)
        pose[:3, 3] = origin
        _, ids, _ = mesh.ray.intersects_location(
            np.tile(origin, (len(local), 1)), local @ pose[:3, :3].T, multiple_hits=False
        )
        labels = np.zeros((128, 128), np.uint16)
        labels.ravel()[ids] = 1
        result.append(dict(source_frame=11 + i * 13, labels=labels, label=1, K=k, world_from_camera=pose))
    return result


def box(height=0.08):
    mesh = trimesh.creation.box([0.05, 0.06, height])
    mesh.apply_translation([0, 0, height / 2])
    return mesh


def test_new_volume_recovers_height_missing_from_flat_prior():
    truth, flattened = box(), box(0.008)
    views = cameras(truth, [[-0.16, -0.13, 0.4], [0.16, -0.13, 0.4], [0, 0.18, 0.4], [0, -0.2, 0.27]])
    mesh, report = silhouette_volume(
        views[:3],
        np.eye(4),
        static_scene=True,
        policy=VolumePolicy(pitch_m=0.002, mask_margin_pixels=0, max_height_m=0.14),
    )
    assert mesh.is_watertight and report["connected_components"] >= 1
    # Rasterized cone boundaries may leave tiny disconnected cells; they are
    # reported and retained, not silently removed to satisfy the fixture.
    parts = mesh.split(only_watertight=False)
    assert max(p.volume for p in parts) / sum(p.volume for p in parts) > 0.999
    assert mesh.extents[2] > 0.065 > flattened.extents[2]
    assert abs(mesh.bounds[0, 2]) < 1e-7
    assert report["construction_sources"] == [11, 24, 37]
    assert silhouette_score(mesh, views[3])["iou"] > silhouette_score(flattened, views[3])["iou"] + 0.2


def test_underdetermined_two_view_volume_is_vetoed_on_real_reserved_silhouette():
    truth, flat = box(), box(0.008)
    views = cameras(truth)
    proposal, _ = silhouette_volume(
        views[:2],
        np.eye(4),
        static_scene=True,
        policy=VolumePolicy(pitch_m=0.002, mask_margin_pixels=0, max_height_m=0.14),
    )
    report = select_candidate({"volume": proposal}, flat, views, [11, 24], [37])
    assert report["selected"] == "volume" and not report["accepted"]
    assert "Reserved-view regression" in report["reason"]


def test_reserved_view_never_changes_shape_construction():
    views = cameras(box())
    a, _ = silhouette_volume(views[:2], np.eye(4), static_scene=True, policy=VolumePolicy(pitch_m=0.002))
    views[2]["labels"][:] = 99
    b, _ = silhouette_volume(views[:2], np.eye(4), static_scene=True, policy=VolumePolicy(pitch_m=0.002))
    np.testing.assert_array_equal(a.vertices, b.vertices)
    np.testing.assert_array_equal(a.faces, b.faces)


def test_height_cap_and_repeated_cameras_abstain():
    views = cameras(box())
    with pytest.raises(ValueError, match="declared bound"):
        silhouette_volume(
            views, np.eye(4), static_scene=True, policy=VolumePolicy(pitch_m=0.002, max_height_m=0.04)
        )
    repeated = copy.deepcopy(views[:2])
    repeated[1]["world_from_camera"] = repeated[0]["world_from_camera"].copy()
    with pytest.raises(ValueError, match="distinct camera"):
        silhouette_volume(repeated, np.eye(4), static_scene=True)
    repeated[1]["source_frame"] = repeated[0]["source_frame"]
    with pytest.raises(ValueError, match="Duplicate source"):
        silhouette_volume(repeated, np.eye(4), static_scene=True)
    with pytest.raises(ValueError, match="static_scene"):
        silhouette_volume(views, np.eye(4))


def test_plane_transform_is_not_applied_as_extra_scale():
    views = cameras(box())
    shift = np.eye(4)
    shift[:3, 3] = [0.4, -0.2, 0.7]
    shifted = [dict(v, world_from_camera=shift @ v["world_from_camera"]) for v in views]
    policy = VolumePolicy(pitch_m=0.002, mask_margin_pixels=0)
    original, _ = silhouette_volume(views[:2], np.eye(4), static_scene=True, policy=policy)
    moved, _ = silhouette_volume(shifted[:2], np.linalg.inv(shift), static_scene=True, policy=policy)
    np.testing.assert_allclose(moved.bounds - shift[:3, 3], original.bounds, atol=0.0021)
    bad = np.diag([2.0, 2, 2, 1])
    with pytest.raises(ValueError, match="rigid"):
        silhouette_volume(views, bad, static_scene=True)


def test_missing_camera_and_excessive_grid_fail_closed():
    views = cameras(box())
    with pytest.raises(ValueError, match="voxel budget"):
        silhouette_volume(views[:2], np.eye(4), static_scene=True, policy=VolumePolicy(max_cells=100))
    views[0]["K"] = np.zeros((3, 3))
    with pytest.raises(ValueError, match="intrinsics"):
        silhouette_volume(views, np.eye(4), static_scene=True)


def test_selection_recovers_known_shape_and_rejects_missing_validation():
    truth, flat = box(), box(0.008)
    views = cameras(truth)
    report = select_candidate({"flat": flat, "correct": truth}, flat, views, [11, 24], [37])
    assert report["selected"] == "correct" and report["accepted"]
    assert not report["validation_used_for_ranking"]
    proposal = select_candidate({"correct": truth}, flat, views[:2], [11, 24], [])
    assert proposal["selected"] == "correct" and not proposal["accepted"]


def test_validation_veto_does_not_select_runner_up(monkeypatch):
    from lidar_video import multiview_shape

    views = cameras(box())
    best, alternative, incumbent = box(), box(0.06), box(0.008)
    values = {id(best): [0.9, 0.9, 0.2], id(alternative): [0.8, 0.8, 0.9], id(incumbent): [0.5, 0.5, 0.5]}
    calls = []

    def score(mesh, view):
        calls.append((id(mesh), view["source_frame"]))
        return dict(iou=values[id(mesh)][[11, 24, 37].index(view["source_frame"])], valid=True)

    monkeypatch.setattr(multiview_shape, "silhouette_score", score)
    report = select_candidate({"best": best, "alternative": alternative}, incumbent, views, [11, 24], [37])
    assert report["selected"] == "best" and not report["accepted"]
    assert (id(alternative), 37) not in calls


def test_single_construction_view_gain_cannot_hide_another_regression(monkeypatch):
    from lidar_video import multiview_shape

    views = cameras(box())
    proposed, baseline = box(), box(0.06)

    def score(mesh, view):
        return dict(
            iou=(0.95 if view["source_frame"] == 11 else 0.49) if mesh is proposed else 0.5, valid=True
        )

    monkeypatch.setattr(multiview_shape, "silhouette_score", score)
    report = select_candidate({"one_view_gain": proposed}, baseline, views, [11, 24], [37])
    assert report["selected"] is None and not report["accepted"]


def test_bounded_pose_refinement_preserves_shape_and_support_height():
    from lidar_video.multiview_shape import refine_support_pose

    truth = box()
    displaced = truth.copy()
    displaced.apply_translation([0.004, -0.004, 0])
    views = cameras(truth)
    corrected, report = refine_support_pose(displaced, views[:2], np.eye(4))
    assert report["construction_sources"] == [11, 24]
    assert not report["accepted"]
    np.testing.assert_allclose(corrected.bounds[:, 2], displaced.bounds[:, 2], atol=1e-10)
    assert corrected.volume == pytest.approx(displaced.volume)
    transform = np.array(report["world_from_original_world"])
    np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-10)
    assert max(abs(np.array(report["translation_support_xy_m"]))) <= 0.008
    for view in views[:2]:
        assert silhouette_score(corrected, view)["iou"] >= silhouette_score(displaced, view)["iou"]
    assert any(
        silhouette_score(corrected, v)["iou"] > silhouette_score(displaced, v)["iou"] for v in views[:2]
    )
    with pytest.raises(ValueError, match="bounded support pose"):
        refine_support_pose(displaced, views[:2], np.eye(4), max_translation_m=0.5)


def test_portable_runner_writes_evidence_and_refuses_overwrite(tmp_path):
    import json

    from PIL import Image

    from lidar_video.multiview_shape_runner import run

    truth = box()
    incumbent = truth.copy()
    incumbent.apply_translation([0.004, -0.004, 0])
    incumbent.export(tmp_path / "incumbent.ply")
    views = cameras(truth)
    encoded = []
    for view in views:
        filename = str(view["source_frame"]) + ".png"
        Image.fromarray(view["labels"].astype(np.uint16)).save(tmp_path / filename)
        encoded.append(
            dict(
                source_frame=view["source_frame"],
                label=view["label"],
                labels=filename,
                K=view["K"].tolist(),
                world_from_camera=view["world_from_camera"].tolist(),
            )
        )
    manifest = dict(
        incumbent_mesh="incumbent.ply",
        proposal_method="support_pose",
        static_scene=True,
        support_from_world=np.eye(4).tolist(),
        views=encoded,
        construction_sources=[11, 24],
        validation_sources=[37],
    )
    path = tmp_path / "input.json"
    path.write_text(json.dumps(manifest))
    result = run(path, tmp_path / "output")
    assert result["proposal_evidence"]["shape_and_scale_preserved"]
    assert (tmp_path / "output/proposal-world.ply").is_file()
    assert json.loads((tmp_path / "output/result.json").read_text())["construction_sources"] == [11, 24]
    with pytest.raises(FileExistsError):
        run(path, tmp_path / "output")
