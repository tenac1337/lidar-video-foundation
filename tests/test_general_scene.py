import copy
import json

import numpy as np
import pytest
import trimesh

from lidar_video.general_scene import InstancePolicy, associate_instances, build_mesh_scene, infer_support
from lidar_video.scene_program_assets import run_asset_scene_program


def fixture():
    """Analytic RGB-D fixture, not a learned reconstruction benchmark."""
    n, h, w = 3, 128, 128
    K = np.array([[180.0, 0, 64], [0, 180, 64], [0, 0, 1]])
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, :3] = np.diag([1.0, -1, -1])
    poses[:, :3, 3] = [[-0.02, 0, 0.8], [0, 0, 0.8], [0.02, 0, 0.8]]
    yy, xx = np.indices((h, w))
    rays = np.stack([(xx - 64) / 180, (yy - 64) / 180, np.ones((h, w))], -1)
    boxes = [([-0.13, -0.03, 0.05], [0.12, 0.10, 0.10]), ([0.12, 0.07, 0.04], [0.10, 0.09, 0.08])]
    depth = np.empty((n, h, w), np.float32)
    rgb = np.zeros((n, h, w, 3), np.uint8)
    detections = {}
    for frame, pose in enumerate(poses):
        depth[frame] = pose[2, 3]
        rgb[frame] = [120, 120, 120]
        detections[frame] = []
        for index, (position, size) in enumerate(boxes):
            z = pose[2, 3] - position[2] - size[2] / 2
            p = rays * z @ pose[:3, :3].T + pose[:3, 3]
            mask = np.all(abs(p[..., :2] - position[:2]) < np.array(size[:2]) / 2, axis=-1)
            depth[frame][mask] = z
            rgb[frame][mask] = [200, 40, 40] if index == 0 else [40, 160, 200]
            detections[frame].append(dict(mask=mask))
    data = dict(
        rgb=rgb,
        depth=depth,
        confidence=np.full(depth.shape, 2, np.uint8),
        K=K,
        world_from_camera=poses,
        rgb_time=np.arange(n) * 0.5,
        depth_time=np.arange(n) * 0.5,
        source_frame=np.array([11, 43, 72]),
    )
    priors = {}
    for index, (position, size) in enumerate(boxes):
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(position)
        priors[f"object_{index:03d}"] = dict(
            visual=mesh,
            collision=[mesh.copy()],
            provenance={"source": "synthetic analytic fixture, not learned"},
            flat_top_support=True,
        )
    return data, detections, priors


def test_generic_ids_reordered_detections_and_metric_mesh_replay(tmp_path):
    data, detections, priors = fixture()
    detections[1].reverse()
    selected, audit = associate_instances(data, detections, static_scene=True)
    assert set(selected) == {"object_000", "object_001"}
    assert not audit["rejected"]
    assert selected["object_000"]["observations"][1]["detection"] == 1
    support = infer_support(data, detections)
    graph = build_mesh_scene(data, selected, priors, support, tmp_path / "source")
    assert all(n.get("square") is None for n in graph["nodes"])
    assert len(graph["edges"]) == 2
    replay = run_asset_scene_program(tmp_path / "source/scene_graph.json", tmp_path / "replay", count=1)
    variants = run_asset_scene_program(
        tmp_path / "source/scene_graph.json",
        tmp_path / "variants",
        randomize=True,
        count=8,
        jitter=0.04,
        seed=19,
    )
    assert replay["replay_nodes_edges_exact"]
    assert variants["attempts"] == [1] * 8
    a = json.loads((tmp_path / "variants/scene_graph_000.json").read_text())
    b = json.loads((tmp_path / "variants/scene_graph_001.json").read_text())
    assert a["nodes"] != b["nodes"]
    for name in selected:
        path = f"assets/{name}/visual.obj"
        assert (tmp_path / "source" / path).read_bytes() == (tmp_path / "variants" / path).read_bytes()


def test_low_confidence_is_reported_not_filled():
    data, detections, _ = fixture()
    for frame, items in detections.items():
        for item in items:
            data["confidence"][frame][item["mask"]] = 0
    selected, audit = associate_instances(data, detections, static_scene=True)
    assert not selected
    assert len(audit["rejected"]) == 6


def test_repeated_view_and_undeclared_motion_not_accepted():
    data, detections, _ = fixture()
    with pytest.raises(ValueError, match="Static-scene"):
        associate_instances(data, detections, static_scene=False)
    data["world_from_camera"][:] = data["world_from_camera"][0]
    data["depth"][:] = data["depth"][0]
    detections = {i: copy.deepcopy(detections[0]) for i in range(3)}
    selected, audit = associate_instances(data, detections, static_scene=True)
    assert not selected
    assert all(r["reason"] == "no distinct camera viewpoints" for r in audit["rejected"])


def test_overlapping_masks_rejected():
    data, detections, _ = fixture()
    detections[0].append(copy.deepcopy(detections[0][0]))
    with pytest.raises(ValueError, match="Overlapping"):
        associate_instances(data, detections, static_scene=True)


def test_bad_scale_or_pose_cannot_be_hidden_by_fitting(tmp_path):
    data, detections, priors = fixture()
    selected, _ = associate_instances(data, detections, static_scene=True)
    support = infer_support(data, detections)
    priors["object_000"]["visual"].apply_translation([0, 0, 0.1])
    priors["object_000"]["collision"][0].apply_translation([0, 0, 0.1])
    with pytest.raises(ValueError, match="disagrees"):
        build_mesh_scene(data, selected, priors, support, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_missing_collision_and_inventory_fail_before_output(tmp_path):
    data, detections, priors = fixture()
    selected, _ = associate_instances(data, detections, static_scene=True)
    support = infer_support(data, detections)
    with pytest.raises(ValueError, match="inventory"):
        build_mesh_scene(data, selected, {}, support, tmp_path / "missing")
    priors["object_000"]["collision"] = []
    with pytest.raises(ValueError, match="collision parts"):
        build_mesh_scene(data, selected, priors, support, tmp_path / "missing")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"support_tolerance_m": 0.02},
        {"min_native_points": True},
        {"seed": -1},
        {"min_valid_fraction": 1.1},
        {"association_distance_m": -1},
    ],
)
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        InstancePolicy(**kwargs)
