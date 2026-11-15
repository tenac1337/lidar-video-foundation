"""Independent ray-generated development scenes, never reserved capture pixels."""

import json

import numpy as np
import pytest
import trimesh

from lidar_video.general_scene import associate_instances, build_mesh_scene, infer_support, rectangular_top
from lidar_video.scene_program_assets import run_asset_scene_program


def shape_fixture():
    platform = trimesh.creation.box(extents=[0.16, 0.16, 0.05])
    platform.apply_translation([-0.12, 0, 0.025])
    cylinder = trimesh.creation.cylinder(radius=0.032, height=0.06, sections=32)
    cylinder.apply_translation([-0.12, 0, 0.08])
    # A concave L assembled from two adjacent rectangular pieces, then rotated.
    parts = []
    for center, size in [
        ([0.12, -0.035, 0.035], [0.12, 0.03, 0.07]),
        ([0.075, 0.015, 0.035], [0.03, 0.07, 0.07]),
    ]:
        part = trimesh.creation.box(extents=size)
        part.apply_translation(center)
        part.apply_transform(trimesh.transformations.rotation_matrix(0.2, [0, 0, 1], [0.12, 0, 0]))
        parts.append(part)
    visuals = [platform, cylinder, trimesh.util.concatenate(parts)]
    collisions = [[platform.copy()], [cylinder.copy()], [p.copy() for p in parts]]
    h = w = 96
    K = np.array([[140.0, 0, 48], [0, 140, 48], [0, 0, 1]])
    yy, xx = np.indices((h, w))
    camera_rays = np.stack([(xx - 48) / 140, (yy - 48) / 140, np.ones((h, w))], -1).reshape(-1, 3)
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, :3, :3] = np.diag([1.0, -1, -1])
    poses[:, :3, 3] = [[-0.035, 0, 0.8], [0, 0, 0.8], [0.035, 0, 0.8]]
    depths, rgbs, detections = [], [], {}
    colors = np.array([[140, 140, 140], [180, 60, 30], [40, 80, 190], [80, 180, 70]], np.uint8)
    for f, pose in enumerate(poses):
        directions = camera_rays @ pose[:3, :3].T
        origins = np.tile(pose[:3, 3], (h * w, 1))
        z = np.full(h * w, pose[2, 3])
        labels = np.zeros(h * w, np.uint16)
        for label, mesh in enumerate(visuals, 1):
            hits, ids, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
            distances = np.full(h * w, np.inf)
            np.minimum.at(distances, ids, pose[2, 3] - hits[:, 2])
            visible = distances < z
            z[visible] = distances[visible]
            labels[visible] = label
        labels = labels.reshape(h, w)
        depths.append(z.reshape(h, w))
        rgbs.append(colors[labels])
        detections[f] = [dict(mask=labels == i) for i in range(1, 4)]
    depth = np.array(depths, np.float32)
    data = dict(
        rgb=np.array(rgbs),
        depth=depth,
        confidence=np.full(depth.shape, 2, np.uint8),
        K=K,
        world_from_camera=poses,
        rgb_time=np.arange(3) * 0.5,
        depth_time=np.arange(3) * 0.5,
        source_frame=np.array([17, 48, 81]),
    )
    selected, audit = associate_instances(data, detections, static_scene=True)
    assert len(selected) == 3 and not audit["rejected"]
    priors = {}
    for name, item in selected.items():
        index = item["reference"]["detection"]
        priors[name] = dict(
            visual=visuals[index],
            collision=collisions[index],
            provenance=dict(source="analytic ray fixture; ground-truth mesh, not inferred"),
        )
    return data, detections, selected, priors


def test_nonbox_stack_preserves_shapes_and_support_for_100_variants(tmp_path):
    data, detections, selected, priors = shape_fixture()
    graph = build_mesh_scene(data, selected, priors, infer_support(data, detections), tmp_path / "scene")
    assert any(e["target"] != "support" for e in graph["edges"])
    assert any(len(n.get("collision_paths", [])) == 2 for n in graph["nodes"])
    summary = run_asset_scene_program(
        tmp_path / "scene/scene_graph.json",
        tmp_path / "variants",
        randomize=True,
        count=100,
        seed=7,
        jitter=0.04,
    )
    assert summary["attempts"] == [1] * 100
    varied = []
    for path in sorted((tmp_path / "variants").glob("scene_graph_*.json")):
        sample = json.loads(path.read_text())
        nodes = {n["id"]: n for n in sample["nodes"]}
        for edge in sample["edges"]:
            c, p = nodes[edge["source"]], nodes[edge["target"]]
            cp, pp = np.array(c["position_m"]), np.array(p["position_m"])
            cs, ps = np.array(c["size_m"]), np.array(p["size_m"])
            assert np.all(abs(cp[:2] - pp[:2]) + cs[:2] / 2 <= ps[:2] / 2 + 1e-8)
            assert abs(cp[2] - cs[2] / 2 - pp[2] - ps[2] / 2) < 1e-8
        varied.append(nodes[next(iter(selected))]["position_m"])
    assert len(varied) == 100 and np.ptp(varied, axis=0)[0] > 0.01
    for name in selected:
        assert (tmp_path / "scene/assets" / name / "visual.obj").read_bytes() == (
            tmp_path / "variants/assets" / name / "visual.obj"
        ).read_bytes()


def test_top_geometry_certificate_rejects_curves_and_holes():
    box = trimesh.creation.box(extents=[1, 1, 1])
    assert rectangular_top([box], box.bounds)
    sphere = trimesh.creation.icosphere(subdivisions=2)
    assert not rectangular_top([sphere], sphere.bounds)
    cylinder = trimesh.creation.cylinder(radius=0.5, height=1)
    assert not rectangular_top([cylinder], cylinder.bounds)
    # Four rails surround a hole: full envelope but not a full supporting top.
    rails = []
    for pos, size in [
        ([-0.4, 0, 0], [0.2, 1, 1]),
        ([0.4, 0, 0], [0.2, 1, 1]),
        ([0, -0.4, 0], [0.6, 0.2, 1]),
        ([0, 0.4, 0], [0.6, 0.2, 1]),
    ]:
        rail = trimesh.creation.box(extents=size)
        rail.apply_translation(pos)
        rails.append(rail)
    assert not rectangular_top(rails, box.bounds)


def test_collision_frame_disagreement_is_rejected(tmp_path):
    data, detections, selected, priors = shape_fixture()
    next(iter(priors.values()))["collision"][0].apply_translation([0.02, 0, 0])
    with pytest.raises(ValueError, match="Visual/collision"):
        build_mesh_scene(data, selected, priors, infer_support(data, detections), tmp_path / "bad")
