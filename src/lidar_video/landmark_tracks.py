"""Corroborate instance chains using actual RGB landmarks in three or more views.

Unlike an instance-level transitive closure, each admitted observation needs
multiple triangulated feature tracks with a consistent reprojection in every
observed camera. Camera calibration is supplied and is never optimized here.
"""

from dataclasses import asdict

import cv2
import networkx as nx
import numpy as np

from lidar_video.multiview_shape import checked_views
from lidar_video.rgb_association import RGBPolicy


def triangulate_landmark(observations, views, policy=None):
    """Linear multi-camera triangulation with explicit per-view geometric checks."""
    policy = policy or RGBPolicy()
    if len(observations) < 3 or len({o["view"] for o in observations}) != len(observations):
        raise ValueError("A landmark needs at least three unique views")
    rows, cameras, centers = [], [], []
    for o in observations:
        view = views[o["view"]]
        camera = np.linalg.inv(view["world_from_camera"])[:3]
        projection = view["K"] @ camera
        u, v = o["pixel"]
        rows.extend([u * projection[2] - projection[0], v * projection[2] - projection[1]])
        cameras.append((camera, projection))
        centers.append(view["world_from_camera"][:3, 3])
    _, _, vh = np.linalg.svd(np.array(rows))
    point = vh[-1]
    if abs(point[3]) < 1e-12:
        return dict(passed=False, reason="point at infinity")
    point = point[:3] / point[3]
    if not np.isfinite(point).all():
        return dict(passed=False, reason="nonfinite landmark")
    errors, depths = [], []
    for o, (camera, projection) in zip(observations, cameras):
        p = projection @ np.r_[point, 1]
        depths.append(float(camera[2] @ np.r_[point, 1]))
        errors.append(float(np.linalg.norm(p[:2] / p[2] - o["pixel"])) if p[2] != 0 else float("inf"))
    rays = point - np.array(centers)
    norms = np.linalg.norm(rays, axis=1)
    if np.any(norms <= 1e-12):
        return dict(passed=False, reason="landmark at camera center")
    rays /= norms[:, None]
    angles = np.degrees(np.arccos(np.clip(rays @ rays.T, -1, 1)))
    baselines = np.linalg.norm(np.array(centers)[:, None] - np.array(centers)[None, :], axis=2)
    passed = (
        min(depths) > 0
        and max(errors) <= policy.max_reprojection_px
        and float(angles.max()) >= policy.min_angle_degrees
        and float(baselines.max()) >= policy.min_baseline_m
    )
    return dict(
        passed=passed,
        world_point=point.tolist(),
        reprojection_errors_px=errors,
        max_parallax_degrees=float(angles.max()),
        observations=observations,
        reason="corroborated multi-view landmark" if passed else "multi-view geometry disagreement",
    )


def corroborate_chains(
    views, pair_checks, *, static_scene=False, policy=None, min_landmarks=4, max_background_fraction=0.2
):
    """Return corroborated 3+ view hypotheses; never infer a missing object shape."""
    if (
        static_scene is not True
        or type(min_landmarks) is not int
        or min_landmarks < 4
        or not 0 <= max_background_fraction <= 0.2
    ):
        raise ValueError("Explicit static scene and conservative landmark policy required")
    policy = policy or RGBPolicy()
    checked = []
    for v in views:
        positive = np.unique(v["labels"])
        positive = positive[positive > 0]
        checked.append(dict(v, label=int(positive[0]) if len(positive) else 1))
    views = checked_views(checked, allow_empty=True)
    objects, features = nx.Graph(), nx.Graph()
    for pair in pair_checks:
        if not pair["passed"]:
            continue
        i, j = pair["left_view"], pair["right_view"]
        if i == j or not (0 <= i < len(views) and 0 <= j < len(views)):
            raise ValueError("Invalid pair view indices")
        labels = [int(pair["left_label"]), int(pair["right_label"])]
        for view, label in zip([i, j], labels):
            if label < 1 or not np.any(views[view]["labels"] == label):
                raise ValueError("Pair label absent from source view")
        a, b = np.asarray(pair["left_pixels"], float), np.asarray(pair["right_pixels"], float)
        if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or not np.isfinite([a, b]).all():
            raise ValueError("Paired finite pixels required")
        objects.add_edge((i, labels[0]), (j, labels[1]))
        for x, y in zip(a, b):
            keys = [(view, *np.round(pixel, 4).tolist()) for view, pixel in zip([i, j], [x, y])]
            for key, view, label, pixel in zip(keys, [i, j], labels, [x, y]):
                if key in features and features.nodes[key]["label"] != label:
                    raise ValueError("Feature has contradictory instance labels")
                features.add_node(key, view=view, pixel=pixel.tolist(), label=label)
            features.add_edge(*keys)
    landmarks, rejected_features = [], []
    for group in nx.connected_components(features):
        observations = [features.nodes[n] for n in sorted(group)]
        if len(observations) < 3:
            continue
        if len({o["view"] for o in observations}) != len(observations):
            rejected_features.append(
                dict(reason="feature identity collides within a view", observations=observations)
            )
            continue
        result = triangulate_landmark(observations, views, policy)
        (landmarks if result["passed"] else rejected_features).append(result)
    accepted, rejected = {}, []
    for group in sorted(nx.connected_components(objects), key=lambda c: sorted(c)):
        nodes = sorted(group)
        observations = [
            dict(view=i, frame=views[i].get("frame", i), source_frame=views[i]["source_frame"], label=label)
            for i, label in nodes
        ]
        record = dict(observations=observations)
        reason = None
        if len(nodes) < 3:
            reason = "fewer than three instance views"
        elif len({i for i, _ in nodes}) != len(nodes):
            reason = "ambiguous instance identity within a view"
        selected = [p for p in landmarks if all((o["view"], o["label"]) in group for o in p["observations"])]
        support = {i: sum(any(o["view"] == i for o in p["observations"]) for p in selected) for i, _ in nodes}
        record.update(landmarks=selected, direct_landmarks_per_view=support)
        if reason is None and min(support.values()) < min_landmarks:
            reason = "insufficient directly corroborated three-view landmarks"
        mask_checks = []
        if reason is None:
            points = np.array([p["world_point"] for p in selected])
            for i, label in nodes:
                v = views[i]
                pose = v["world_from_camera"]
                camera = (points - pose[:3, 3]) @ pose[:3, :3]
                front = camera[:, 2] > 1e-6
                uv = np.zeros((len(points), 2), int)
                uv[front] = np.rint(
                    camera[front, :2] / camera[front, 2:] * np.diag(v["K"])[:2] + v["K"][:2, 2]
                ).astype(int)
                inside = (
                    front
                    & (uv[:, 0] >= 0)
                    & (uv[:, 0] < v["labels"].shape[1])
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < v["labels"].shape[0])
                )
                margin = int(np.ceil(policy.max_reprojection_px))
                target = (
                    cv2.dilate(
                        (v["labels"] == label).astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8)
                    )
                    > 0
                )
                hit = target[uv[inside, 1], uv[inside, 0]]
                background = (v["labels"][uv[inside, 1], uv[inside, 0]] == 0) & ~hit
                known = hit | background
                fraction = float(background.sum() / max(1, known.sum()))
                mask_checks.append(dict(view=i, background_fraction=fraction, known_points=int(known.sum())))
                if fraction > max_background_fraction:
                    reason = "landmarks contradict observed background in another view"
        record["mask_checks"] = mask_checks
        if reason:
            record["reason"] = reason
            rejected.append(record)
        else:
            record["evidence"] = (
                "Actual feature tracks triangulated across at least three views; per-view direct support and background checks"
            )
            accepted[f"landmark_object_{len(accepted):03d}"] = record
    return dict(
        tracks=accepted,
        rejected=rejected,
        rejected_features=rejected_features,
        policy=dict(
            **asdict(policy), min_landmarks=min_landmarks, max_background_fraction=max_background_fraction
        ),
        scope="Multi-view RGB instance hypotheses; no sensor-depth confirmation or complete reconstruction",
        inventory_complete=False,
        native_depth_validated=False,
    )
