"""RGB correspondence hypotheses independent of unreliable object depth returns.

SIFT features, supplied metric cameras and instance masks constrain association.
No scene inventory, object class, support plane or incumbent mesh is consulted.
Specular appearance, repeated texture and camera error remain failure modes.
"""

from dataclasses import asdict, dataclass
from itertools import product

import cv2
import networkx as nx
import numpy as np

from lidar_video.multiview_shape import checked_views


@dataclass(frozen=True)
class RGBPolicy:
    geometry_guided: bool = False
    max_descriptor_distance: float = 0.5
    max_features: int = 8000
    ratio: float = 0.75
    min_matches: int = 4
    min_baseline_m: float = 0.01
    min_angle_degrees: float = 1.0
    max_reprojection_px: float = 2.0
    min_spread_px: float = 3.0
    core_margin_px: float = 2.0

    def __post_init__(self):
        if (
            type(self.geometry_guided) is not bool
            or not np.isfinite(self.max_descriptor_distance)
            or not 0 < self.max_descriptor_distance <= 1
        ):
            raise ValueError("Invalid descriptor guidance policy")
        if type(self.max_features) is not int or not 100 <= self.max_features <= 20000:
            raise ValueError("Invalid feature budget")
        if type(self.min_matches) is not int or not 3 <= self.min_matches <= 100:
            raise ValueError("At least three unique matches required")
        values = [
            self.ratio,
            self.min_baseline_m,
            self.min_angle_degrees,
            self.max_reprojection_px,
            self.min_spread_px,
            self.core_margin_px,
        ]
        if not np.isfinite(values).all() or not 0 < self.ratio < 1 or min(values[1:]) <= 0:
            raise ValueError("Invalid RGB association policy")


def features(rgb, labels, policy=None):
    """Extract features whose full nominal keypoint radius lies within one mask."""
    policy = policy or RGBPolicy()
    rgb, labels = np.asarray(rgb), np.asarray(labels)
    if (
        rgb.dtype != np.uint8
        or rgb.shape != (*labels.shape, 3)
        or labels.dtype.kind not in "ui"
        or labels.ndim != 2
    ):
        raise ValueError("Registered RGB uint8 image and indexed mask required")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.SIFT_create(nfeatures=policy.max_features)
    keypoints, descriptors = detector.detectAndCompute(gray, (labels > 0).astype(np.uint8) * 255)
    radius = np.zeros(labels.shape, np.float32)
    for label in np.unique(labels):
        if label:
            mask = (labels == label).astype(np.uint8)
            distance = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            radius[mask > 0] = distance[mask > 0]
    selected, seen = [], set()
    for i, key in enumerate(keypoints):
        x, y = np.rint(key.pt).astype(int)
        if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]):
            continue
        if (x, y) in seen or radius[y, x] < max(policy.core_margin_px, key.size / 2):
            continue
        selected.append(i)
        seen.add((x, y))
    if not selected:
        return dict(
            pixels=np.empty((0, 2)), labels=np.empty(0, int), descriptors=np.empty((0, 128), np.float32)
        )
    pixels = np.array([keypoints[i].pt for i in selected])
    coords = np.rint(pixels).astype(int)
    desc = descriptors[selected]
    # RootSIFT normalization; no feature or geometry learned on the evaluation set.
    desc = np.sqrt(desc / np.maximum(desc.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)
    return dict(pixels=pixels, labels=labels[coords[:, 1], coords[:, 0]], descriptors=desc)


def triangulate_matches(left_pixels, right_pixels, left, right, policy=None):
    """Triangulate known pixel pairs; reject negative depth and weak parallax."""
    policy = policy or RGBPolicy()
    poses = [np.asarray(v["world_from_camera"], float) for v in [left, right]]
    if np.linalg.norm(poses[0][:3, 3] - poses[1][:3, 3]) < policy.min_baseline_m:
        raise ValueError("Insufficient camera baseline")
    pixels = [np.asarray(p, float) for p in [left_pixels, right_pixels]]
    if (
        pixels[0].shape != pixels[1].shape
        or pixels[0].ndim != 2
        or pixels[0].shape[1] != 2
        or not all(np.isfinite(p).all() for p in pixels)
    ):
        raise ValueError("Finite paired pixel coordinates required")
    if not len(pixels[0]):
        return np.empty((0, 3)), np.empty(0, bool)
    cameras = [np.linalg.inv(p)[:3] for p in poses]
    projections = [np.asarray(v["K"]) @ c for v, c in zip([left, right], cameras)]
    points_h = cv2.triangulatePoints(*projections, pixels[0].T, pixels[1].T).T
    finite = np.abs(points_h[:, 3]) > 1e-12
    points = np.full((len(points_h), 3), np.nan)
    points[finite] = points_h[finite, :3] / points_h[finite, 3:]
    valid = finite & np.isfinite(points).all(axis=1)
    for projection, camera, expected in zip(projections, cameras, pixels):
        homogeneous = np.column_stack([points, np.ones(len(points))])
        projected = homogeneous @ projection.T
        depth = homogeneous @ camera[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            error = np.linalg.norm(projected[:, :2] / projected[:, 2:] - expected, axis=1)
        valid &= (depth > 0) & (error <= policy.max_reprojection_px)
    rays = [points - p[:3, 3] for p in poses]
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.sum(rays[0] * rays[1], axis=1) / (
            np.linalg.norm(rays[0], axis=1) * np.linalg.norm(rays[1], axis=1)
        )
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    valid &= angle >= policy.min_angle_degrees
    return points, valid


def guided_matches(lf, rf, left, right, policy):
    """Mutual ratio tests within calibrated epipolar bands, with an absolute cap."""
    if len(lf["pixels"]) * len(rf["pixels"]) > 4_000_000:
        raise ValueError("Guided feature distance matrix exceeds the pair budget")
    relative = np.linalg.inv(right["world_from_camera"]) @ left["world_from_camera"]
    x, y, z = relative[:3, 3]
    if np.linalg.norm([x, y, z]) < policy.min_baseline_m:
        return {}, {}
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    fundamental = np.linalg.inv(right["K"]).T @ skew @ relative[:3, :3] @ np.linalg.inv(left["K"])
    a, b = [np.column_stack([f["pixels"], np.ones(len(f["pixels"]))]) for f in [lf, rf]]
    lines_right, lines_left = a @ fundamental.T, b @ fundamental
    residual = np.abs(lines_right @ b.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        forward_error = residual / np.linalg.norm(lines_right[:, :2], axis=1)[:, None]
        backward_error = residual / np.linalg.norm(lines_left[:, :2], axis=1)[None, :]
    permitted = (forward_error <= policy.max_reprojection_px) & (backward_error <= policy.max_reprojection_px)
    da, db = lf["descriptors"], rf["descriptors"]
    distance = np.sqrt(
        np.maximum(np.sum(da * da, axis=1)[:, None] + np.sum(db * db, axis=1)[None, :] - 2 * da @ db.T, 0)
    )
    distance[~permitted] = np.inf

    def nearest(matrix):
        result = {}
        for i, row in enumerate(matrix):
            order = np.argsort(row)[:2]
            if (
                len(order) == 2
                and np.isfinite(row[order]).all()
                and row[order[0]] < policy.ratio * row[order[1]]
                and row[order[0]] <= policy.max_descriptor_distance
            ):
                result[i] = int(order[0])
        return result

    return nearest(distance), nearest(distance.T)


def pair_evidence(left_features, right_features, left, right, policy=None):
    """Mutual descriptor ratio matches followed by calibrated geometric checks."""
    policy = policy or RGBPolicy()
    lf, rf = left_features, right_features
    if min(len(lf["pixels"]), len(rf["pixels"])) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def ratio(a, b):
        return {
            m.queryIdx: m.trainIdx
            for neighbors in matcher.knnMatch(a, b, k=2)
            if len(neighbors) == 2
            for m, n in [neighbors]
            if m.distance < policy.ratio * n.distance
        }

    if policy.geometry_guided:
        forward, backward = guided_matches(lf, rf, left, right, policy)
    else:
        forward = ratio(lf["descriptors"], rf["descriptors"])
        backward = ratio(rf["descriptors"], lf["descriptors"])
    pairs = np.array([(i, j) for i, j in forward.items() if backward.get(j) == i], int).reshape(-1, 2)
    if not len(pairs):
        return []
    a, b = lf["pixels"][pairs[:, 0]], rf["pixels"][pairs[:, 1]]
    try:
        points, valid = triangulate_matches(a, b, left, right, policy)
    except ValueError as exc:
        if str(exc) == "Insufficient camera baseline":
            return []
        raise
    pairs, a, b, points = pairs[valid], a[valid], b[valid], points[valid]
    groups = {}
    for n, (i, j) in enumerate(pairs):
        key = (int(lf["labels"][i]), int(rf["labels"][j]))
        groups.setdefault(key, []).append(n)
    result = []
    for (ll, rl), ids in sorted(groups.items()):
        spread = [float(np.linalg.norm(np.ptp(p[ids], axis=0))) for p in [a, b]]
        result.append(
            dict(
                left_label=ll,
                right_label=rl,
                matches=len(ids),
                spread_px=spread,
                passed=len(ids) >= policy.min_matches and min(spread) >= policy.min_spread_px,
                left_pixels=a[ids].tolist(),
                right_pixels=b[ids].tolist(),
                world_points=points[ids].tolist(),
            )
        )
    return result


def associate_rgb(views, *, static_scene=False, policy=None):
    """Keep only unambiguous complete pairwise-supported observation groups.

    Tracks are hypotheses, not depth-validated shapes or a complete scene graph.
    Missing pair evidence is not converted into a positive correspondence.
    """
    if static_scene is not True:
        raise ValueError("Explicit static_scene=True required")
    policy = policy or RGBPolicy()
    if len(views) < 2:
        raise ValueError("At least two source views required")
    # checked_views also validates camera calibration and unique source IDs.
    checked = []
    for v in views:
        positive = np.unique(v["labels"])
        positive = positive[positive > 0]
        checked.append(dict(v, label=int(positive[0]) if len(positive) else 1))
    views = checked_views(checked, allow_empty=True)
    extracted = [features(v["rgb"], v["labels"], policy) for v in views]
    graph = nx.Graph()
    for i, v in enumerate(views):
        graph.add_nodes_from((i, int(label)) for label in np.unique(v["labels"]) if label > 0)
    checks = []
    for i, left in enumerate(views):
        for j in range(i + 1, len(views)):
            for evidence in pair_evidence(extracted[i], extracted[j], left, views[j], policy):
                evidence.update(left_view=i, right_view=j)
                checks.append(evidence)
                if evidence["passed"]:
                    graph.add_edge((i, evidence["left_label"]), (j, evidence["right_label"]))
    tracks, rejected = {}, []
    for component in sorted(nx.connected_components(graph), key=lambda c: sorted(c)):
        nodes = sorted(component)
        reason = None
        if len(nodes) < 2:
            reason = "no corroborating RGB features"
        elif len({i for i, _ in nodes}) != len(nodes):
            reason = "ambiguous many-to-one RGB association"
        elif any(not graph.has_edge(a, b) for k, a in enumerate(nodes) for b in nodes[k + 1 :]):
            reason = "RGB chain lacks all-pairs corroboration"
        observations = [
            dict(view=i, frame=views[i].get("frame", i), source_frame=views[i]["source_frame"], label=label)
            for i, label in nodes
        ]
        if reason:
            rejected.append(dict(observations=observations, reason=reason))
        else:
            tracks[f"rgb_object_{len(tracks):03d}"] = dict(
                observations=observations,
                evidence="Mutual RGB features with calibrated triangulation; no sensor depth confirmation",
            )
    return dict(
        policy=asdict(policy),
        tracks=tracks,
        pair_checks=checks,
        rejected=rejected,
        feature_counts=[len(f["pixels"]) for f in extracted],
        scope="Development RGB association hypotheses; supplied cameras and static appearance assumptions",
        inventory_complete=False,
        native_depth_validated=False,
    )


def extend_tracks(views, result, *, min_fraction=0.8, min_points=4, min_spread_px=3.0):
    """Project seed RGB landmarks into additional masks with explicit weaker scope.

    No new descriptor match is claimed by extension. Conflicting seed claims to
    the same observation are rejected, not merged or silently assigned. Never
    use these projected masks as independently matched 3D surface measurements.
    """
    if (
        not 0.5 < min_fraction <= 1
        or type(min_points) is not int
        or min_points < 3
        or not np.isfinite(min_spread_px)
        or min_spread_px <= 0
    ):
        raise ValueError("Invalid landmark-extension policy")
    checked = []
    for v in views:
        labels = np.asarray(v["labels"])
        positive = np.unique(labels)
        positive = positive[positive > 0]
        checked.append(dict(v, label=int(positive[0]) if len(positive) else 1))
    views = checked_views(checked, allow_empty=True)
    seed_policy = RGBPolicy(**result.get("policy", {}))
    proposals = {}
    for name, track in result["tracks"].items():
        if any(
            type(o["view"]) is not int
            or not 0 <= o["view"] < len(views)
            or o["source_frame"] != views[o["view"]]["source_frame"]
            or not np.any(views[o["view"]]["labels"] == o["label"])
            for o in track["observations"]
        ):
            raise ValueError("Seed observation does not match supplied views")

        seeds = {(o["view"], o["label"]) for o in track["observations"]}
        seed_checks = [
            p
            for p in result["pair_checks"]
            if p["passed"]
            and (p["left_view"], p["left_label"]) in seeds
            and (p["right_view"], p["right_label"]) in seeds
        ]
        if not seed_checks:
            continue
        raw_points = np.concatenate([np.asarray(p["world_points"], float) for p in seed_checks])
        if raw_points.ndim != 2 or raw_points.shape[1] != 3 or not np.isfinite(raw_points).all():
            raise ValueError("Finite triangulated 3D landmarks required")
        points, unique_ids = np.unique(np.round(raw_points, 7), axis=0, return_index=True)
        tolerance = float(
            result.get("policy", {}).get("max_reprojection_px", RGBPolicy().max_reprojection_px)
        )
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Positive finite pixel perturbation required")
        perturbed = []
        for signs in product([-1, 1], repeat=4):
            variants = []
            for check in seed_checks:
                left_pixels = np.asarray(check["left_pixels"], float) + tolerance * np.array(signs[:2])
                right_pixels = np.asarray(check["right_pixels"], float) + tolerance * np.array(signs[2:])
                variant, _ = triangulate_matches(
                    left_pixels,
                    right_pixels,
                    views[check["left_view"]],
                    views[check["right_view"]],
                    seed_policy,
                )
                # Perturbations probe sensitivity, not additional correspondences.
                variants.append(variant)
            perturbed.append(np.concatenate(variants)[unique_ids])
        observations = [dict(o, evidence="seed mutual RGB feature matches") for o in track["observations"]]
        tests = []
        for i, v in enumerate(views):
            if any(seed[0] == i for seed in seeds):
                continue
            pose, k, labels = np.asarray(v["world_from_camera"]), np.asarray(v["K"]), np.asarray(v["labels"])
            camera = (points - pose[:3, 3]) @ pose[:3, :3]
            front = camera[:, 2] > 1e-6
            uv = np.zeros((len(points), 2), int)
            uv[front] = np.rint(camera[front, :2] / camera[front, 2:] * np.diag(k)[:2] + k[:2, 2]).astype(int)
            inside = (
                front
                & (uv[:, 0] >= 0)
                & (uv[:, 1] >= 0)
                & (uv[:, 0] < labels.shape[1])
                & (uv[:, 1] < labels.shape[0])
            )
            if inside.sum() < min_points:
                tests.append(dict(view=i, reason="insufficient in-frame landmarks"))
                continue
            hits_all = np.zeros(len(points), labels.dtype)
            hits_all[inside] = labels[uv[inside, 1], uv[inside, 0]]
            robust = inside.copy()
            for variant in perturbed:
                pc = (variant - pose[:3, 3]) @ pose[:3, :3]
                good = np.isfinite(pc).all(axis=1) & (pc[:, 2] > 1e-6)
                pixels = np.zeros((len(points), 2), int)
                pixels[good] = np.rint(pc[good, :2] / pc[good, 2:] * np.diag(k)[:2] + k[:2, 2]).astype(int)
                good &= (
                    (pixels[:, 0] >= 0)
                    & (pixels[:, 0] < labels.shape[1])
                    & (pixels[:, 1] >= 0)
                    & (pixels[:, 1] < labels.shape[0])
                )
                variant_labels = np.zeros(len(points), labels.dtype)
                variant_labels[good] = labels[pixels[good, 1], pixels[good, 0]]
                robust &= good & (variant_labels == hits_all)
            hits = np.where(robust[inside], hits_all[inside], 0)
            ids, counts = np.unique(hits[hits > 0], return_counts=True)
            eligible = []
            for label, count in zip(ids, counts):
                projected = uv[inside][hits == label]
                unique = np.unique(projected, axis=0)
                spread = float(np.linalg.norm(np.ptp(unique, axis=0)))
                fraction = float(count / len(points))
                if len(unique) >= min_points and fraction >= min_fraction and spread >= min_spread_px:
                    eligible.append(
                        dict(label=int(label), fraction=fraction, unique_pixels=len(unique), spread_px=spread)
                    )
            tests.append(dict(view=i, candidates=eligible))
            if len(eligible) == 1:
                observations.append(
                    dict(
                        view=i,
                        frame=v.get("frame", i),
                        source_frame=v["source_frame"],
                        **eligible[0],
                        evidence="projected seed RGB landmarks; no new descriptor corroboration",
                    )
                )
        proposals[name] = dict(
            observations=observations,
            seed_observations=track["observations"],
            landmark_count=len(points),
            pixel_perturbation_px=tolerance,
            perturbation_corners=16,
            checks=tests,
            accepted=True,
        )
    claims = {}
    for name, track in proposals.items():
        for o in track["observations"]:
            claims.setdefault((o["view"], o["label"]), []).append(name)
    for observation, names in claims.items():
        if len(names) > 1:
            for name in names:
                proposals[name]["accepted"] = False
                proposals[name].setdefault("conflicts", []).append(
                    dict(view=observation[0], label=observation[1], claimants=names)
                )
    return dict(
        policy=dict(min_fraction=min_fraction, min_points=min_points, min_spread_px=min_spread_px),
        tracks=proposals,
        scope="Mask-projection extensions screened at all 16 pixel-error corners; sensitivity check, not a confidence bound or independent surface matches",
        native_depth_validated=False,
        inventory_complete=False,
    )


def main():
    """Portable manifest runner; keep RGB and depth evidence explicitly separate."""
    import argparse
    import hashlib
    import json
    from pathlib import Path

    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    views = []
    inputs = []
    for view in config["views"]:
        rgb_path, labels_path = root / view["rgb"], root / view["labels"]
        views.append(
            dict(
                view,
                rgb=np.asarray(Image.open(rgb_path).convert("RGB")),
                labels=np.asarray(Image.open(labels_path)),
            )
        )
        inputs.append(
            dict(
                source_frame=view["source_frame"],
                rgb_sha256=hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
                labels_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            )
        )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(dict(config, input_hashes=inputs), indent=2) + "\n")
    try:
        result = associate_rgb(
            views,
            static_scene=config.get("static_scene", False),
            policy=RGBPolicy(**config.get("policy", {})),
        )
        (args.output / "associations.json").write_text(json.dumps(result, indent=2) + "\n")
        if config.get("extend_tracks", True):
            extended = extend_tracks(views, result)
            (args.output / "extensions.json").write_text(json.dumps(extended, indent=2) + "\n")
    except ValueError as error:
        (args.output / "failure.json").write_text(
            json.dumps(dict(error=str(error), inventory_complete=False), indent=2) + "\n"
        )
        raise
    print(json.dumps(dict(rgb_seed_tracks=len(result["tracks"]), inventory_complete=False)))


if __name__ == "__main__":
    main()
