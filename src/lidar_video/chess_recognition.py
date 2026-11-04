"""Synthetic-only RGB-D chess classifier; calibrated board pose is required.

Whole scenes, not crops or adjacent frames, are split before training. No FEN or
labels enter feature extraction. Forest vote fractions are not calibrated certainty.
"""

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from lidar_video.capture import camera_points, validate, write_json
from lidar_video.chess_data import SYMBOLS, random_placement, render_board


def features(capture, calibration, cells=12):
    validate(capture)
    transform = np.asarray(calibration["board_from_world"], float)
    square_size = float(calibration["square_size_m"])
    if (
        transform.shape != (4, 4)
        or not np.isfinite(square_size)
        or square_size <= 0
        or not np.isfinite(transform).all()
    ):
        raise ValueError("Invalid board calibration")
    if (
        not np.allclose(transform[3], [0, 0, 0, 1])
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
        or np.linalg.det(transform[:3, :3]) < 0
    ):
        raise ValueError("board_from_world must be a proper rigid transform")
    heights = np.zeros((64, cells * cells))
    colors, zs, seen = [[] for _ in range(64)], [[] for _ in range(64)], np.zeros((64, cells * cells), bool)
    for i, depth in enumerate(capture["depth"]):
        k = capture["K"] if capture["K"].ndim == 2 else capture["K"][i]
        points = camera_points(depth, k).reshape(-1, 3)
        pose = transform @ capture["world_from_camera"][i]
        points = points @ pose[:3, :3].T + pose[:3, 3]
        rgb = capture["rgb"][i].reshape(-1, 3) / 255
        valid = (depth.ravel() > 0) & (capture["confidence"][i].ravel() >= 1)
        uv = points[:, :2] / square_size + 4
        valid &= np.all((uv >= 0) & (uv < 8), axis=1) & (points[:, 2] > -0.004) & (points[:, 2] < 0.15)
        uv, points, rgb = uv[valid], points[valid], rgb[valid]
        grid = np.floor(uv).astype(int)
        square = grid[:, 0] + 8 * grid[:, 1]
        sub = np.minimum(((uv - grid) * cells).astype(int), cells - 1)
        cell = sub[:, 0] + cells * sub[:, 1]
        seen[square, cell] = True
        np.maximum.at(heights, (square, cell), points[:, 2])
        for s in range(64):
            mask = (square == s) & (points[:, 2] > 0.006)
            colors[s].extend(rgb[mask])
            zs[s].extend(points[mask, 2])
    rows = []
    for s in range(64):
        color = np.quantile(colors[s], [0.25, 0.5, 0.75], axis=0).ravel() if colors[s] else np.zeros(9)
        height = np.quantile(zs[s], [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]) if zs[s] else np.zeros(6)
        rows.append(np.r_[heights[s], color, height, (heights[s] > 0.006).mean()])
    return np.array(rows, dtype=np.float32), seen.mean(axis=1)


def save_forest(forest, path):
    # Numeric NPZ, never pickle or object arrays. Each tree uses global child indices.
    children_left, children_right, feature, threshold, values, roots = [], [], [], [], [], []
    offset = 0
    for estimator in forest.estimators_:
        tree = estimator.tree_
        roots.append(offset)
        children_left.extend(np.where(tree.children_left >= 0, tree.children_left + offset, -1))
        children_right.extend(np.where(tree.children_right >= 0, tree.children_right + offset, -1))
        feature.extend(tree.feature)
        threshold.extend(tree.threshold)
        value = tree.value[:, 0, :]
        values.extend(value / value.sum(axis=1, keepdims=True))
        offset += tree.node_count
    np.savez_compressed(
        path,
        left=children_left,
        right=children_right,
        feature=feature,
        threshold=threshold,
        values=values,
        roots=roots,
        classes=forest.classes_,
        n_features=forest.n_features_in_,
    )


def probabilities(model_path, x):
    with np.load(model_path, allow_pickle=False) as m:
        if x.ndim != 2 or x.shape[1] != int(m["n_features"]) or not np.isfinite(x).all():
            raise ValueError("Feature shape or values do not match model")
        scores = np.zeros((len(x), len(m["classes"])))
        for root in m["roots"]:
            nodes = np.full(len(x), root, dtype=int)
            for _ in range(len(m["left"])):
                active = m["left"][nodes] >= 0
                if not active.any():
                    break
                n = nodes[active]
                right = x[active, m["feature"][n]] > m["threshold"][n]
                nodes[active] = np.where(right, m["right"][n], m["left"][n])
            else:
                raise ValueError("Invalid cyclic forest")
            scores += m["values"][nodes]
        return scores / len(m["roots"]), m["classes"].copy()


def recognize(capture, calibration, model_path, threshold=0.55):
    x, coverage = features(capture, calibration)
    scores, classes = probabilities(model_path, x)
    best = np.argmax(scores, axis=1)
    ranked = np.sort(scores, axis=1)
    accepted = (
        (scores.max(axis=1) >= threshold) & ((ranked[:, -1] - ranked[:, -2]) >= 0.15) & (coverage > 0.50)
    )
    return dict(
        symbols=[str(classes[c]) if ok else "?" for c, ok in zip(best, accepted)],
        raw_symbols=[str(classes[c]) for c in best],
        vote_fraction=scores.max(axis=1).tolist(),
        observed_cell_fraction=coverage.tolist(),
        unknown_count=int((~accepted).sum()),
        evidence="synthetic-trained RGB-D forest with supplied board calibration",
        uncertainty="vote fraction is uncalibrated; unknown is abstention, not reliable out-of-distribution detection",
        board_orientation=calibration.get("orientation", dict(resolved=True, source="supplied board axes")),
    )


def benchmark(out, train_scenes=24, test_scenes=8):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    calibration = dict(square_size_m=0.045, board_from_world=np.eye(4).tolist())
    datasets = {}
    for split, count, base in (("train", train_scenes, 10000), ("test", test_scenes, 90000)):
        xs, ys, coverage = [], [], []
        for i in range(count):
            placement = random_placement(base + i)
            capture = render_board(out / "dataset" / split / f"scene-{i:03d}", placement, base + i)
            x, visible = features(capture, calibration)
            xs.extend(x)
            ys.extend(placement)
            coverage.extend(visible)
            print(f"Rendered {split} scene {i + 1}/{count}", flush=True)
        datasets[split] = np.array(xs), np.array(ys), np.array(coverage)
    x, y, _ = datasets["train"]
    forest = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=1, class_weight="balanced", random_state=7, n_jobs=2
    ).fit(x, y)
    model = out / "forest.npz"
    save_forest(forest, model)
    x, y, coverage = datasets["test"]
    scores, classes = probabilities(model, x)
    predicted = classes[scores.argmax(axis=1)]
    ranked = np.sort(scores, axis=1)
    accepted = (scores.max(axis=1) >= 0.55) & ((ranked[:, -1] - ranked[:, -2]) >= 0.15) & (coverage > 0.5)
    occupied = y != "."
    result = dict(
        train_scenes=train_scenes,
        test_scenes=test_scenes,
        test_squares=len(y),
        labels=list(SYMBOLS),
        confusion_matrix=confusion_matrix(y, predicted, labels=list(SYMBOLS)).tolist(),
        raw_square_accuracy=float(np.mean(predicted == y)),
        raw_occupied_accuracy=float(np.mean(predicted[occupied] == y[occupied])),
        accepted_fraction=float(accepted.mean()),
        accepted_accuracy=float(np.mean(predicted[accepted] == y[accepted])) if accepted.any() else None,
        exact_board_accuracy=float(np.mean(np.all((predicted == y).reshape(-1, 64), axis=1))),
        exported_model_max_error=float(np.abs(scores - forest.predict_proba(x)).max()),
        split="disjoint scene seeds; no frame or crop leakage; shared procedural geometry style",
        limits="synthetic-only; known board pose; no unseen chess-set or real camera validation",
    )
    np.savez_compressed(
        out / "features.npz",
        train_x=datasets["train"][0],
        train_y=datasets["train"][1],
        test_x=x,
        test_y=y,
        test_scores=scores,
        test_coverage=coverage,
    )
    write_json(out / "benchmark.json", result)
    return result
