"""Optional SuperPoint/LightGlue adapter with calibrated instance-level checks.

The learned matcher proposes image correspondences. Metric camera checks and
unambiguous all-pairs association remain mandatory. Neural confidence is not a
calibrated probability of object identity or accurate surface reconstruction.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from lidar_video.rgb_association import RGBPolicy, associate_rgb, extend_tracks, triangulate_matches


def interior_labels(pixels, labels, margin=2.0):
    """Label only keypoints inside an instance core; never assign image background."""
    pixels, labels = np.asarray(pixels), np.asarray(labels)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
        raise ValueError("Finite Nx2 keypoints required")
    if (
        labels.ndim != 2
        or labels.dtype.kind not in "ui"
        or np.any(labels < 0)
        or not np.isfinite(margin)
        or margin <= 0
    ):
        raise ValueError("Nonnegative indexed masks and a positive margin required")
    core = np.zeros_like(labels)
    for label in np.unique(labels):
        if label:
            distance = cv2.distanceTransform(np.pad((labels == label).astype(np.uint8), 1), cv2.DIST_L2, 5)[
                1:-1, 1:-1
            ]
            core[distance >= margin] = label
    coords = np.rint(pixels).astype(int)
    inside = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < labels.shape[1])
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < labels.shape[0])
    )
    result = np.zeros(len(pixels), labels.dtype)
    result[inside] = core[coords[inside, 1], coords[inside, 0]]
    return result


def geometric_pairs(lf, rf, matches, scores, left, right, policy=None, *, dominance=0.7, min_score=0.1):
    """Screen learned matches, then require dominant bidirectional instance links."""
    policy = policy or RGBPolicy()
    matches, scores = np.asarray(matches), np.asarray(scores, float)
    if (
        matches.ndim != 2
        or matches.shape[1] != 2
        or matches.dtype.kind not in "ui"
        or scores.shape != (len(matches),)
    ):
        raise ValueError("Integer Nx2 matches and one score per match required")
    if (
        not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
        or not 0.5 < dominance <= 1
        or not np.isfinite(min_score)
        or not 0 <= min_score <= 1
    ):
        raise ValueError("Invalid match scores or instance dominance")
    if np.any(matches < 0) or (
        len(matches)
        and (matches[:, 0].max() >= len(lf["pixels"]) or matches[:, 1].max() >= len(rf["pixels"]))
    ):
        raise ValueError("Match index outside feature arrays")
    # Refuse many-to-one feature mappings rather than inflate support counts.
    if any(len(np.unique(matches[:, axis])) != len(matches) for axis in [0, 1]):
        raise ValueError("Learned feature matches must be one-to-one")
    keep = (lf["labels"][matches[:, 0]] > 0) & (rf["labels"][matches[:, 1]] > 0) & (scores >= min_score)
    matches, scores = matches[keep], scores[keep]
    if not len(matches):
        return []
    a, b = lf["pixels"][matches[:, 0]], rf["pixels"][matches[:, 1]]
    # Distinct feature indices at the same pixel are not independent witnesses.
    seen_a, seen_b, distinct = set(), set(), []
    for n, (x, y) in enumerate(zip(np.rint(a).astype(int), np.rint(b).astype(int))):
        if tuple(x) not in seen_a and tuple(y) not in seen_b:
            distinct.append(n)
            seen_a.add(tuple(x))
            seen_b.add(tuple(y))
    matches, scores, a, b = matches[distinct], scores[distinct], a[distinct], b[distinct]
    try:
        points, valid = triangulate_matches(a, b, left, right, policy)
    except ValueError as error:
        if str(error) == "Insufficient camera baseline":
            return []
        raise
    matches, scores, a, b, points = matches[valid], scores[valid], a[valid], b[valid], points[valid]
    ll, rl = lf["labels"][matches[:, 0]], rf["labels"][matches[:, 1]]
    records = []
    for left_label, right_label in sorted(set(zip(ll.tolist(), rl.tolist()))):
        ids = np.flatnonzero((ll == left_label) & (rl == right_label))
        fractions = [
            float(len(ids) / np.count_nonzero(ll == left_label)),
            float(len(ids) / np.count_nonzero(rl == right_label)),
        ]
        spread = [float(np.linalg.norm(np.ptp(p[ids], axis=0))) for p in [a, b]]
        records.append(
            dict(
                left_label=int(left_label),
                right_label=int(right_label),
                matches=len(ids),
                spread_px=spread,
                instance_fractions=fractions,
                passed=len(ids) >= policy.min_matches
                and min(spread) >= policy.min_spread_px
                and min(fractions) >= dominance,
                left_pixels=a[ids].tolist(),
                right_pixels=b[ids].tolist(),
                world_points=points[ids].tolist(),
                neural_scores=scores[ids].tolist(),
            )
        )
    return records


class LightGlueProvider:
    """Loads optional model dependencies only when learned inference is requested."""

    def __init__(self, device="cpu", *, max_keypoints=4096, checkpoint=None):
        import torch
        from lightglue import LightGlue, SuperPoint

        if (
            device not in {"cpu", "mps", "cuda"}
            or type(max_keypoints) is not int
            or not 256 <= max_keypoints <= 8192
        ):
            raise ValueError("Explicit device and bounded keypoint budget required")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        self.torch, self.device, self.checkpoint = torch, device, checkpoint
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        self.matcher = (
            LightGlue(
                features="superpoint",
                flash=False,
                mp=False,
                depth_confidence=-1,
                width_confidence=-1,
                filter_threshold=0.1,
            )
            .eval()
            .to(device)
        )
        self.frames, self.pairs = 0, 0

    def extract(self, rgb, labels, policy):
        torch = self.torch
        with torch.inference_mode():
            tensor = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float().to(self.device) / 255
            features = self.extractor.extract(tensor, resize=None)
        pixels = features["keypoints"][0].cpu().numpy()
        result = dict(
            pixels=pixels, labels=interior_labels(pixels, labels, policy.core_margin_px), model=features
        )
        if self.checkpoint:
            self.checkpoint(
                "features",
                self.frames,
                dict(keypoints=len(pixels), interior_keypoints=int(np.count_nonzero(result["labels"]))),
            )
        self.frames += 1
        return result

    def match(self, lf, rf, left, right, policy):
        with self.torch.inference_mode():
            matched = self.matcher(dict(image0=lf["model"], image1=rf["model"]))
        pairs = matched["matches"][0].cpu().numpy()
        scores = matched["scores"][0].cpu().numpy()
        result = geometric_pairs(lf, rf, pairs, scores, left, right, policy)
        if self.checkpoint:
            self.checkpoint(
                "pair",
                self.pairs,
                dict(
                    left_source=left["source_frame"],
                    right_source=right["source_frame"],
                    raw_matches=len(pairs),
                    checks=result,
                ),
            )
        self.pairs += 1
        return result


def main():
    import argparse
    import hashlib
    import subprocess
    import sys

    from PIL import Image

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = p.parse_args()
    config = json.loads(args.manifest.read_text())
    if config.get("static_scene") is not True:
        raise ValueError("Explicit static_scene=True required")
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    sys.path.insert(0, str(args.upstream.resolve()))
    views, inputs = [], []
    for v in config["views"]:
        rgb, labels = args.manifest.parent / v["rgb"], args.manifest.parent / v["labels"]
        views.append(
            dict(v, rgb=np.array(Image.open(rgb).convert("RGB")), labels=np.array(Image.open(labels)))
        )
        inputs.append(
            dict(
                source_frame=v["source_frame"],
                rgb_sha256=hashlib.sha256(rgb.read_bytes()).hexdigest(),
                labels_sha256=hashlib.sha256(labels.read_bytes()).hexdigest(),
            )
        )
    args.output.mkdir(parents=True, exist_ok=False)
    plan = dict(
        manifest=config,
        upstream_revision=revision,
        device=args.device,
        inputs=inputs,
        policy=vars(RGBPolicy()),
        extractor="SuperPoint",
        max_keypoints=4096,
        resize=None,
        matcher="LightGlue",
        flash=False,
        mixed_precision=False,
        depth_confidence=-1,
        width_confidence=-1,
        filter_threshold=0.1,
        min_instance_dominance=0.7,
        independent_scene_evaluation=False,
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))

    def checkpoint(stage, index, value):
        (args.output / f"{stage}-{index:03d}.json").write_text(json.dumps(value, indent=2))
        print(stage, index, value.get("raw_matches", value.get("interior_keypoints")), flush=True)

    try:
        provider = LightGlueProvider(args.device, checkpoint=checkpoint)
        result = associate_rgb(
            views, static_scene=True, feature_provider=provider.extract, pair_provider=provider.match
        )
        result["method"] = "SuperPoint-LightGlue with calibrated geometry and dominant instance links"
        (args.output / "associations.json").write_text(json.dumps(result, indent=2))
        (args.output / "extensions.json").write_text(json.dumps(extend_tracks(views, result), indent=2))
        import torch

        weights = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(torch.hub.get_dir()) / "checkpoints").glob("*.pth")
        }
        (args.output / "weight-hashes.json").write_text(json.dumps(weights, indent=2))
        print("Completed", len(result["tracks"]), "tracks", flush=True)
    except Exception as exc:
        (args.output / "failure.json").write_text(
            json.dumps(dict(error_type=type(exc).__name__, reason=str(exc)), indent=2)
        )
        raise


if __name__ == "__main__":
    main()
