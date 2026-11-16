"""Development-only multi-view generic detection with preserved raw proposals.

Separate discovery from expensive mesh generation. Counts are diagnostics, never
inventory completeness or detection accuracy. No category oracle is used here.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from lidar_video.capture import load_capture, normalize, write_json
from lidar_video.general_perception import disambiguate_masks
from lidar_video.general_run import digest
from lidar_video.general_scene import InstancePolicy, associate_instances


def associate_condition(data, raw, out, duplicate_iou):
    """Keep raw indices beside cleaned labels so exclusions remain inspectable."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cleaned, cleanup, mapping = {}, {}, {}
    for frame, items in sorted(raw.items()):
        cleaned[frame], cleanup[str(frame)] = disambiguate_masks(items, duplicate_iou=duplicate_iou)
        labels = np.zeros(data["depth"][frame].shape, np.uint16)
        if len(cleaned[frame]) > 65535:
            raise ValueError("Too many instance labels")
        mapping[str(frame)] = []
        for label, item in enumerate(cleaned[frame], 1):
            labels[item["mask"]] = label
            mapping[str(frame)].append(dict(label=label, raw_detection=item["raw_detection"]))
        Image.fromarray(labels).save(out / f"labels-{frame:05d}.png")
    selected, audit = associate_instances(data, cleaned, static_scene=True, policy=InstancePolicy())
    tracks = {
        name: dict(
            reference_source_frame=item["reference"]["source_frame"],
            observations=[
                dict(
                    frame=o["frame"],
                    source_frame=o["source_frame"],
                    label=o["detection"] + 1,
                    raw_detection=cleaned[o["frame"]][o["detection"]]["raw_detection"],
                    quality=o["quality"],
                )
                for o in item["observations"]
            ],
        )
        for name, item in selected.items()
    }
    write_json(out / "cleanup.json", cleanup)
    write_json(out / "label-map.json", mapping)
    write_json(out / "association.json", audit)
    write_json(out / "tracks.json", tracks)
    return dict(
        duplicate_iou=duplicate_iou,
        raw_counts={str(k): len(v) for k, v in raw.items()},
        clean_counts={str(k): len(v) for k, v in cleaned.items()},
        accepted_tracks=len(selected),
        rejection_counts=dict(Counter(x["reason"] for x in audit["rejected"])),
        inventory_complete=False,
    )


def discover(
    capture,
    out,
    *,
    frames,
    prompts=("object", "small object"),
    duplicate_ious=(0.8, 0.5),
    detector_factory=None,
    checkpoint=None,
):
    """One fresh attempt; persist each frame before advancing; never generate meshes."""
    prompts = tuple(prompts)
    if (
        not prompts
        or len(set(prompts)) != len(prompts)
        or any(not isinstance(p, str) or not p.strip() for p in prompts)
    ):
        raise ValueError("Unique nonempty prompts required")
    if (
        not duplicate_ious
        or len(set(duplicate_ious)) != len(duplicate_ious)
        or any(not np.isfinite(x) or not 0 < x <= 1 for x in duplicate_ious)
    ):
        raise ValueError("Unique duplicate-IoU thresholds in (0,1] required")
    data = normalize(load_capture(capture))
    if (
        len(frames) < 2
        or len(set(frames)) != len(frames)
        or any(type(f) is not int or not 0 <= f < len(data["depth"]) for f in frames)
    ):
        raise ValueError("At least two distinct valid frame indices required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    status = dict(
        schema="generic-discovery-development-v1",
        status="started",
        scope="development",
        capture_sha256=digest(capture),
        frames=frames,
        source_frames=[int(data["source_frame"][f]) for f in frames],
        prompts=prompts,
        detector_threshold=0.3,
        duplicate_ious=list(duplicate_ious),
        image_shape=list(data["rgb"].shape[1:]),
        inventory_complete=False,
        frozen_evaluation_ready=False,
        conditions={},
    )
    write_json(out / "status.json", status)
    started = time.monotonic()
    raw = {p: {} for p in prompts}
    detector = None
    try:
        if detector_factory is None:
            from roboskills.tools.sam3.sam3 import Sam

            detector_factory = Sam
        detector = detector_factory()
        for frame in frames:
            result = detector.detect(data["rgb"][frame], prompts, threshold=0.3)
            Image.fromarray(data["rgb"][frame]).save(out / f"rgb-{frame:05d}.png")
            for index, prompt in enumerate(prompts):
                items = result.get(prompt, [])
                raw[prompt][frame] = items
                prefix = out / f"raw-p{index:02d}-f{frame:05d}"
                np.savez_compressed(
                    prefix.with_suffix(".npz"), **{str(i): d["mask"] for i, d in enumerate(items)}
                )
                write_json(
                    prefix.with_suffix(".json"),
                    dict(
                        prompt=prompt,
                        frame=frame,
                        source_frame=int(data["source_frame"][frame]),
                        detections=[{k: v for k, v in d.items() if k != "mask"} for d in items],
                    ),
                )
            status["last_completed_frame"] = frame
            write_json(out / "status.json", status)
            if checkpoint:
                checkpoint(out, f"detections-frame-{frame}")
        detector.close()
        detector = None
        for index, prompt in enumerate(prompts):
            for threshold in duplicate_ious:
                name = f"p{index:02d}-nms-{threshold:g}"
                result = associate_condition(data, raw[prompt], out / name, threshold)
                status["conditions"][name] = dict(prompt=prompt, **result)
                write_json(out / "status.json", status)
        status["status"] = "diagnostics_completed"
    except BaseException as error:
        status.update(
            status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        if detector is not None:
            try:
                detector.close()
            except Exception as cleanup_error:
                status["cleanup_error"] = str(cleanup_error)
        status["elapsed_s"] = time.monotonic() - started
        write_json(out / "status.json", status)
    if checkpoint:
        checkpoint(out, "discovery-complete")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(discover(args.capture, args.out, frames=args.frames), indent=2))


if __name__ == "__main__":
    main()
