"""Replay reliability gates against saved SAM detections without model inference."""

import argparse
import json
from pathlib import Path

import numpy as np

from lidar_video.capture import load_capture, write_json
from lidar_video.gpu_reconstruction import assign_instances
from lidar_video.object_evidence import assess_depth


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("learned", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = load_capture(args.learned / "capture.npz")
    cal = json.loads((args.learned / "calibration.json").read_text())
    frame = int(cal.get("reference_frame", 0))
    raw = json.loads((args.learned / "sam3_raw_detections.json").read_text())
    with np.load(args.learned / "sam3_raw_masks.npz", allow_pickle=False) as masks:
        raw = [dict(d, mask=masks[str(i)]) for i, d in enumerate(raw)]
    baseline = json.loads((args.learned / "masks/objects.json").read_text())
    results = []
    for obj in baseline:
        quality, _, _ = assess_depth(data, cal, frame, raw[obj["raw_index"]]["mask"])
        results.append(dict(square=obj["square"], raw_index=obj["raw_index"], quality=quality))
    _, candidates, audit = assign_instances(data, cal, frame, raw, "skip")
    report = dict(
        status="reliability gates replayed; no neural inference or mesh acceptance",
        previous_inference_jobs=len(baseline),
        blocked_previous_jobs=[r["square"] for r in results if not r["quality"]["passed"]],
        remaining_candidates=[r["square"] for r in candidates],
        per_object=results,
        segmentation_audit=audit,
        next_required="Segment additional views and corroborate instances before generating replacement meshes",
    )
    write_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
