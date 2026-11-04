"""Reload portable assets, decode videos, replay physics, and check Scenic samples."""

import argparse
import json
from pathlib import Path

import cv2
import mujoco

from lidar_video.capture import write_json
from lidar_video.scenic_export import validate_sample
from lidar_video.simulation import replay_check


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    graph = json.loads((args.result / "scene_graph.json").read_text())
    samples = json.loads((args.result / "scenic_samples.json").read_text())["samples"]
    for sample in samples:
        validate_sample(sample)
    objects = [n for n in graph["nodes"] if n["kind"] == "object"]
    for node in objects:
        for relative in [node["mesh_path"], *node["collision_paths"]]:
            path = (args.result / relative).resolve()
            assert path.is_relative_to(args.result.resolve()) and path.is_file()
    report = dict(objects=len(objects), scenic_samples_validated=len(samples))
    for name in ("simulation", "sample_simulation"):
        folder = args.result / name
        model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
        video = cv2.VideoCapture(str(folder / "rollout/mujoco_ext1.mp4"))
        frames = 0
        while video.read()[0]:
            frames += 1
        video.release()
        assert frames == 30
        replay = replay_check(folder / "rollout")
        assert replay["checked_intervals"] == 29
        assert max(replay["max_qpos_error"], replay["max_qvel_error"]) < 1e-7
        report[name] = dict(bodies=model.nbody, decoded_video_frames=frames, replay=replay)
    write_json(args.result / "independent_checks.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
