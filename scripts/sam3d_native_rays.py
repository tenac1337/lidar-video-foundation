"""Adapter for the pinned RoboSkills worker: retain exact sensor rays in point maps.

Only its points(depth, K) helper is replaced, for both model conditioning and pose
scoring. Segmentation, SAM 3D inference, scale conversion and pose scoring stay in
upstream code. Legacy MoGe depth files without sample metadata keep upstream rays.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

# The isolated SAM 3D environment needs only NumPy from our native-depth module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lidar_video.native_depth import sensor_camera_points  # noqa: E402


def run(worker_path, args):
    spec = importlib.util.spec_from_file_location("roboskills_sam3d_worker", worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    # Upstream arguments: video, masks, depth, out, frame, label, samples, ...
    with np.load(args[2], allow_pickle=False) as archive:
        if "depth_sample_id" in archive or "native_K" in archive:
            frames = archive["frames"]
            rows = np.flatnonzero(frames == int(args[4]))
            if len(rows) != 1:
                raise ValueError("SAM 3D frame must uniquely match native depth metadata")
            row = int(rows[0])
            data = {key: archive[key] for key in archive.files}
            camera = sensor_camera_points(data, row)
            if not np.isfinite(camera[data["valid"][row]]).all():
                raise ValueError("Valid SAM 3D depth has missing native rays")

            def points(depth, k):
                if depth.shape != camera.shape[:2] or not np.array_equal(
                    depth, data["depth"][row].astype(np.float32), equal_nan=True
                ):
                    raise ValueError("SAM 3D depth differs from its native point map")
                return camera.copy()

            worker.points = points
    worker.run(*args)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:])
