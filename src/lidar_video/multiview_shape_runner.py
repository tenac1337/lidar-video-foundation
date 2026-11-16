"""Run a frozen multi-view shape proposal from a portable JSON manifest.

Usage: python -m lidar_video.multiview_shape_runner manifest.json output_directory
Mesh vertices, camera poses and support_from_world must share the same metric
world frame. The output is a visual proposal, not a physics-ready scene asset.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from lidar_video.multiview_shape import VolumePolicy, refine_support_pose, select_candidate, silhouette_volume


def run(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    config = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    views = [dict(v, labels=np.asarray(Image.open(root / v["labels"]))) for v in config["views"]]
    construction = config["construction_sources"]
    validation = config["validation_sources"]
    ids = [v["source_frame"] for v in views]
    split = construction + validation
    if len(split) != len(set(split)) or set(split) != set(ids):
        raise ValueError("Construction and validation must partition unique observations")
    output.mkdir(parents=True, exist_ok=False)
    # Save the declared policy before constructing or scoring any candidate.
    (output / "manifest.json").write_text(json.dumps(config, indent=2) + "\n")
    try:
        incumbent = trimesh.load(root / config["incumbent_mesh"], force="mesh", process=False)
        training = [v for v in views if v["source_frame"] in construction]
        method = config.get("proposal_method", "silhouette_volume")
        if method == "support_pose":
            if config.get("static_scene") is not True:
                raise ValueError("Explicit static_scene=True required")
            mesh, evidence = refine_support_pose(
                incumbent, training, config["support_from_world"], **config.get("policy", {})
            )
        elif method == "silhouette_volume":
            mesh, evidence = silhouette_volume(
                training,
                config["support_from_world"],
                policy=VolumePolicy(**config.get("policy", {})),
                static_scene=config.get("static_scene", False),
            )
        else:
            raise ValueError("Unknown proposal_method")
        mesh.export(output / "proposal-world.ply")
        result = select_candidate(
            {method: mesh},
            incumbent,
            views,
            construction,
            validation,
            **config.get("selection_policy", {}),
        )
        result["proposal_evidence"] = evidence
    except ValueError as exc:
        result = dict(accepted=False, reason=str(exc), qualitative_win_established=False)
    result["scope"] = "Silhouette screening only; no appearance, collision, dynamics or completeness claim"
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
