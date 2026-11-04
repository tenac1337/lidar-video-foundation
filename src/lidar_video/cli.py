import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="RGB-D to RoboSkills simulation, scene graph, and Scenic")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    p = sub.add_parser("import-stray")
    p.add_argument("capture", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fps", type=float, default=2.0)
    p = sub.add_parser("board-calibrate")
    p.add_argument("capture", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frame", type=int)
    p.add_argument("--a1-pixel", type=float, nargs=2, metavar=("U", "V"))
    p.add_argument("--square-size", type=float)
    p = sub.add_parser("chess-reconstruct-gpu")
    p.add_argument("capture", type=Path)
    calibration = p.add_mutually_exclusive_group(required=True)
    calibration.add_argument("--calibration", type=Path)
    calibration.add_argument("--auto-calibrate", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--native-script", type=Path, required=True)
    p.add_argument("--reference-frame", type=int, default=0)
    p.add_argument("--mesh-policy", choices=["strict", "fallback"], default="strict")
    p.add_argument("--ambiguous-policy", choices=["error", "skip"], default="error")
    p.add_argument("--workers", type=int, choices=[1, 2], default=1)
    p.add_argument("--samples", type=int, default=20)
    p = sub.add_parser("chess-reconstruct")
    p.add_argument("capture", type=Path)
    calibration = p.add_mutually_exclusive_group(required=True)
    calibration.add_argument("--calibration", type=Path)
    calibration.add_argument("--auto-calibrate", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--masks", type=Path)
    p.add_argument("--mesh-manifest", type=Path)
    p.add_argument("--mesh-policy", choices=["strict", "fallback"], default="strict")
    p.add_argument("--model", type=Path)
    p.add_argument("--context", type=Path)
    p.add_argument("--static-scene", action="store_true")
    p.add_argument("--fusion-frames", type=Path)
    p.add_argument("--min-height", type=float, default=0.006)
    p.add_argument("--reference-frame", type=int, default=0)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--no-simulation", action="store_true")
    for name in ("demo", "synthetic", "chess-benchmark", "robot-demo"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        if name in ("demo", "synthetic"):
            p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("chess-demo")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("learned-baseline")
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("chess-recognize")
    p.add_argument("capture", type=Path)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--context", type=Path)
    p.add_argument("--out", type=Path, required=True)
    for name in ("validate", "process", "import-r3d", "upstream-smoke"):
        p = sub.add_parser(name)
        p.add_argument("capture", type=Path)
        if name != "validate":
            p.add_argument("--out", type=Path, required=True)
        if name == "process":
            p.add_argument("--samples", type=int, default=20)
            p.add_argument("--no-simulation", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "import-stray":
            from lidar_video.stray import import_stray

            result = import_stray(args.capture, args.out, fps=args.fps)
        elif args.command == "board-calibrate":
            from lidar_video.board_calibration import calibrate
            from lidar_video.capture import load_capture

            result = calibrate(
                load_capture(args.capture),
                args.out,
                frame=args.frame,
                a1_pixel=args.a1_pixel,
                square_size=args.square_size,
            )
        elif args.command == "chess-reconstruct-gpu":
            from lidar_video.gpu_reconstruction import run_gpu

            result = run_gpu(
                args.capture,
                args.calibration,
                args.out,
                args.native_script,
                args.samples,
                mesh_policy=args.mesh_policy,
                reference_frame=args.reference_frame,
                ambiguous_policy=args.ambiguous_policy,
                workers=args.workers,
            )
        elif args.command == "chess-reconstruct":
            from lidar_video.reconstruction import reconstruct

            result = reconstruct(
                args.capture,
                args.calibration,
                args.out,
                masks=args.masks,
                fusion_frames=json.loads(args.fusion_frames.read_text()) if args.fusion_frames else None,
                min_height=args.min_height,
                mesh_manifest=args.mesh_manifest,
                mesh_policy=args.mesh_policy,
                model=args.model,
                context=args.context,
                static_scene=args.static_scene,
                reference_frame=args.reference_frame,
                samples=args.samples,
                simulation=not args.no_simulation,
            )
        elif args.command == "doctor":
            from lidar_video.pipeline import doctor

            result = doctor()
        elif args.command == "chess-benchmark":
            from lidar_video.chess_recognition import benchmark

            result = benchmark(args.out)
        elif args.command == "chess-demo":
            from lidar_video.chess_scene import chess_demo

            result = chess_demo(args.model, args.out)
        elif args.command == "robot-demo":
            from lidar_video.manipulation import run_manipulation

            result = run_manipulation(args.out)
        elif args.command == "learned-baseline":
            from lidar_video.baseline import run_baseline

            result = run_baseline(args.repo, args.out)
        elif args.command == "chess-recognize":
            from lidar_video.capture import load_capture, write_json
            from lidar_video.chess_recognition import recognize
            from lidar_video.chess_scene import board_from_observation, sample_legal_scenarios, scene_graph

            args.out.mkdir(parents=True, exist_ok=False)
            result = recognize(
                load_capture(args.capture), json.loads(args.calibration.read_text()), args.model
            )
            write_json(args.out / "recognition.json", result)
            if args.context is not None:
                board = board_from_observation(result, json.loads(args.context.read_text()))
                graph = scene_graph(board, evidence=result["evidence"])
                write_json(args.out / "scene_graph.json", graph)
                result["scenarios"] = sample_legal_scenarios(graph, args.out / "scenarios")
            else:
                result["game_state"] = (
                    "placement only; supply --context to validate legality and sample moves"
                )
        elif args.command == "demo":
            from lidar_video.pipeline import demo

            result = demo(args.out, args.seed)
        elif args.command == "synthetic":
            from lidar_video.synthetic import generate

            result = {"capture": str(generate(args.out, seed=args.seed))}
        elif args.command == "validate":
            from lidar_video.capture import load_capture, validate

            result = validate(load_capture(args.capture))
        elif args.command == "import-r3d":
            from lidar_video.capture import import_r3d

            result = import_r3d(args.capture, args.out)
        elif args.command == "upstream-smoke":
            from lidar_video.upstream import upstream_smoke

            result = upstream_smoke(args.capture, args.out)
        else:
            from lidar_video.pipeline import process

            result = process(
                args.capture, args.out, samples=args.samples, simulate_output=not args.no_simulation
            )
        print(json.dumps(result, indent=2))
        if args.command == "robot-demo" and not result["passed"]:
            parser.exit(3, "Robot verification failed; see the saved verification.json.\n")
        if args.command == "chess-demo" and result["status"] != "completed":
            parser.exit(3, "Chess recognition needs review; no legal scenario was generated.\n")
    except (ValueError, FileExistsError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
