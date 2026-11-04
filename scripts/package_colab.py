"""Build an explicit source/input allowlist for the clean Colab notebooks."""

import argparse
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--iphone-input", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    entries = [(root / "pyproject.toml", Path("pyproject.toml"))]
    for directory in (root / "src/lidar_video", root / "scripts"):
        entries.extend((p, p.relative_to(root)) for p in sorted(directory.glob("*.py")))
    if args.iphone_input:
        entries.extend(
            (args.iphone_input / name, Path("iphone-input") / name)
            for name in ("capture.npz", "board_calibration.json", "input_manifest.json")
        )
    else:
        directory = root / "tests/fixtures/chess-small"
        entries.extend(
            (directory / name, (directory / name).relative_to(root))
            for name in ("capture.npz", "board_calibration.json")
        )
    for source, _ in entries:
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Required regular file missing or symlinked: {source}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "x", zipfile.ZIP_DEFLATED) as archive:
        for source, destination in entries:
            archive.write(source, destination)
    print(f"Created {args.out}: {len(entries)} explicitly selected files")


if __name__ == "__main__":
    main()
