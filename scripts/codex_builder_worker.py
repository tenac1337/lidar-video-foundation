"""Optional repair worker: new candidate files only, fixed measured evidence.

Uses a fresh Codex session with workspace-write limited to a new repair folder.
No permission bypasses. The audit loop independently checks the proposed case.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def command(packet_path, report_path, binary="codex"):
    packet_path, report_path = Path(packet_path).resolve(), Path(report_path).resolve()
    if report_path.exists():
        raise ValueError("Refusing to overwrite a builder proposal")
    packet = json.loads(packet_path.read_text())
    case = json.loads(Path(packet["case"]).read_text())
    workspace = report_path.parent / "repair-workspace"
    workspace.mkdir(exist_ok=False)
    props = {
        key: {"enum": [value]} for key, value in case.items() if key not in ("mesh_manifest", "scene_dir")
    }
    for key in ("mesh_manifest", "scene_dir"):
        props[key] = {"type": ["string", "null"]}
    schema = report_path.with_suffix(".schema.json")
    schema.write_text(
        json.dumps(
            {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
        )
    )
    prompt = (
        f"Read repair request {packet_path}, its case and decision. Repair justified failures using NEW "
        f"candidate files only in {workspace}. Frozen source evidence, masks, reference frames, expected "
        "inventory and numerical gates must remain unchanged. Do not read or use any final test split. "
        "Do not change source code. No network, uploads, GPU launches or extra packages. Reuse existing "
        "local tools where possible. Preserve validation metadata and resolve all manifest paths "
        "absolutely. If no evidence-backed repair is possible, return the unchanged case. "
        "Your final response is the next case JSON using absolute paths; the CLI saves it. "
        "Missing measurements must not be replaced by invented observations. "
        f"Available Python: {sys.executable}; project source: {Path(__file__).resolve().parents[1] / 'src'}."
    )
    return [
        binary,
        "exec",
        "--sandbox",
        "workspace-write",
        "--ephemeral",
        "--skip-git-repo-check",
        "--cd",
        str(workspace),
        "--output-schema",
        str(schema),
        "--output-last-message",
        str(report_path),
        "--json",
        prompt,
    ]


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: codex_builder_worker.py REPAIR_PACKET NEXT_CASE_JSON")
    result = subprocess.run(
        command(*sys.argv[1:], binary=os.environ.get("CODEX_BIN", "codex")), stdin=subprocess.DEVNULL
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
