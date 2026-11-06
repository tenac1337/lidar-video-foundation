"""Compare fixed export check policies and summarize independently labeled cases.

These modes do not reconstruct, fit or repair scenes. Scenic replay executes trusted
project code; neither an evidence seal nor a subprocess is a security sandbox.
"""

import time
from pathlib import Path

from lidar_video.audit_checks import _exports, validate_graph
from lidar_video.creation_audit import digest, read_json, save

MODES = ("export", "geometry", "consistency")


def _inventory(scene):
    scene = Path(scene).resolve()
    if not scene.is_dir():
        raise ValueError("Scene directory does not exist")
    files = {}
    for path in sorted(scene.rglob("*")):
        if path.is_file():
            if not path.resolve().is_relative_to(scene):
                raise ValueError("Scene file escapes the scene directory")
            files[str(path.relative_to(scene))] = digest(path)
    return files


def seal_exports(scene, out):
    """Record a prior file inventory outside the scene; a seal is not a quality verdict."""
    scene, out = Path(scene).resolve(), Path(out).resolve()
    if out.is_relative_to(scene):
        raise ValueError("Evidence seal must be outside the scene directory")
    record = dict(schema="export-evidence-seal-v1", files=_inventory(scene))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        import json

        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return {"status": "sealed", "path": str(out), "sha256": digest(out)}


def _read_export(scene):
    graph = read_json(scene / "scene_graph.json")
    samples = read_json(scene / "scenic_samples.json")
    if not isinstance(graph, dict) or not isinstance(samples, dict):
        raise ValueError("Export records must be JSON objects")
    if (not isinstance(graph.get("nodes"), list) or not graph["nodes"]
            or any(not isinstance(n, dict) for n in graph["nodes"])):
        raise ValueError("Export graph needs nodes")
    if not isinstance(samples.get("samples"), list) or not samples["samples"]:
        raise ValueError("Export requires a nonempty sample list")
    if not (scene / "scenario.scenic").read_text().strip():
        raise ValueError("Scenic program is empty")
    return graph, samples


def compare_exports(manifest_file, out):
    """Run all policies on the same snapshot, without giving checkers validity labels."""
    import subprocess

    manifest_file, out = Path(manifest_file).resolve(), Path(out).resolve()
    manifest_digest = digest(manifest_file)
    manifest = read_json(manifest_file)
    if manifest.get("schema") != "export-comparison-v1":
        raise ValueError("Expected export-comparison-v1")
    expected = manifest.get("expected_objects")
    if (
        not isinstance(expected, list) or not expected
        or any(not isinstance(x, str) or not x.strip() for x in expected)
        or len(set(expected)) != len(expected)
    ):
        raise ValueError("expected_objects must contain unique nonempty IDs")
    scene = (manifest_file.parent / manifest["scene_dir"]).resolve()
    if out.is_relative_to(scene):
        raise ValueError("Comparison outputs must be outside the input scene")
    before = _inventory(scene)
    seal = manifest.get("evidence_seal")
    seal_path = (manifest_file.parent / seal).resolve() if seal else None
    seal_hash = digest(seal_path) if seal_path and seal_path.is_file() else None
    out.mkdir(parents=True, exist_ok=False)
    decisions = []
    for mode in MODES:
        started = time.perf_counter()
        folder = out / mode
        folder.mkdir()
        checks = []
        try:
            graph, samples = _read_export(scene)
            checks = ["readable_export"]
            if mode != "export":
                validate_graph(graph, expected)
                for sample in samples["samples"]:
                    if not isinstance(sample, dict):
                        raise ValueError("Sample graph must be an object")
                    validate_graph(sample, expected)
                checks.append("graph_and_sample_geometry")
            if mode == "consistency":
                if seal_path is None or not seal_path.is_file():
                    raise ImportError("Required evidence seal is unavailable")
                seal_record = read_json(seal_path)
                if seal_record.get("schema") != "export-evidence-seal-v1":
                    raise ValueError("Unsupported evidence seal")
                if seal_record.get("files") != before:
                    raise ValueError("Export files differ from their prior evidence seal")
                checks.append("evidence_integrity")
                detail = _exports(graph, scene, expected, folder)
                save(folder / "exports.json", detail)
                checks.append("metadata_assets_and_program_replay")
            decision, reason = "accept", "Selected checks passed"
        except (ImportError, subprocess.TimeoutExpired) as error:
            decision, reason = "blocked", str(error)
        except (ValueError, OSError, KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
            decision, reason = "reject", str(error)
        decisions.append(dict(
            mode=mode, decision=decision, reason=reason,
            completed_checks=checks,
            elapsed_seconds=time.perf_counter() - started,
        ))
        # Enforce a common input snapshot even when a weaker policy did not inspect all files.
        if _inventory(scene) != before or digest(manifest_file) != manifest_digest:
            raise ValueError("Comparison input changed during execution; discard partial decisions")
        current_seal = digest(seal_path) if seal_path and seal_path.is_file() else None
        if current_seal != seal_hash:
            raise ValueError("Evidence seal changed during comparison")
    result = dict(
        schema="export-comparison-result-v1", decisions=decisions,
        input_files=before, manifest_sha256=manifest_digest, evidence_seal_sha256=seal_hash,
        mode_scope={
            "export": "Readable nonempty graph, samples and program; no semantic validation",
            "geometry": "Export checks plus graph/sample inventory, support and envelope constraints",
            "consistency": "Geometry plus prior evidence seal, metadata/assets and actual Scenic replay",
        },
        limits=[
            "Fixed export policies only: no reconstruction, repair, physical accuracy or dynamics comparison.",
            "Modes have different checking costs; record elapsed time rather than claim matched compute.",
            "An evidence seal records file integrity, not scene truth or absence of prior inspection.",
        ],
    )
    save(out / "comparison.json", result)
    return result


def summarize_decisions(cases):
    """Aggregate externally supplied labels; never infer correctness from a gate verdict."""
    if not isinstance(cases, list) or not cases:
        raise ValueError("Nonempty labeled cases are required")
    seen = set()
    for case in cases:
        if not all(isinstance(case.get(k), str) and case[k].strip()
                   for k in ("case_id", "scene_id", "label_reason")):
            raise ValueError("Each case requires case_id, scene_id and independent label_reason")
        if case["case_id"] in seen:
            raise ValueError("Duplicate case_id")
        seen.add(case["case_id"])
        if case.get("label") not in ("valid", "invalid", "unknown"):
            raise ValueError("Label must be valid, invalid or unknown")
        records = case.get("decisions", [])
        if len(records) != len(MODES) or {r.get("mode") for r in records} != set(MODES):
            raise ValueError("Every case needs exactly one decision for every mode")
        if any(r.get("decision") not in ("accept", "reject", "blocked") for r in records):
            raise ValueError("Decision must be accept, reject or blocked")

    def group(rows):
        summaries = {}
        for mode in MODES:
            counts = {label: dict.fromkeys(("accept", "reject", "blocked"), 0)
                      for label in ("valid", "invalid", "unknown")}
            for case in rows:
                row = next(r for r in case["decisions"] if r["mode"] == mode)
                counts[case["label"]][row["decision"]] += 1
            valid, invalid = sum(counts["valid"].values()), sum(counts["invalid"].values())
            known_accepted = counts["valid"]["accept"] + counts["invalid"]["accept"]
            blocked = sum(c["blocked"] for c in counts.values())
            summaries[mode] = dict(
                counts=counts, total_cases=len(rows), known_valid=valid, known_invalid=invalid,
                invalid_acceptance_rate=counts["invalid"]["accept"] / invalid if invalid else None,
                invalid_rejection_rate=counts["invalid"]["reject"] / invalid if invalid else None,
                valid_retention_rate=counts["valid"]["accept"] / valid if valid else None,
                invalid_fraction_among_known_accepted=(
                    counts["invalid"]["accept"] / known_accepted if known_accepted else None
                ),
                decision_coverage=(len(rows) - blocked) / len(rows),
                accepted_unknown=counts["unknown"]["accept"],
            )
        return summaries

    scene_ids = sorted({case["scene_id"] for case in cases})
    return dict(
        schema="export-comparison-summary-v1", unit="labeled_artifact_case",
        case_count=len(cases), scene_count=len(scene_ids), modes=group(cases),
        per_scene={scene: group([c for c in cases if c["scene_id"] == scene]) for scene in scene_ids},
        limits=[
            "Corruptions of the same scene are dependent; case_count is not an independent-scene count.",
            "Blocked decisions are not successful rejections; inspect coverage and all three counts.",
            "Unknown labels are excluded from validity rates but retained in coverage and counts.",
        ],
    )


def summarize_comparisons(manifest_file, out):
    """Join external labels to recorded comparisons; retain hashes for the joined inputs."""
    manifest_file, out = Path(manifest_file).resolve(), Path(out).resolve()
    manifest_hash = digest(manifest_file)
    manifest = read_json(manifest_file)
    if manifest.get("schema") != "labeled-export-comparisons-v1":
        raise ValueError("Expected labeled-export-comparisons-v1")
    hashes = {str(manifest_file): manifest_hash}
    cases = []
    for case in manifest.get("cases", []):
        path = (manifest_file.parent / case["result"]).resolve()
        if str(path) in hashes:
            raise ValueError("A comparison result cannot be counted more than once")
        hashes[str(path)] = digest(path)
        record = read_json(path)
        if record.get("schema") != "export-comparison-result-v1":
            raise ValueError("Unsupported comparison result")
        cases.append(dict(case, decisions=record["decisions"]))
    result = summarize_decisions(cases)
    result["input_hashes"] = hashes
    if any(digest(path) != value for path, value in hashes.items()):
        raise ValueError("Comparison or label inputs changed during aggregation")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        import json

        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
