import copy
import json
from pathlib import Path

import pytest

from lidar_video import comparison
from lidar_video.comparison import compare_exports, summarize_comparisons, summarize_decisions
from lidar_video.comparison_fixtures import generate_comparison_fixtures
from lidar_video.creation_audit import read_json


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("comparison") / "fixtures"
    fixtures = generate_comparison_fixtures(out)
    results = {}
    for case in fixtures["cases"]:
        path = Path(case["manifest"])
        results[case["case_id"]] = compare_exports(path, path.parent / "comparison")
    return fixtures, results


def test_modes_distinguish_geometric_and_program_contract_faults(generated):
    _, results = generated
    decisions = {
        name: {r["mode"]: r["decision"] for r in result["decisions"]}
        for name, result in results.items()
    }
    assert decisions["clean"] == dict.fromkeys(comparison.MODES, "accept")
    for name in ("overlap", "missing_object"):
        assert decisions[name] == dict(export="accept", geometry="reject", consistency="reject")
    for name in ("sample_outside_program", "sample_metadata", "stale_seal"):
        assert decisions[name] == dict(export="accept", geometry="accept", consistency="reject")
    program = results["sample_outside_program"]["decisions"][-1]
    assert "program replay" in program["reason"]


def test_fixture_labels_join_without_treating_corruptions_as_independent_scenes(generated, tmp_path):
    fixtures, _ = generated
    report = summarize_comparisons(fixtures["labels"], tmp_path / "summary.json")
    assert report["case_count"] == 6
    assert report["scene_count"] == 1
    assert report["modes"]["export"]["invalid_acceptance_rate"] == 1
    assert report["modes"]["consistency"]["valid_retention_rate"] == 1
    assert report["modes"]["consistency"]["counts"]["invalid"]["reject"] == 5
    with pytest.raises(FileExistsError):
        summarize_comparisons(fixtures["labels"], tmp_path / "summary.json")


def test_blocked_and_unknown_cases_have_explicit_denominators():
    def case(name, label, decision):
        return dict(case_id=name, scene_id="one", label=label, label_reason="External reference",
                    decisions=[dict(mode=mode, decision=decision) for mode in comparison.MODES])
    rows = [case("bad", "invalid", "blocked"), case("good", "valid", "accept"),
            case("unknown", "unknown", "accept")]
    result = summarize_decisions(rows)["modes"]["consistency"]
    assert result["invalid_acceptance_rate"] == 0
    assert result["invalid_rejection_rate"] == 0
    assert result["decision_coverage"] == pytest.approx(2 / 3)
    assert result["counts"]["invalid"]["blocked"] == 1
    assert result["accepted_unknown"] == 1
    assert result["invalid_fraction_among_known_accepted"] == 0
    only_unknown = summarize_decisions(rows[-1:])["modes"]["export"]
    assert only_unknown["invalid_acceptance_rate"] is None
    assert only_unknown["valid_retention_rate"] is None
    assert only_unknown["invalid_fraction_among_known_accepted"] is None


def test_unpaired_modes_and_duplicate_cases_are_rejected():
    row = dict(case_id="a", scene_id="scene", label="valid", label_reason="External reference",
               decisions=[dict(mode=mode, decision="accept") for mode in comparison.MODES])
    with pytest.raises(ValueError, match="Duplicate case"):
        summarize_decisions([row, row])
    broken = copy.deepcopy(row)
    broken["decisions"][1]["mode"] = "export"
    with pytest.raises(ValueError, match="exactly one"):
        summarize_decisions([broken])
    with pytest.raises(ValueError, match="Nonempty"):
        summarize_decisions([])


def test_missing_dependency_is_blocked_not_successful_rejection(generated, tmp_path, monkeypatch):
    fixtures, _ = generated
    def unavailable(*args):
        raise ImportError("Optional runtime unavailable")
    monkeypatch.setattr(comparison, "_exports", unavailable)
    result = compare_exports(fixtures["cases"][0]["manifest"], tmp_path / "result")
    assert result["decisions"][-1]["decision"] == "blocked"
    assert all(r["decision"] == "accept" for r in result["decisions"][:2])


def test_missing_evidence_seal_is_blocked(generated, tmp_path):
    fixtures, _ = generated
    original = Path(fixtures["cases"][0]["manifest"])
    manifest = read_json(original)
    manifest.pop("evidence_seal")
    manifest["scene_dir"] = str(original.parent / "scene")
    path = tmp_path / "case.json"
    path.write_text(json.dumps(manifest))
    result = compare_exports(path, tmp_path / "out")
    assert result["decisions"][-1]["decision"] == "blocked"


def test_duplicate_result_files_cannot_inflate_case_counts(generated, tmp_path):
    fixtures, _ = generated
    manifest = read_json(fixtures["labels"])
    extra = dict(manifest["cases"][0], case_id="duplicate-under-another-name")
    manifest["cases"].append(extra)
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="counted more than once"):
        summarize_comparisons(path, tmp_path / "summary.json")


def test_input_mutation_during_mode_execution_invalidates_comparison(tmp_path, monkeypatch):
    fixtures = generate_comparison_fixtures(tmp_path / "fixtures")
    manifest = Path(fixtures["cases"][0]["manifest"])
    def mutate(graph, scene, expected, out):
        path = scene / "scene_graph.json"
        path.write_text(path.read_text() + " ")
        return dict(status="pass")
    monkeypatch.setattr(comparison, "_exports", mutate)
    out = tmp_path / "compared"
    with pytest.raises(ValueError, match="input changed"):
        compare_exports(manifest, out)
    assert not (out / "comparison.json").exists()


@pytest.mark.parametrize("field", ["case_id", "scene_id"])
def test_result_label_identity_mismatch_is_rejected(generated, tmp_path, field):
    fixtures, _ = generated
    manifest = read_json(fixtures["labels"])
    manifest["cases"][0][field] = "wrong-identity"
    path = tmp_path / "mislabeled.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match"):
        summarize_comparisons(path, tmp_path / "summary.json")


def test_comparison_manifest_requires_identity(generated, tmp_path):
    fixtures, _ = generated
    original = Path(fixtures["cases"][0]["manifest"])
    manifest = read_json(original)
    manifest.pop("case_id")
    path = tmp_path / "missing-identity.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="requires case_id"):
        compare_exports(path, tmp_path / "out")


@pytest.mark.parametrize("asset_kind", ["visual", "collision"])
@pytest.mark.parametrize("collapsed", [False, True])
def test_resealed_mesh_assets_require_nonzero_surface_area(generated, tmp_path, asset_kind, collapsed):
    import shutil

    import trimesh

    fixtures, _ = generated
    original = Path(fixtures["cases"][0]["manifest"]).parent
    scene = tmp_path / "scene"
    shutil.copytree(original / "scene", scene)
    graph = read_json(scene / "scene_graph.json")
    samples = read_json(scene / "scenic_samples.json")
    metadata = dict(mesh_path="visual.ply", collision_paths=["collision.ply"])
    for record in [graph, *samples["samples"]]:
        next(n for n in record["nodes"] if n["id"] == "left").update(metadata)
    for kind in ("visual", "collision"):
        mesh = trimesh.creation.box(extents=[0.02, 0.02, 0.03])
        if collapsed and kind == asset_kind:
            mesh.vertices[:] = 0
        mesh.export(scene / f"{kind}.ply")
    (scene / "scene_graph.json").write_text(json.dumps(graph))
    (scene / "scenic_samples.json").write_text(json.dumps(samples))
    # A current seal proves byte integrity, not that the surface is usable.
    comparison.seal_exports(scene, tmp_path / "seal.json")
    manifest = read_json(original / "case.json")
    path = tmp_path / "case.json"
    path.write_text(json.dumps(manifest))
    result = compare_exports(path, tmp_path / "comparison")
    assert all(row["decision"] == "accept" for row in result["decisions"][:2])
    consistency = result["decisions"][-1]
    assert consistency["decision"] == ("reject" if collapsed else "accept")
    if collapsed:
        assert "nonzero-area" in consistency["reason"]
