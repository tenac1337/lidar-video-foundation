"""Reports cannot turn absent or rejected machine evidence into acceptance."""

import sys

import pytest

from lidar_video import creation_audit as audit


@pytest.fixture
def case(tmp_path, monkeypatch):
    from lidar_video import audit_checks

    monkeypatch.setattr(
        audit_checks, "run_checks", lambda *a: {r: {"status": "pass", "findings": []} for r in audit.ROLES}
    )
    capture = tmp_path / "evidence.npz"
    capture.write_bytes(b"fixture")
    p = tmp_path / "case.json"
    audit.save(
        p,
        dict(
            schema="creation-audit-case-v1",
            evaluation_scope="development",
            expected_objects=["a1"],
            capture=str(capture),
        ),
    )
    return p


def reviews(folder, status="pass"):
    for role in audit.ROLES:
        p = folder / "packets" / f"{role}.json"
        packet = audit.read_json(p)
        audit.save(
            folder / "reviews" / f"{role}.json",
            dict(
                schema="creation-audit-review-v1",
                round_id=packet["round_id"],
                role=role,
                packet_sha256=audit.digest(p),
                status=status,
                summary="Fixture review",
                findings=[],
                evidence=["checks.json"],
            ),
        )


def test_missing_stale_and_valid_reviews(case, tmp_path):
    run = tmp_path / "run"
    assert audit.prepare(case, run)["status"] == "awaiting_reviews"
    reviews(run)
    assert audit.finalize(run)["status"] == "accepted"
    p = run / "reviews/geometry.json"
    r = audit.read_json(p)
    r["round_id"] = "previous-round"
    audit.save(p, r)
    assert audit.finalize(run)["status"] == "invalid"


@pytest.mark.parametrize("machine_status", ["fail", "blocked"])
def test_agent_pass_never_overrides_machine_rejection(case, tmp_path, monkeypatch, machine_status):
    from lidar_video import audit_checks

    monkeypatch.setattr(
        audit_checks,
        "run_checks",
        lambda *a: {
            r: {"status": machine_status if r == "geometry" else "pass", "findings": []} for r in audit.ROLES
        },
    )
    run = tmp_path / "run"
    audit.prepare(case, run)
    reviews(run)
    assert audit.finalize(run)["status"] == "rejected"


def test_changed_capture_or_forged_evidence_invalidates(case, tmp_path):
    run = tmp_path / "run"
    audit.prepare(case, run)
    reviews(run)
    p = run / "reviews/geometry.json"
    r = audit.read_json(p)
    r["evidence"] = ["../../not-an-artifact"]
    audit.save(p, r)
    assert audit.finalize(run)["status"] == "invalid"
    reviews(run)
    (tmp_path / "evidence.npz").write_bytes(b"changed")
    assert audit.finalize(run)["status"] == "invalid"


def test_observation_masks_are_frozen(case, tmp_path):
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"mask")
    mesh = tmp_path / "mesh.ply"
    mesh.write_bytes(b"mesh")
    pose = tmp_path / "pose.json"
    pose.write_text("{}")
    manifest = tmp_path / "meshes.json"
    audit.save(
        manifest,
        {
            "objects": {
                "a1": {"mesh": "mesh.ply", "pose": "pose.json", "observations": [{"mask": "mask.png"}]}
            }
        },
    )
    c = audit.read_json(case)
    c["mesh_manifest"] = str(manifest)
    audit.save(case, c)
    run = tmp_path / "run"
    audit.prepare(case, run)
    reviews(run)
    mask.write_bytes(b"changed")
    assert audit.finalize(run)["status"] == "invalid"


def test_final_evaluation_cannot_be_repaired(case, tmp_path):
    c = audit.read_json(case)
    c["evaluation_scope"] = "final"
    audit.save(case, c)
    with pytest.raises(ValueError, match="development-only"):
        audit.prepare(case, tmp_path / "run")


def worker_file(tmp_path, body):
    p = tmp_path / "worker.py"
    p.write_text(body)
    return [sys.executable, str(p)]


def test_worker_success_requires_report_and_timeout_is_failure(tmp_path):
    cmd = worker_file(tmp_path, "pass")
    with pytest.raises(ValueError, match="without writing"):
        audit.invoke(cmd, "packet", tmp_path / "absent.json", tmp_path / "empty.log", 5)
    cmd = worker_file(tmp_path, "import time; time.sleep(30)")
    with pytest.raises(ValueError, match="timed out"):
        audit.invoke(cmd, "packet", tmp_path / "absent.json", tmp_path / "timeout.log", 0.1)


def test_loop_runs_all_workers_and_stops_on_acceptance(case, tmp_path):
    worker = worker_file(
        tmp_path,
        """import sys,json,hashlib
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads(p.read_text());v=r['report_contract']
v.update(status='pass',summary='Executed fixture worker',findings=[],evidence=['checks.json'],packet_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
Path(sys.argv[2]).write_text(json.dumps(v))
""",
    )
    result = audit.run_loop(case, tmp_path / "loop", worker)
    assert result["status"] == "accepted"
    assert len(result["rounds"]) == 1
    assert len(list((tmp_path / "loop/round-01/reviews").glob("*.json"))) == 4


def test_loop_records_failure_instead_of_accepting_partial_reports(case, tmp_path):
    cmd = worker_file(tmp_path, "raise SystemExit(7)")
    result = audit.run_loop(case, tmp_path / "loop", cmd)
    assert result["status"] == "failed"
    assert "exit 7" in result["error"]


def test_empty_expected_objects_rejected(case, tmp_path):
    c = audit.read_json(case)
    c["expected_objects"] = []
    audit.save(case, c)
    with pytest.raises(ValueError, match="nonempty"):
        audit.prepare(case, tmp_path / "run")


def test_builder_cannot_change_expected_inventory(case, tmp_path, monkeypatch):
    from lidar_video import audit_checks

    monkeypatch.setattr(
        audit_checks, "run_checks", lambda *a: {r: {"status": "fail", "findings": []} for r in audit.ROLES}
    )
    worker = worker_file(
        tmp_path,
        """import sys,json,hashlib
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads(p.read_text());v=r['report_contract']
v.update(status='fail',summary='Fixture failure',findings=[],evidence=['checks.json'],packet_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
Path(sys.argv[2]).write_text(json.dumps(v))
""",
    )
    b = tmp_path / "builder.py"
    b.write_text("""import sys,json
from pathlib import Path
request=json.loads(Path(sys.argv[1]).read_text());case=json.loads(Path(request['case']).read_text())
case['expected_objects']=['b2']
Path(sys.argv[2]).write_text(json.dumps(case))
""")
    result = audit.run_loop(case, tmp_path / "loop", worker, builder=[sys.executable, str(b)])
    assert result["status"] == "failed"
    assert "fixed evidence" in result["error"]


def test_unchanged_repair_stops_without_repeating_inference(case, tmp_path, monkeypatch):
    from lidar_video import audit_checks

    monkeypatch.setattr(
        audit_checks, "run_checks", lambda *a: {r: {"status": "blocked", "findings": []} for r in audit.ROLES}
    )
    worker = worker_file(
        tmp_path,
        """import sys,json,hashlib
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads(p.read_text());v=r['report_contract']
v.update(status='blocked',summary='Fixture blocked',findings=[],evidence=['checks.json'],packet_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
Path(sys.argv[2]).write_text(json.dumps(v))
""",
    )
    b = tmp_path / "builder.py"
    b.write_text("""import sys,json,shutil
from pathlib import Path
request=json.loads(Path(sys.argv[1]).read_text());shutil.copyfile(request['case'],sys.argv[2])
""")
    result = audit.run_loop(case, tmp_path / "loop", worker, builder=[sys.executable, str(b)])
    assert result["status"] == "no_progress"
    assert len(result["rounds"]) == 1


def test_codex_worker_uses_read_only_and_binds_schema(case, tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/codex_audit_worker.py"
    spec = importlib.util.spec_from_file_location("worker", script)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    run = tmp_path / "run"
    audit.prepare(case, run)
    packet, report = run / "packets/geometry.json", run / "reviews/geometry.json"
    cmd = worker.command(packet, report)
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert "--dangerously-bypass-approvals-and-sandbox" not in cmd
    assert "--model" not in cmd
    schema = audit.read_json(report.with_suffix(".schema.json"))
    assert schema["properties"]["packet_sha256"]["enum"] == [audit.digest(packet)]
    assert "checks.json" in schema["properties"]["evidence"]["items"]["enum"]
    assert all(".." not in p for p in schema["properties"]["evidence"]["items"]["enum"])


def test_codex_builder_has_separate_workspace_and_fixed_case_schema(case, tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/codex_builder_worker.py"
    spec = importlib.util.spec_from_file_location("builder", script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    packet = tmp_path / "repair.json"
    audit.save(packet, {"case": str(case), "decision": "decision.json"})
    report = tmp_path / "next-case.json"
    cmd = builder.command(packet, report)
    assert cmd[cmd.index("--sandbox") + 1] == "workspace-write"
    assert Path(cmd[cmd.index("--cd") + 1]) == tmp_path / "repair-workspace"
    assert "--add-dir" not in cmd
    schema = audit.read_json(report.with_suffix(".schema.json"))
    assert schema["properties"]["expected_objects"] == {"enum": [["a1"]]}


def test_repair_cannot_rebind_relative_observation_paths(case, tmp_path, monkeypatch):
    from lidar_video import audit_checks

    monkeypatch.setattr(
        audit_checks, "run_checks", lambda *a: {r: {"status": "fail", "findings": []} for r in audit.ROLES}
    )
    for name in ("mesh.ply", "pose.json", "mask.png"):
        (tmp_path / name).write_bytes(b"original")
    manifest = tmp_path / "meshes.json"
    audit.save(
        manifest,
        {
            "objects": {
                "a1": {"mesh": "mesh.ply", "pose": "pose.json", "observations": [{"mask": "mask.png"}]}
            }
        },
    )
    c = audit.read_json(case)
    c["mesh_manifest"] = str(manifest)
    audit.save(case, c)
    worker = worker_file(
        tmp_path,
        """import sys,json,hashlib
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads(p.read_text());v=r['report_contract']
v.update(status='fail',summary='Failure',findings=[],evidence=['checks.json'],packet_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
Path(sys.argv[2]).write_text(json.dumps(v))
""",
    )
    builder = tmp_path / "builder.py"
    builder.write_text("""import sys,json,shutil
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text());c=json.loads(Path(r['case']).read_text())
old=Path(c['mesh_manifest']);new=Path(sys.argv[2]).parent/'new';new.mkdir()
shutil.copyfile(old,new/'meshes.json');(new/'mask.png').write_bytes(b'changed mask')
c['mesh_manifest']=str(new/'meshes.json');Path(sys.argv[2]).write_text(json.dumps(c))
""")
    r = audit.run_loop(case, tmp_path / "loop", worker, builder=[sys.executable, str(builder)])
    assert r["status"] == "failed"
    assert "validation metadata" in r["error"]


def test_repaired_candidate_is_rechecked_in_a_new_bound_round(case, tmp_path, monkeypatch):
    from lidar_video import audit_checks

    scene = tmp_path / "broken"
    scene.mkdir()
    (scene / "artifact.txt").write_text("broken")
    c = audit.read_json(case)
    c["scene_dir"] = str(scene)
    audit.save(case, c)

    def checks(c, *args):
        repaired = (Path(c["scene_dir"]) / "artifact.txt").read_text() == "repaired"
        return {r: {"status": "pass" if repaired else "fail", "findings": []} for r in audit.ROLES}

    from pathlib import Path

    monkeypatch.setattr(audit_checks, "run_checks", checks)
    worker = worker_file(
        tmp_path,
        """import sys,json,hashlib
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads(p.read_text());v=r['report_contract']
v.update(status='pass',summary='Fixture review',findings=[],evidence=['checks.json'],packet_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
Path(sys.argv[2]).write_text(json.dumps(v))
""",
    )
    builder = tmp_path / "builder.py"
    builder.write_text("""import sys,json
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text());c=json.loads(Path(r['case']).read_text())
p=Path(sys.argv[2]).parent/'new-scene';p.mkdir();(p/'artifact.txt').write_text('repaired')
c['scene_dir']=str(p);Path(sys.argv[2]).write_text(json.dumps(c))
""")
    result = audit.run_loop(case, tmp_path / "loop", worker, builder=[sys.executable, str(builder)])
    assert result["status"] == "accepted"
    assert [r["status"] for r in result["rounds"]] == ["rejected", "accepted"]
    locks = [audit.read_json(tmp_path / "loop" / f"round-{i:02d}" / "lock.json") for i in (1, 2)]
    assert locks[0]["round_id"] != locks[1]["round_id"]
    assert locks[0]["inputs"] != locks[1]["inputs"]
    assert (scene / "artifact.txt").read_text() == "broken"


def test_policy_change_invalidates_even_unanimous_reviews(case, tmp_path, monkeypatch):
    policy = tmp_path / "policy.py"
    policy.write_text("original")
    monkeypatch.setattr(audit, "policy_hashes", lambda: {str(policy): audit.digest(policy)})
    run = tmp_path / "run"
    audit.prepare(case, run)
    reviews(run)
    policy.write_text("weakened")
    assert audit.finalize(run)["status"] == "invalid"


@pytest.mark.parametrize("mutation", ["delete", "change", "add"])
def test_supporting_check_artifacts_are_required_and_frozen(case, tmp_path, monkeypatch, mutation):
    from lidar_video import audit_checks

    def run_checks(c, root, out):
        out.mkdir()
        audit.save(out / "fresh_scenic_samples.json", {"samples": [{"fixture": 1}, {"fixture": 2}]})
        audit.save(out / "replayed_scenic_samples.json", {"samples": [{"fixture": 1}]})
        (out / "scenic_compile.log").write_text("Executed fixture sampler")
        return {
            role: dict(status="pass", findings=[dict(id="exports.shared_ids")] if role == "exports" else [])
            for role in audit.ROLES
        }

    monkeypatch.setattr(audit_checks, "run_checks", run_checks)
    run = tmp_path / "run"
    audit.prepare(case, run)
    reviews(run)
    assert audit.finalize(run)["status"] == "accepted"
    path = run / "checks/fresh_scenic_samples.json"
    if mutation == "delete":
        path.unlink()
    elif mutation == "change":
        audit.save(path, {"samples": []})
    else:
        (run / "checks/extra.json").write_text("{}")
    assert audit.finalize(run)["status"] == "invalid"


def test_pass_cannot_be_prepared_without_claimed_sample_artifact(case, tmp_path, monkeypatch):
    from lidar_video import audit_checks

    monkeypatch.setattr(
        audit_checks,
        "run_checks",
        lambda *args: {
            role: dict(status="pass", findings=[dict(id="exports.shared_ids")] if role == "exports" else [])
            for role in audit.ROLES
        },
    )
    with pytest.raises(ValueError, match="Missing supporting export evidence"):
        audit.prepare(case, tmp_path / "run")
