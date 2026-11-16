import base64
import hashlib
import io
import json
import zipfile

import pytest

from lidar_video.run_checkpoint import notebook_chunks, pack, payload_from_notebook, restore


def test_chunked_notebook_roundtrip_and_truncation(tmp_path):
    (tmp_path / "result").write_text("complete")
    payload = pack(tmp_path, ["result"], stage="completed")
    chunks = notebook_chunks(payload, chunk_chars=100)
    notebook = {"cells": [{"outputs": [{"data": {"application/json": x}}]} for x in reversed(chunks)]}
    recovered = payload_from_notebook(notebook, expected_sha256=payload["sha256"])
    assert recovered == payload
    restore(recovered, tmp_path / "verified", expected_sha256=payload["sha256"])
    # Colab saved an output-hidden placeholder even though the live page showed it.
    notebook["cells"][0]["outputs"] = [{"data": {"text/plain": "Output hidden; open in Colab to view."}}]
    with pytest.raises(ValueError, match="Incomplete"):
        payload_from_notebook(notebook, expected_sha256=payload["sha256"])


def test_rejects_duplicate_mixed_and_tampered_notebook_chunks(tmp_path):
    (tmp_path / "result").write_text("complete")
    payload = pack(tmp_path, ["result"], stage="completed")
    chunks = notebook_chunks(payload, chunk_chars=100)

    def notebook(pieces):
        return {"cells": [{"outputs": [{"data": {"application/json": x}}]} for x in pieces]}

    with pytest.raises(ValueError, match="Duplicate"):
        payload_from_notebook(notebook([chunks[0], *chunks[:-1]]), expected_sha256=payload["sha256"])
    with pytest.raises(ValueError, match="Incomplete"):
        payload_from_notebook(
            notebook([dict(chunks[0], sha256="0" * 64), *chunks[1:]]), expected_sha256=payload["sha256"]
        )
    forged = dict(payload, sha256="0" * 64)
    forged_chunks = [dict(x, sha256=payload["sha256"]) for x in notebook_chunks(forged, chunk_chars=100)]
    with pytest.raises(ValueError, match="identity"):
        payload_from_notebook(notebook(forged_chunks), expected_sha256=payload["sha256"])


def test_notebook_roundtrip_preserves_binary_and_refuses_overwrite(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "masks.npz").write_bytes(bytes(range(256)))
    (root / "status.json").write_text('{"status":"failed"}')
    payload = pack(root, ["masks.npz", "status.json"], stage="detected")
    notebook = json.loads(json.dumps({"cells": [{"outputs": [{"data": {"application/json": payload}}]}]}))
    recovered = notebook["cells"][0]["outputs"][0]["data"]["application/json"]
    out = restore(recovered, tmp_path / "restored", expected_sha256=payload["sha256"])
    assert (out / "masks.npz").read_bytes() == (root / "masks.npz").read_bytes()
    assert json.loads((out / "checkpoint-receipt.json").read_text())["verified"]
    with pytest.raises(FileExistsError):
        restore(recovered, out, expected_sha256=payload["sha256"])


def test_rejects_tamper_limits_and_external_files(tmp_path):
    (tmp_path / "a").write_text("abc")
    payload = pack(tmp_path, ["a"], stage="test")
    with pytest.raises(ValueError, match="identity"):
        restore(payload, tmp_path / "bad", expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="hash or size"):
        restore(
            dict(payload, data_base64=base64.b64encode(b"bad").decode()),
            tmp_path / "bad",
            expected_sha256=payload["sha256"],
        )
    with pytest.raises(ValueError, match="byte limit"):
        pack(tmp_path, ["a"], stage="test", max_bytes=2)
    with pytest.raises(ValueError, match="path"):
        pack(tmp_path, ["../outside"], stage="test")
    (tmp_path / "linked").symlink_to(tmp_path / "a")
    with pytest.raises(ValueError, match="escapes"):
        pack(tmp_path, ["linked"], stage="test")
    assert not (tmp_path / "bad").exists()


def test_even_hash_consistent_archive_cannot_escape(tmp_path):
    buffer = io.BytesIO()
    data = b"untrusted"
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("../escape", data)
        z.writestr(
            "MANIFEST.json",
            json.dumps({"../escape": dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())}),
        )
    blob = buffer.getvalue()
    payload = dict(
        schema="development-checkpoint-v1",
        stage="bad",
        files=1,
        bytes=len(blob),
        sha256=hashlib.sha256(blob).hexdigest(),
        data_base64=base64.b64encode(blob).decode(),
    )
    with pytest.raises(ValueError, match="path"):
        restore(payload, tmp_path / "bad", expected_sha256=payload["sha256"])
    assert not (tmp_path / "bad").exists()
