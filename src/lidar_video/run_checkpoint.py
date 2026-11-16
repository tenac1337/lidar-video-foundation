"""Small, explicit artifact checkpoints recoverable from saved notebook outputs.

Emit the returned JSON with IPython display under application/json. Persistence
is only confirmed after reading the saved notebook back and restoring its hash.
This is a bounded transport, not automatic cloud storage or an authenticity proof.
"""

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

SCHEMA = "development-checkpoint-v1"
MAX_BYTES = 16 * 1024 * 1024


def notebook_chunks(payload, chunk_chars=1_100_000):
    """Emit each piece in a separate cell to avoid notebook output truncation.

    Colab can replace a large cell's saved output with an 'Output hidden' stub,
    even when the live page displays the data. A saved notebook must still be
    recovered and hash-verified before acknowledging a checkpoint.
    """
    if type(chunk_chars) is not int or not 1 <= chunk_chars <= 1_100_000:
        raise ValueError("Invalid notebook chunk size")
    if payload.get("schema") != SCHEMA:
        raise ValueError("Unexpected checkpoint schema")
    encoded = json.dumps(payload, ensure_ascii=True)
    if len(encoded) > 2 * MAX_BYTES:
        raise ValueError("Checkpoint JSON exceeds transport limit")
    pieces = [encoded[i : i + chunk_chars] for i in range(0, len(encoded), chunk_chars)]
    if len(pieces) > 128:
        raise ValueError("Too many notebook chunks")
    return [
        dict(checkpoint_chunk=i, total=len(pieces), sha256=payload["sha256"], text=text)
        for i, text in enumerate(pieces)
    ]


def payload_from_notebook(notebook, *, expected_sha256):
    """Read JSON outputs as data only; reject incomplete, mixed or duplicate pieces.

    This checks transport structure. Call restore to verify the archive and every
    member before acknowledging persistence. Notebook source is never executed.
    """
    items = [
        output.get("data", {}).get("application/json")
        for cell in notebook.get("cells", [])
        for output in cell.get("outputs", [])
    ]
    items = [x for x in items if isinstance(x, dict) and x.get("sha256") == expected_sha256]
    if len(items) == 1 and items[0].get("schema") == SCHEMA:
        return items[0]
    if not items or len(items) > 128:
        raise ValueError("Missing or oversized notebook checkpoint")
    total = items[0].get("total")
    if type(total) is not int or total != len(items):
        raise ValueError("Incomplete notebook checkpoint")
    indices = [x.get("checkpoint_chunk") for x in items]
    if any(type(i) is not int for i in indices) or set(indices) != set(range(total)):
        raise ValueError("Duplicate or invalid notebook chunks")
    if any(
        x.get("total") != total or not isinstance(x.get("text"), str) or len(x["text"]) > 1_100_000
        for x in items
    ):
        raise ValueError("Invalid notebook chunk content")
    if sum(len(x["text"]) for x in items) > 2 * MAX_BYTES:
        raise ValueError("Notebook checkpoint exceeds transport limit")
    payload = json.loads("".join(x["text"] for x in sorted(items, key=lambda x: x["checkpoint_chunk"])))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SCHEMA
        or payload.get("sha256") != expected_sha256
    ):
        raise ValueError("Unexpected checkpoint identity")
    return payload


def safe_name(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Invalid checkpoint path")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or str(path) != name
        or name in {"MANIFEST.json", "checkpoint-receipt.json", "."}
    ):
        raise ValueError("Invalid checkpoint path")
    return name


def pack(root, paths, *, stage, max_bytes=MAX_BYTES):
    """Pack only the explicitly listed files. Never discover files or credentials."""
    root = Path(root).resolve()
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("A checkpoint stage is required")
    names = [safe_name(str(p)) for p in paths]
    if not names or len(names) > 256 or len(names) != len(set(names)):
        raise ValueError("Expected 1..256 unique files")
    manifest, blobs, total = {}, {}, 0
    for name in sorted(names):
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError("Checkpoint file escapes its root or is missing")
        size = path.stat().st_size
        if total + size > max_bytes:
            raise ValueError("Checkpoint exceeds byte limit; split it into stages")
        blob = path.read_bytes()
        total += len(blob)
        if total > max_bytes:
            raise ValueError("Checkpoint changed beyond byte limit")
        blobs[name] = blob
        manifest[name] = dict(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, blob in blobs.items():
            archive.writestr(name, blob)
        archive.writestr("MANIFEST.json", json.dumps(manifest, sort_keys=True))
    blob = buffer.getvalue()
    if len(blob) > max_bytes + 1024 * 1024:
        raise ValueError("Compressed checkpoint exceeds transport limit")
    return dict(
        schema=SCHEMA,
        stage=stage,
        bytes=len(blob),
        files=len(blobs),
        sha256=hashlib.sha256(blob).hexdigest(),
        data_base64=base64.b64encode(blob).decode(),
    )


def restore(payload, destination, *, expected_sha256, max_bytes=MAX_BYTES):
    """Verify all members before creating a new directory; refuse overwrite."""
    if payload.get("schema") != SCHEMA or payload.get("sha256") != expected_sha256:
        raise ValueError("Unexpected checkpoint identity")
    encoded = payload.get("data_base64")
    if not isinstance(encoded, str) or len(encoded) > 4 * (max_bytes + 1024 * 1024) // 3 + 4:
        raise ValueError("Invalid checkpoint size")
    blob = base64.b64decode(encoded, validate=True)
    if len(blob) != payload.get("bytes") or hashlib.sha256(blob).hexdigest() != expected_sha256:
        raise ValueError("Checkpoint hash or size mismatch")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        members = archive.infolist()
        names = [i.filename for i in members]
        if (
            len(names) != len(set(names))
            or names.count("MANIFEST.json") != 1
            or not 2 <= len(names) <= 257
            or sum(i.file_size for i in members) > max_bytes + 65536
        ):
            raise ValueError("Invalid checkpoint inventory or expansion size")
        manifest = json.loads(archive.read("MANIFEST.json"))
        if set(names) != set(manifest) | {"MANIFEST.json"} or len(manifest) != payload.get("files"):
            raise ValueError("Checkpoint manifest mismatch")
        if any(str(parent) in manifest for name in manifest for parent in PurePosixPath(name).parents):
            raise ValueError("Checkpoint member paths conflict")
        verified = {}
        for name, spec in manifest.items():
            safe_name(name)
            item = archive.getinfo(name)
            if item.is_dir() or ((item.external_attr >> 16) & 0o170000) not in (0, 0o100000):
                raise ValueError("Only regular checkpoint files are allowed")
            data = archive.read(name)
            if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise ValueError("Checkpoint member hash mismatch")
            verified[name] = data
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in verified.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (destination / "checkpoint-receipt.json").write_text(
        json.dumps(
            dict(
                schema=SCHEMA, stage=payload["stage"], sha256=expected_sha256, files=manifest, verified=True
            ),
            indent=2,
        )
        + "\n"
    )
    return destination
