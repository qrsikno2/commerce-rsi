"""Public-download provenance tests with tiny bytes and mocked HTTPS metadata."""

import hashlib
import json

import pytest

from commerce_rsi.data import download
from commerce_rsi.agent_training import model_source


REVISION = "a" * 40
OFFICIAL = "https://huggingface.co"
MIRROR = "https://hf-mirror.example"


@pytest.fixture
def remote(monkeypatch):
    payloads = {
        "test.csv": b"qid,query,item_id,user_id,ori_rating,ori_review\nq,request,p,u,5,private\n",
        "sampled_item_metadata_1M.jsonl": b'{"item_id":"p","category":"Toys","metadata":"public title"}\n',
        "README.md": b"# Public fixture dataset\n",
    }
    entries = []
    for name, data in payloads.items():
        entry = {"path": name, "size": len(data),
                 "oid": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()}
        if name != "README.md":
            entry["lfs"] = {"oid": hashlib.sha256(data).hexdigest()}
        entries.append(entry)
    calls, transfers = [], []

    def metadata(url):
        assert url.startswith(OFFICIAL + "/api/")
        calls.append(url)
        return entries if "/tree/" in url else {"sha": REVISION}

    def fetch(url, path, size, **kwargs):
        name = url.rsplit("/", 1)[1]
        transfers.append(url)
        assert size == len(payloads[name])
        path.write_bytes(payloads[name])

    monkeypatch.setattr(download, "_json_url", metadata)
    monkeypatch.setattr(download, "fetch_file", fetch)
    monkeypatch.setattr(download, "clear_chunks", lambda _: None)
    return payloads, entries, calls, transfers, fetch


def test_fresh_download_records_actual_transfer_and_rerun_preserves_every_manifest_byte(remote, tmp_path):
    _, _, calls, transfers, _ = remote
    report = download.download_dataset(tmp_path, revision=REVISION, transfer_endpoint=MIRROR)
    assert len(transfers) == 3
    assert report["status"] == "complete" and report["mirror_used"]
    assert all(row["url"].startswith(MIRROR) and row["acquisition"] == "downloaded" for row in report["files"])
    # Existing source notes and formatting must survive, not merely equal JSON values.
    report["verification_basis"] = "Historically verified original transfer notes"
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps(report, indent=3) + "\n\n")
    before, modified = manifest.read_bytes(), manifest.stat().st_mtime_ns
    returned = download.download_dataset(tmp_path, revision=REVISION, transfer_endpoint=OFFICIAL)
    assert returned == report
    assert manifest.read_bytes() == before and manifest.stat().st_mtime_ns == modified
    assert len(calls) == 4  # Fresh API + pinned tree verification on both invocations.
    assert len(transfers) == 3


def test_interrupted_download_preserves_first_actual_url_when_resume_changes_endpoint(remote, monkeypatch, tmp_path):
    _, _, _, transfers, fetch = remote

    def interrupted(url, path, size, **kwargs):
        if url.endswith("sampled_item_metadata_1M.jsonl"):
            raise OSError("simulated interruption")
        return fetch(url, path, size, **kwargs)

    monkeypatch.setattr(download, "fetch_file", interrupted)
    with pytest.raises(OSError, match="interruption"):
        download.download_dataset(tmp_path, revision=REVISION)
    partial = json.loads((tmp_path / "source_manifest.json").read_text())
    assert partial["status"] == "partial" and len(partial["files"]) == 1
    first = partial["files"][0].copy()
    assert first["url"].startswith(OFFICIAL)
    monkeypatch.setattr(download, "fetch_file", fetch)
    complete = download.download_dataset(tmp_path, revision=REVISION, transfer_endpoint=MIRROR)
    assert complete["files"][0] == first
    assert all(row["url"].startswith(MIRROR) for row in complete["files"][1:])
    assert complete["status"] == "complete" and complete["mirror_used"]
    assert len(transfers) == 3


@pytest.mark.parametrize("corruption", ["local_bytes", "recorded_hash", "recorded_revision", "upstream_checksum", "missing_file"])
def test_existing_manifest_is_not_trusted_without_current_integrity_checks(remote, tmp_path, corruption):
    _, entries, _, transfers, _ = remote
    download.download_dataset(tmp_path, revision=REVISION)
    path = tmp_path / "source_manifest.json"
    if corruption == "local_bytes":
        target = tmp_path / "test.csv"
        target.write_bytes(b"X" * target.stat().st_size)
    elif corruption == "recorded_hash":
        raw = json.loads(path.read_text())
        raw["files"][0]["sha256"] = "f" * 64
        path.write_text(json.dumps(raw))
    elif corruption == "recorded_revision":
        raw = json.loads(path.read_text())
        raw["revision"] = "b" * 40
        path.write_text(json.dumps(raw))
    elif corruption == "upstream_checksum":
        entries[0]["lfs"]["oid"] = "f" * 64
    else:
        (tmp_path / "test.csv").unlink()
    before = path.read_bytes()
    with pytest.raises(ValueError):
        download.download_dataset(tmp_path, revision=REVISION)
    assert path.read_bytes() == before
    assert len(transfers) == 3


def test_unrecorded_existing_bytes_are_not_claimed_as_new_transfer(remote, tmp_path):
    payloads, _, _, transfers, _ = remote
    (tmp_path / "test.csv").write_bytes(payloads["test.csv"])
    report = download.download_dataset(tmp_path, revision=REVISION, transfer_endpoint=MIRROR)
    prior = report["files"][0]
    assert prior["url"] is None and prior["retrieved_at_utc"] is None
    assert prior["acquisition"] == "preexisting_verified_origin_unknown"
    assert len(transfers) == 2
    before = (tmp_path / "source_manifest.json").read_bytes()
    assert download.download_dataset(tmp_path, revision=REVISION) == report
    assert (tmp_path / "source_manifest.json").read_bytes() == before


def test_existing_partial_origins_are_explicitly_unknown(remote, tmp_path):
    (tmp_path / "test.csv.partial").write_bytes(b"earlier interrupted transfer")
    report = download.download_dataset(tmp_path, revision=REVISION, transfer_endpoint=MIRROR)
    assert report["files"][0]["prior_partial_origin"] == "unrecorded; URL identifies the finishing attempt"


def test_requested_revision_must_match_fresh_metadata(remote, tmp_path):
    with pytest.raises(ValueError, match="immutable revision"):
        download.download_dataset(tmp_path, revision="b" * 40)
    assert not (tmp_path / "source_manifest.json").exists()


@pytest.fixture
def model_remote(monkeypatch):
    payloads = {"config.json": b'{"model_type":"test-fixture-only"}\n',
                "model.safetensors": b"tiny inert test bytes; not real model weights",
                "README.md": b"# Mock model card\n"}
    entries = []
    for name, data in payloads.items():
        entry = {"path": name, "type": "file", "size": len(data),
                 "oid": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()}
        if name.endswith(".safetensors"):
            entry["lfs"] = {"oid": hashlib.sha256(data).hexdigest()}
        entries.append(entry)
    calls, transfers = [], []

    def metadata(url):
        assert url.startswith(OFFICIAL + "/api/models/")
        calls.append(url)
        return entries if "/tree/" in url else {"sha": model_source.REVISION,
                                                 "cardData": {"license_name": "test-license"},
                                                 "safetensors": {"total": 0}}

    def fetch(url, path, size, **kwargs):
        name = url.rsplit("/", 1)[1]
        transfers.append(url)
        assert size == len(payloads[name])
        path.write_bytes(payloads[name])

    monkeypatch.setattr(model_source, "_json_url", metadata)
    monkeypatch.setattr(model_source, "fetch_file", fetch)
    monkeypatch.setattr(model_source, "clear_chunks", lambda _: None)
    return payloads, entries, calls, transfers, fetch


def test_model_receipt_preservation_keeps_existing_base_checkpoint_hash(model_remote, tmp_path):
    # Only these tiny fixture bytes are hashed; no downloaded model is loaded/read.
    from commerce_rsi.agent_training.backend import base_model_hash

    _, _, calls, transfers, _ = model_remote
    report = model_source.download_model(tmp_path, transfer_endpoint=MIRROR)
    report["downloaded_at_utc"] = "2020-01-01T00:00:00+00:00"
    report["verification_notes"] = "Preserve original route, timestamp, and formatting"
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps(report, indent=3) + "\n\n")
    before, modified = manifest.read_bytes(), manifest.stat().st_mtime_ns
    policy_base_hash = base_model_hash(tmp_path)
    returned = model_source.download_model(tmp_path, transfer_endpoint=OFFICIAL)
    assert returned == report
    assert manifest.read_bytes() == before and manifest.stat().st_mtime_ns == modified
    assert base_model_hash(tmp_path) == policy_base_hash
    assert len(calls) == 4 and len(transfers) == 3


@pytest.mark.parametrize("corruption", ["model_id", "revision", "recorded_hash", "weights", "upstream_checksum"])
def test_model_rerun_rejects_mismatched_identity_or_bytes(model_remote, tmp_path, corruption):
    _, entries, _, transfers, _ = model_remote
    model_source.download_model(tmp_path, transfer_endpoint=MIRROR)
    path = tmp_path / "source_manifest.json"
    raw = json.loads(path.read_text())
    if corruption in {"model_id", "revision"}:
        raw[corruption] = "different"
        path.write_text(json.dumps(raw))
    elif corruption == "recorded_hash":
        raw["files"][0]["sha256"] = "f" * 64
        path.write_text(json.dumps(raw))
    elif corruption == "weights":
        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"X" * weights.stat().st_size)
    else:
        entries[1]["lfs"]["oid"] = "f" * 64
    before = path.read_bytes()
    with pytest.raises(ValueError):
        model_source.download_model(tmp_path)
    assert path.read_bytes() == before and len(transfers) == 3


def test_model_resume_retains_each_actual_transfer_url(model_remote, monkeypatch, tmp_path):
    _, _, _, transfers, fetch = model_remote

    def interrupted(url, path, size, **kwargs):
        if url.endswith("model.safetensors"):
            raise OSError("simulated interruption")
        return fetch(url, path, size, **kwargs)

    monkeypatch.setattr(model_source, "fetch_file", interrupted)
    with pytest.raises(OSError):
        model_source.download_model(tmp_path, transfer_endpoint=OFFICIAL)
    partial = json.loads((tmp_path / "source_manifest.json").read_text())
    assert partial["status"] == "partial" and len(partial["files"]) == 1
    first = partial["files"][0].copy()
    monkeypatch.setattr(model_source, "fetch_file", fetch)
    completed = model_source.download_model(tmp_path, transfer_endpoint=MIRROR)
    assert completed["files"][0] == first
    assert completed["files"][0]["url"].startswith(OFFICIAL)
    assert all(row["url"].startswith(MIRROR) for row in completed["files"][1:])
    assert len(transfers) == 3
