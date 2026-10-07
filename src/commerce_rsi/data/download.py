"""Pinned public downloads with byte, format and upstream LFS checksums."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .transfer import clear_chunks, fetch_file, request

REPO = "McAuley-Lab/Amazon-C4"
FILES = ("test.csv", "sampled_item_metadata_1M.jsonl", "README.md")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_url(url: str):
    with urllib.request.urlopen(request(url), timeout=45) as response:
        return json.load(response)


def _check_format(path: Path) -> None:
    if path.name == "test.csv":
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not {"qid", "query", "item_id", "user_id"}.issubset(reader.fieldnames or []):
                raise ValueError("Downloaded CSV does not have Amazon-C4 columns")
            if not next(reader, None):
                raise ValueError("Empty query file")
    elif path.name.endswith(".jsonl"):
        with path.open(encoding="utf-8") as stream:
            item = json.loads(next(stream))
        if not {"item_id", "category", "metadata"}.issubset(item):
            raise ValueError("Downloaded metadata has unexpected schema")


def download_dataset(output_dir: Path, endpoint: str = "https://huggingface.co",
                     revision: str | None = None, transfer_endpoint: str | None = None) -> dict:
    endpoint = endpoint.rstrip("/")
    transfer_endpoint = (transfer_endpoint or endpoint).rstrip("/")
    if not endpoint.startswith("https://") or not transfer_endpoint.startswith("https://"):
        raise ValueError("HTTPS is required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info = _json_url(f"{endpoint}/api/datasets/{REPO}" + (f"/revision/{revision}" if revision else ""))
    resolved = info["sha"]
    if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError("Source did not resolve to a full commit hash")
    if revision and revision != resolved:
        raise ValueError("Requested immutable revision does not match source")
    tree = _json_url(f"{endpoint}/api/datasets/{REPO}/tree/{resolved}?recursive=true&expand=false")
    entries = {entry["path"]: entry for entry in tree}
    existing_manifest = output_dir / "source_manifest.json"
    old = json.loads(existing_manifest.read_text()) if existing_manifest.exists() else None
    recorded = {}
    if old is not None:
        if not isinstance(old, dict):
            raise ValueError("Existing manifest is not an object")
        if old.get("dataset") != REPO:
            raise ValueError("Existing manifest belongs to a different dataset")
        if old.get("revision") != resolved:
            raise ValueError("Use a new output directory for a different dataset revision")
        if not isinstance(old.get("files"), list):
            raise ValueError("Existing manifest has no valid file inventory")
        for record in old["files"]:
            name = record.get("name") if isinstance(record, dict) else None
            if name not in FILES or name in recorded:
                raise ValueError("Existing manifest contains unknown or duplicate files")
            recorded[name] = record

    def valid(candidate: Path, entry: dict) -> bool:
        expected_size = entry["size"]
        if not candidate.is_file() or candidate.stat().st_size != expected_size:
            return False
        expected = entry.get("lfs", {}).get("oid")
        if expected:
            return sha256_file(candidate) == expected
        blob = hashlib.sha1(f"blob {expected_size}\0".encode() + candidate.read_bytes()).hexdigest()
        return blob == entry["oid"]

    # Revalidate the pinned remote metadata and every existing file before any
    # transfer or manifest write. A cache hit must not disguise stale/corrupt data.
    for name in FILES:
        if name not in entries:
            raise ValueError(f"Pinned source metadata does not contain {name}")
        entry, path = entries[name], output_dir / name
        if path.exists():
            if not valid(path, entry):
                raise ValueError(f"Existing file {path} has wrong checksum; refusing to overwrite")
            _check_format(path)
        if name in recorded:
            record = recorded[name]
            expected_record = {"bytes": entry["size"], "sha256": sha256_file(path) if path.is_file() else None,
                               "upstream_lfs_sha256": entry.get("lfs", {}).get("oid"),
                               "upstream_git_blob_sha1": entry["oid"]}
            if not path.is_file() or any(record.get(key) != value for key, value in expected_record.items()):
                raise ValueError(f"Existing manifest/file verification disagrees for {name}")
            if record.get("url") is None and record.get("acquisition") != "preexisting_verified_origin_unknown":
                raise ValueError(f"Existing manifest lacks acquisition provenance for {name}")
            if record.get("url") is not None and (not isinstance(record["url"], str) or not record["url"].startswith("https://")):
                raise ValueError(f"Existing manifest has invalid acquisition URL for {name}")
    if old is not None and set(recorded) == set(FILES):
        # Do not rewrite even whitespace: prepared protocols may hash these bytes.
        return old

    manifest = old if old is not None else {
        "dataset": REPO,
        "official_source": f"https://huggingface.co/datasets/{REPO}",
        "endpoint": endpoint,
        "transfer_endpoint": transfer_endpoint,
        "transfer_endpoint_note": "Initial configured endpoint; per-file URLs are authoritative when transfers resume through another endpoint.",
        "mirror_used": endpoint != "https://huggingface.co" or transfer_endpoint != endpoint,
        "revision": resolved,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_basis": "Sizes and LFS SHA256 from the repository tree at the pinned revision, fetched from the metadata endpoint; non-LFS files checked by Git blob SHA1. File transfer may use a different endpoint. Metadata is independently official exactly when endpoint is https://huggingface.co.",
        "files": [],
    }
    for name in FILES:
        if name in recorded:
            continue
        entry = entries[name]
        expected = entry.get("lfs", {}).get("oid")
        expected_size = entry["size"]
        path = output_dir / name
        url = f"{transfer_endpoint}/datasets/{REPO}/resolve/{resolved}/{name}"

        acquired_now = not path.exists()
        prior_partial = False
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".partial")
            prior_partial = partial.exists() or (partial.parent / (partial.name + ".chunks")).exists()
            start = time.monotonic()
            fetch_file(url, partial, expected_size, direct=transfer_endpoint != endpoint)
            if not valid(partial, entry):
                raise ValueError(f"Downloaded size/checksum mismatch for {name}")
            partial.rename(path)
            clear_chunks(partial)
            print(json.dumps({"downloaded": name, "bytes": expected_size,
                              "seconds": round(time.monotonic() - start, 3)}), flush=True)
        _check_format(path)
        manifest["files"].append({"name": name, "url": url if acquired_now else None, "bytes": path.stat().st_size,
                                  "sha256": sha256_file(path), "upstream_lfs_sha256": expected,
                                  "upstream_git_blob_sha1": entry["oid"],
                                  "acquisition": "downloaded" if acquired_now else "preexisting_verified_origin_unknown",
                                  "prior_partial_origin": "unrecorded; URL identifies the finishing attempt" if prior_partial else None,
                                  "retrieved_at_utc": datetime.now(timezone.utc).isoformat() if acquired_now else None})
        manifest["mirror_used"] = bool(manifest.get("mirror_used")) or endpoint != "https://huggingface.co" or (acquired_now and transfer_endpoint != "https://huggingface.co")
        manifest["status"] = "complete" if len(manifest["files"]) == len(FILES) else "partial"
        # Save verified per-file provenance before fetching the next file. A retry
        # through a different endpoint must preserve the earlier acquisition URL.
        existing_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
