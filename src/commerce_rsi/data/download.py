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
    output_dir.mkdir(parents=True, exist_ok=True)
    info = _json_url(f"{endpoint}/api/datasets/{REPO}" + (f"/revision/{revision}" if revision else ""))
    resolved = info["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError("Source did not resolve to a full commit hash")
    if revision and revision != resolved:
        raise ValueError("Requested immutable revision does not match source")
    tree = _json_url(f"{endpoint}/api/datasets/{REPO}/tree/{resolved}?recursive=true&expand=false")
    entries = {entry["path"]: entry for entry in tree}
    manifest = {
        "dataset": REPO,
        "official_source": f"https://huggingface.co/datasets/{REPO}",
        "endpoint": endpoint,
        "transfer_endpoint": transfer_endpoint,
        "mirror_used": endpoint != "https://huggingface.co" or transfer_endpoint != endpoint,
        "revision": resolved,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_basis": "Sizes and LFS SHA256 from the repository tree at the pinned revision, fetched from the metadata endpoint; non-LFS files checked by Git blob SHA1. File transfer may use a different endpoint. Metadata is independently official exactly when endpoint is https://huggingface.co.",
        "files": [],
    }
    existing_manifest = output_dir / "source_manifest.json"
    if existing_manifest.exists():
        old = json.loads(existing_manifest.read_text())
        if old["revision"] != resolved:
            raise ValueError("Use a new output directory for a different dataset revision")
    for name in FILES:
        entry = entries[name]
        expected = entry.get("lfs", {}).get("oid")
        expected_size = entry["size"]
        path = output_dir / name
        url = f"{transfer_endpoint}/datasets/{REPO}/resolve/{resolved}/{name}"

        def valid(candidate: Path) -> bool:
            if not candidate.exists() or candidate.stat().st_size != expected_size:
                return False
            if expected:
                return sha256_file(candidate) == expected
            blob = hashlib.sha1(f"blob {expected_size}\0".encode() + candidate.read_bytes()).hexdigest()
            return blob == entry["oid"]

        if path.exists() and not valid(path):
            raise ValueError(f"Existing file {path} has wrong checksum; refusing to overwrite")
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".partial")
            start = time.monotonic()
            fetch_file(url, partial, expected_size, direct=transfer_endpoint != endpoint)
            if not valid(partial):
                raise ValueError(f"Downloaded size/checksum mismatch for {name}")
            partial.rename(path)
            clear_chunks(partial)
            print(json.dumps({"downloaded": name, "bytes": expected_size,
                              "seconds": round(time.monotonic() - start, 3)}), flush=True)
        _check_format(path)
        manifest["files"].append({"name": name, "url": url, "bytes": path.stat().st_size,
                                  "sha256": sha256_file(path), "upstream_lfs_sha256": expected,
                                  "upstream_git_blob_sha1": entry["oid"]})
    existing_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
