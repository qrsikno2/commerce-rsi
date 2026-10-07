"""Acquire the specified public research model with official per-file hashes."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from commerce_rsi.data.download import _json_url, sha256_file
from commerce_rsi.data.transfer import clear_chunks, fetch_file

MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"


def download_model(output: Path, transfer_endpoint="https://hf-mirror.com"):
    output = Path(output)
    transfer_endpoint = transfer_endpoint.rstrip("/")
    if not transfer_endpoint.startswith("https://"):
        raise ValueError("HTTPS is required")
    output.mkdir(parents=True, exist_ok=True)
    info = _json_url(f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}")
    if info["sha"] != REVISION:
        raise ValueError("Unexpected model revision")
    tree = _json_url(f"https://huggingface.co/api/models/{MODEL}/tree/{REVISION}?recursive=true&expand=false")
    entries = {}
    for entry in tree:
        name = entry["path"]
        if entry["type"] != "file" or name.startswith("."):
            continue
        if "/" in name or name == "source_manifest.json" or name in entries:
            raise ValueError("Unexpected nested, reserved, or duplicate model file")
        entries[name] = entry
    if not entries:
        raise ValueError("Pinned model metadata has no downloadable files")
    manifest_path = output / "source_manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    recorded = {}
    if old is not None:
        if not isinstance(old, dict) or old.get("model_id") != MODEL or old.get("revision") != REVISION:
            raise ValueError("Existing model manifest identity/revision differs from the requested model")
        if not isinstance(old.get("files"), list):
            raise ValueError("Existing model manifest has no valid file inventory")
        for record in old["files"]:
            name = record.get("name") if isinstance(record, dict) else None
            if name not in entries or name in recorded:
                raise ValueError("Existing model manifest contains unknown or duplicate files")
            recorded[name] = record

    def valid(candidate, entry):
        size = entry["size"]
        if not candidate.is_file() or candidate.stat().st_size != size:
            return False
        expected = entry.get("lfs", {}).get("oid")
        if expected:
            return sha256_file(candidate) == expected
        return hashlib.sha1(f"blob {size}\0".encode() + candidate.read_bytes()).hexdigest() == entry["oid"]

    for name, entry in entries.items():
        path = output / name
        if path.exists() and not valid(path, entry):
            raise ValueError(f"Existing model file has wrong checksum: {name}")
        if name in recorded:
            expected_record = {"bytes": entry["size"], "sha256": sha256_file(path) if path.is_file() else None,
                               "upstream_lfs_sha256": entry.get("lfs", {}).get("oid")}
            if not path.is_file() or any(recorded[name].get(key) != value for key, value in expected_record.items()):
                raise ValueError(f"Existing model manifest/file verification disagrees for {name}")
    if old is not None and set(recorded) == set(entries):
        # The backend hashes this JSON with the model files. Rewriting a verified
        # receipt would change checkpoint identity without changing any weights.
        return old

    manifest = old if old is not None else {"model_id": MODEL, "revision": REVISION,
                "metadata_endpoint": "https://huggingface.co", "transfer_endpoint": transfer_endpoint,
                "transfer_endpoint_note": "Initial configured endpoint; per-file URLs are authoritative when transfers resume through another endpoint.",
                "license": info["cardData"]["license_name"],
                "purpose": "Non-commercial research/evaluation under the Qwen Research License",
                "parameter_count": info["safetensors"]["total"],
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "files": []}
    for name, entry in entries.items():
        if name in recorded:
            continue
        path = output / name
        size = entry["size"]
        expected = entry.get("lfs", {}).get("oid")
        url = f"{transfer_endpoint}/{MODEL}/resolve/{REVISION}/{name}"
        acquired_now = not path.exists()
        prior_partial = False
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".partial")
            prior_partial = partial.exists() or (partial.parent / (partial.name + ".chunks")).exists()
            fetch_file(url, partial, size, direct=transfer_endpoint != "https://huggingface.co", workers=4)
            if not valid(partial, entry):
                raise ValueError(f"Checksum mismatch: {name}")
            partial.rename(path)
            clear_chunks(partial)
        manifest["files"].append({"name": name, "bytes": size, "sha256": sha256_file(path),
                                  "upstream_lfs_sha256": expected, "url": url if acquired_now else None,
                                  "acquisition": "downloaded" if acquired_now else "preexisting_verified_origin_unknown",
                                  "prior_partial_origin": "unrecorded; URL identifies the finishing attempt" if prior_partial else None})
        manifest["status"] = "complete" if len(manifest["files"]) == len(entries) else "partial"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"verified": name, "bytes": size}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--transfer-endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()
    download_model(args.output, args.transfer_endpoint)
