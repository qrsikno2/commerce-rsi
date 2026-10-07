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
    output.mkdir(parents=True, exist_ok=True)
    info = _json_url(f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}")
    if info["sha"] != REVISION:
        raise ValueError("Unexpected model revision")
    tree = _json_url(f"https://huggingface.co/api/models/{MODEL}/tree/{REVISION}?recursive=true&expand=false")
    manifest = {"model_id": MODEL, "revision": REVISION,
                "metadata_endpoint": "https://huggingface.co", "transfer_endpoint": transfer_endpoint,
                "license": info["cardData"]["license_name"],
                "purpose": "Non-commercial research/evaluation under the Qwen Research License",
                "parameter_count": info["safetensors"]["total"],
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "files": []}
    for entry in tree:
        name = entry["path"]
        if entry["type"] != "file" or name.startswith("."):
            continue
        if "/" in name:
            raise ValueError("Unexpected nested model file")
        path = output / name
        size = entry["size"]
        expected = entry.get("lfs", {}).get("oid")

        def valid(candidate):
            if not candidate.exists() or candidate.stat().st_size != size:
                return False
            if expected:
                return sha256_file(candidate) == expected
            return hashlib.sha1(f"blob {size}\0".encode() + candidate.read_bytes()).hexdigest() == entry["oid"]

        if path.exists() and not valid(path):
            raise ValueError(f"Existing model file has wrong checksum: {name}")
        url = f"{transfer_endpoint}/{MODEL}/resolve/{REVISION}/{name}"
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".partial")
            fetch_file(url, partial, size, direct=transfer_endpoint != "https://huggingface.co", workers=4)
            if not valid(partial):
                raise ValueError(f"Checksum mismatch: {name}")
            partial.rename(path)
            clear_chunks(partial)
        manifest["files"].append({"name": name, "bytes": size, "sha256": sha256_file(path),
                                  "upstream_lfs_sha256": expected, "url": url})
        print(json.dumps({"verified": name, "bytes": size}), flush=True)
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--transfer-endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()
    download_model(args.output, args.transfer_endpoint)
