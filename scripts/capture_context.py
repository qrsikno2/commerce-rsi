"""Archive code hashes and explicit dependency versions without credentials."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
names = ("torch", "numpy", "transformers", "peft", "accelerate", "safetensors", "pyarrow",
         "huggingface-hub", "tokenizers", "pytest", "setuptools", "pip")
versions = {name: importlib.metadata.version(name) for name in names}
files = sorted(path for directory in ("src", "scripts", "configs", "patches")
               for path in (root / directory).rglob("*")
               if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
report = {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
          "git_commit": git.stdout.strip() if git.returncode == 0 else None,
          "versions": versions,
          "sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"output": str(args.output), "source_files": len(files), "versions": versions}))
