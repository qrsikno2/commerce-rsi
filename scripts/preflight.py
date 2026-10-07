"""Record only explicit hardware/runtime facts, never environment credentials."""

import json
import platform
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


def capture(args):
    return subprocess.run(args, capture_output=True, text=True, timeout=20, check=True).stdout.strip()


output = Path("artifacts/preflight.json")
disk = shutil.disk_usage(Path.cwd())
report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
          "hostname": socket.gethostname(), "project_path": str(Path.cwd()),
          "python": platform.python_version(), "torch": torch.__version__,
          "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
          "gpu": capture(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
          "mount": capture(["findmnt", "-T", str(Path.cwd()), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"]),
          "volume_total_bytes": disk.total, "volume_free_bytes": disk.free,
          "container_memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
          "container_cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
          "environment": "Dedicated venv; preinstalled CUDA PyTorch reused read-only through system-site-packages",
          "network": "Official Hugging Face metadata via server-local http://127.0.0.1:7890; mirrored file transfers verified with official digests"}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
