"""Recreate the pinned, minimally patched OpenRSI checkout without overwriting it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ORIGIN = "https://github.com/FrontisAI/OpenRSI.git"
COMMIT = "71ae803a035d5e3b78c19fa49ed9f67d0550cbaa"
BRANCH = "commerce-rsi-adapter"
PATCH_SHA256 = "5825091d125cd08b37e81b4df0609fe37c09735aaea06521b31f374f72cd95d9"
PROJECT = Path(__file__).resolve().parents[1]


def command(args: list[str], *, check: bool = True, timeout: int = 30):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{result.stderr.strip()}")
    return result


def bootstrap(checkout: Path, *, check_only: bool = False) -> dict:
    patch = PROJECT / "patches" / "openrsi-single-card.patch"
    if hashlib.sha256(patch.read_bytes()).hexdigest() != PATCH_SHA256:
        raise ValueError("Stored OpenRSI patch SHA256 differs from the reviewed patch")
    checkout = checkout.absolute()
    if checkout.is_symlink():
        raise ValueError("Checkout path is a symlink; refusing to modify an indirect target")
    existed = checkout.exists()
    if not existed and check_only:
        return {"status": "would_clone_and_patch", "checkout": str(checkout),
                "source": ORIGIN, "commit": COMMIT, "branch": BRANCH, "patch_sha256": PATCH_SHA256}
    if not existed:
        checkout.parent.mkdir(parents=True, exist_ok=True)
        # Clone only public source. No installation, model download, hook, driver,
        # or daemon setup is part of this script. Interrupted clones are retained.
        command(["git", "clone", "--no-checkout", ORIGIN, str(checkout)], timeout=180)
        command(["git", "-C", str(checkout), "switch", "--create", BRANCH, COMMIT])

    def git(*args: str, check: bool = True):
        return command(["git", "-C", str(checkout), *args], check=check)

    repository_root = Path(git("rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if repository_root != checkout.resolve():
        raise ValueError("Existing target is not the root of its own Git checkout")
    if git("remote", "get-url", "origin").stdout.strip().removesuffix(".git") != ORIGIN.removesuffix(".git"):
        raise ValueError("Existing checkout origin is not the official OpenRSI repository")
    head = git("rev-parse", "HEAD").stdout.strip()
    if head != COMMIT:
        raise ValueError(f"Existing HEAD is {head}; expected {COMMIT}. Existing files were preserved.")
    branch = git("symbolic-ref", "--quiet", "--short", "HEAD", check=False).stdout.strip()
    if branch != BRANCH:
        raise ValueError(f"Existing checkout must already use branch {BRANCH}; preserving current branch {branch!r}")

    reverse = git("apply", "--reverse", "--check", str(patch), check=False)
    if reverse.returncode == 0:
        status = "already_patched"
    else:
        applicable = git("apply", "--check", "--whitespace=error", str(patch), check=False)
        if applicable.returncode:
            raise ValueError("Existing files do not match either the original or patched source; preserved without changes")
        if check_only:
            status = "would_apply_patch"
        else:
            git("apply", "--whitespace=error", str(patch))
            git("apply", "--reverse", "--check", str(patch))
            status = "patched_existing" if existed else "cloned_and_patched"
    return {"status": status, "checkout": str(checkout.resolve()), "source": ORIGIN,
            "commit": head, "branch": branch, "patch_sha256": PATCH_SHA256,
            "check_only": check_only}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=PROJECT / "third_party" / "OpenRSI")
    parser.add_argument("--check-only", action="store_true", help="Validate/describe without writing or downloading")
    args = parser.parse_args()
    try:
        report = bootstrap(args.checkout, check_only=args.check_only)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"bootstrap failed: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
