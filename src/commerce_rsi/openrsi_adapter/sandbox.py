"""Linux-only, fail-closed isolation for JSON-in/JSON-out sampler functions.

Generated source runs only after a new process has dropped privileges and enabled
Landlock and a seccomp syscall allowlist. The interpreter imports its dependencies
before denying all filesystem access. This is intentionally a CPU/std-lib worker:
the trusted trainer owns model weights, labels, validation, and output checking.
"""

from __future__ import annotations

import ctypes
import errno
import json
import math
import os
import platform
import random
import resource
import selectors
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any


class SandboxError(RuntimeError):
    """A failed candidate or unavailable isolation; callers must not fall back."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class SandboxLimits:
    wall_seconds: float = 5.0
    cpu_seconds: int = 2
    memory_mb: int = 256
    input_bytes: int = 16 * 1024 * 1024
    output_bytes: int = 1024 * 1024
    source_bytes: int = 65536


class SandboxedSampler:
    """Bridge the trusted trainer's views to a fresh isolated JSON worker.

    Public product texts are truncated to 256 characters for bounded transfer;
    every catalog ID remains available. This view is identical across candidate
    strategies. Hidden query labels and trainer/model objects are never sent.
    """

    def __init__(self, source: str, known_positives=None, *, limits: SandboxLimits | None = None):
        self.source = source
        self.known_positives = {str(key): tuple(sorted(value))
                                for key, value in (known_positives or {}).items()}
        self.limits = limits or SandboxLimits(input_bytes=32 * 1024 * 1024, memory_mb=512)
        self._candidate_object = None
        self._candidate_payload = None

    def sample_negatives(self, batch, candidate_view, model_view, rng, num_negatives):
        if self._candidate_object is not candidate_view:
            self._candidate_payload = {
                "item_ids": list(candidate_view.item_ids),
                "texts": [text[:256] for text in candidate_view.texts],
                "categories": [category[:128] for category in candidate_view.categories],
            }
            self._candidate_object = candidate_view
        public_batch = [{"qid": row["qid"], "query": row["query"], "item_id": row["item_id"],
                         "known_positive_ids": sorted(set(self.known_positives.get(row["qid"], ()))
                                                      | set(row.get("known_positive_item_ids", ())))}
                        for row in batch]
        public_model = {}
        if model_view is not None:
            for key in ("step", "total_steps", "progress"):
                value = model_view.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    public_model[key] = value
        seed = int(rng.integers(0, 2**32))
        return run_sampler(self.source, public_batch, self._candidate_payload, public_model,
                           seed, num_negatives, limits=self.limits)


def _validate_limits(limits: SandboxLimits) -> None:
    if not 0 < limits.wall_seconds <= 60 or not math.isfinite(limits.wall_seconds):
        raise ValueError("wall_seconds must be finite and in (0, 60]")
    if not 1 <= limits.cpu_seconds <= 30 or not 64 <= limits.memory_mb <= 4096:
        raise ValueError("cpu_seconds or memory_mb outside allowed limits")
    if not all(0 < value <= 64 * 1024 * 1024 for value in (
        limits.input_bytes, limits.output_bytes, limits.source_bytes,
    )):
        raise ValueError("IO and source limits must be in (0, 64 MiB]")


def run_sampler(
    source: str,
    batch: Any,
    candidate_view: Any,
    model_view: Any,
    seed: int,
    num_negatives: int,
    *,
    limits: SandboxLimits | None = None,
) -> Any:
    """Execute sample_negatives in an isolated worker and return its JSON value.

    Inputs must contain only the public/train whitelist. Validate output IDs,
    positive exclusion, shape, and duplicates in the trusted trainer afterwards.
    Nothing is executed in the parent, including candidate module-level code.
    """
    limits = limits or SandboxLimits()
    _validate_limits(limits)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SandboxError("isolation_unavailable", "requires Linux x86_64")
    if not isinstance(source, str) or len(source.encode()) > limits.source_bytes:
        raise SandboxError("source_limit")
    if not isinstance(seed, int) or not isinstance(num_negatives, int) or num_negatives < 1:
        raise ValueError("seed and a positive num_negatives must be integers")
    try:
        request = json.dumps({
            "source": source, "batch": batch, "candidate_view": candidate_view,
            "model_view": model_view, "seed": seed, "num_negatives": num_negatives,
            "limits": asdict(limits),
        }, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise SandboxError("invalid_input", str(exc)) from exc
    if len(request) > limits.input_bytes:
        raise SandboxError("input_limit")
    # A clean environment, -P (safe path), -s and -S exclude user/project startup
    # code, while retaining the explicit PYTHONHASHSEED. -I would ignore that seed.
    # The worker source is trusted; generated code is carried only through stdin.
    process = subprocess.Popen(
        [sys.executable, "-P", "-s", "-S", os.path.abspath(__file__), "--worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={"LANG": "C.UTF-8", "PYTHONHASHSEED": str(seed % (2**32))},
        close_fds=True, start_new_session=True,
    )
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    output = bytearray()
    errors = bytearray()
    offset = 0
    deadline = time.monotonic() + limits.wall_seconds
    try:
        with selectors.DefaultSelector() as selector:
            for stream, mode in ((process.stdin, selectors.EVENT_WRITE),
                                 (process.stdout, selectors.EVENT_READ),
                                 (process.stderr, selectors.EVENT_READ)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, mode)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SandboxError("timeout", f"wall limit {limits.wall_seconds}s")
                for key, _ in selector.select(min(remaining, 0.1)):
                    stream = key.fileobj
                    if stream is process.stdin:
                        try:
                            offset += os.write(stream.fileno(), request[offset:offset + 65536])
                        except BrokenPipeError:
                            offset = len(request)
                        if offset == len(request):
                            selector.unregister(stream)
                            stream.close()
                    else:
                        chunk = os.read(stream.fileno(), 65536)
                        if not chunk:
                            selector.unregister(stream)
                            stream.close()
                        else:
                            (output if stream is process.stdout else errors).extend(chunk)
                            if len(output) + len(errors) > limits.output_bytes:
                                raise SandboxError("output_limit")
            try:
                returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as exc:
                raise SandboxError("timeout") from exc
        if returncode != 0:
            reason = "timeout" if returncode in (-signal.SIGXCPU, -signal.SIGKILL) else "worker_failed"
            raise SandboxError(reason, errors.decode(errors="replace")[-2000:])
        try:
            response = json.loads(output)
        except (ValueError, UnicodeError) as exc:
            raise SandboxError("invalid_output", "worker did not return one JSON object") from exc
        if not isinstance(response, dict) or "ok" not in response:
            raise SandboxError("invalid_output")
        if not response["ok"]:
            raise SandboxError(response.get("reason", "candidate_failed"), str(response.get("detail", "")))
        return response["result"]
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            if not stream.closed:
                stream.close()


class _LandlockRuleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]


def _restrict_process(limits: SandboxLimits) -> None:
    """Install irreversible restrictions before interpreting candidate source."""
    if platform.machine() != "x86_64":
        raise RuntimeError("seccomp filter supports only x86_64")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.restype = ctypes.c_int
    libc.syscall.restype = ctypes.c_long

    def check(value: int, operation: str) -> int:
        if value < 0:
            number = ctypes.get_errno()
            raise OSError(number, operation + ": " + os.strerror(number))
        return value

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (limits.memory_mb * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    os.chdir("/")
    # Each worker starts by exec, so it never inherits the trainer's address space.
    # If launched as non-root it keeps that UID, with the same kernel restrictions.
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    check(libc.prctl(4, 0, 0, 0, 0), "PR_SET_DUMPABLE")
    check(libc.prctl(38, 1, 0, 0, 0), "PR_SET_NO_NEW_PRIVS")
    abi = check(libc.syscall(444, 0, 0, 1), "landlock ABI query")
    if abi < 1:
        raise RuntimeError("Landlock ABI 1 or newer is required")
    # ABI 1 controls all of these operations. No allow rules are added: filesystem
    # data access is denied globally. Seccomp additionally blocks ABI-1 truncate,
    # metadata mutation, networking, processes, tracing, and privilege operations.
    ruleset = _LandlockRuleset((1 << 13) - 1)
    ruleset_fd = check(libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0),
                       "landlock create ruleset")
    try:
        check(libc.syscall(446, ruleset_fd, 0), "landlock restrict self")
    finally:
        os.close(ruleset_fd)

    # Minimal CPython runtime operations after imports/JSON input have completed.
    # open/openat, socket, clone/fork, execve, kill, ptrace, ioctl, io_uring and all
    # path-mutating syscalls are absent. x32/other architectures cannot bypass it.
    allowed_syscalls = (
        0, 1, 3, 5, 8,       # read, write, close, fstat, lseek (inherited pipes only)
        9, 10, 11, 12, 25, 28,  # mmap, mprotect, munmap, brk, mremap, madvise
        13, 14, 15, 24,       # rt_sigaction/procmask/return, sched_yield
        35, 39, 60, 63,       # nanosleep, getpid, exit, uname
        96, 97, 98, 99,       # gettimeofday, getrlimit, getrusage, sysinfo
        102, 104, 107, 108, 110,  # uid/gid/euid/egid/ppid
        131, 186, 202, 204,   # sigaltstack, gettid, futex, sched_getaffinity
        218, 228, 229, 230, 231,  # clear_tid, clocks, clock_nanosleep, exit_group
        273, 334,            # robust_list, rseq (entropy sources are denied)
    )
    instructions = [
        _SockFilter(0x20, 0, 0, 4),             # load seccomp_data.arch
        _SockFilter(0x15, 1, 0, 0xC000003E),    # AUDIT_ARCH_X86_64
        _SockFilter(0x06, 0, 0, 0x80000000),    # kill process on wrong arch
        _SockFilter(0x20, 0, 0, 0),             # load syscall number
    ]
    for syscall in allowed_syscalls:
        instructions += [_SockFilter(0x15, 0, 1, syscall),
                         _SockFilter(0x06, 0, 0, 0x7FFF0000)]
    instructions.append(_SockFilter(0x06, 0, 0, 0x00050000 | errno.EPERM))
    instruction_array = (_SockFilter * len(instructions))(*instructions)
    program = _SockFprog(len(instructions), instruction_array)
    check(libc.prctl(22, 2, ctypes.byref(program), 0, 0), "PR_SET_SECCOMP")


def _worker() -> None:
    response: dict[str, Any]
    try:
        # The parent is trusted and validates this message before launching us.
        request = json.loads(sys.stdin.buffer.read(64 * 1024 * 1024 + 1))
        limits = SandboxLimits(**request["limits"])
        _validate_limits(limits)
        random.seed(request["seed"])
        try:
            _restrict_process(limits)
        except BaseException as exc:
            raise SandboxError("isolation_unavailable", f"{type(exc).__name__}: {exc}") from exc
        namespace: dict[str, Any] = {"math": math, "random": random}
        exec(compile(request["source"], "<sampler>", "exec"), namespace)
        sampler = namespace.get("sample_negatives")
        if not callable(sampler):
            raise SandboxError("invalid_interface", "sample_negatives is not callable")
        result = sampler(request["batch"], request["candidate_view"], request["model_view"],
                         random.Random(request["seed"]), request["num_negatives"])
        response = {"ok": True, "result": result}
    except BaseException as exc:
        response = {"ok": False,
                    "reason": exc.reason if isinstance(exc, SandboxError) else "candidate_failed",
                    "detail": f"{type(exc).__name__}: {exc}"[:2000]}
    # JSON-only serialization prevents object deserialization in the parent.
    try:
        encoded = json.dumps(response, allow_nan=False, separators=(",", ":"))
    except BaseException as exc:
        encoded = json.dumps({"ok": False, "reason": "invalid_output", "detail": type(exc).__name__})
    sys.stdout.write(encoded)
    sys.stdout.flush()


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("This module is an internal isolated sampler worker")
    _worker()
