"""Security boundaries tested with trusted adversarial fixtures, never model code."""

import os
from pathlib import Path
import tempfile
import pytest

from commerce_rsi.openrsi_adapter.sandbox import SandboxError, SandboxLimits, run_sampler


def execute(body, **kwargs):
    source = "def sample_negatives(batch, candidate_view, model_view, rng, num_negatives):\n"
    source += "\n".join("    " + line for line in body.splitlines())
    return run_sampler(source, [{"positive_id": "p"}], ["p", "a", "b"], None, 7, 2, **kwargs)


def test_sampler_is_deterministic_and_can_use_stdlib():
    body = "import math, random\nreturn [rng.sample([x for x in candidate_view if x != row['positive_id']], num_negatives) for row in batch]"
    first = execute(body)
    assert first == execute(body)
    assert set(first[0]) == {"a", "b"}
    assert execute("return [random.random(), hash('repeatable')]" ) == execute(
        "return [random.random(), hash('repeatable')]"
    )


@pytest.mark.parametrize("operation", ["read", "write", "truncate", "unlink"])
def test_private_file_and_trainer_are_protected(tmp_path, operation):
    target = tmp_path / "private_answers_or_trainer.py"
    target.write_text("protected data")
    bodies = {
        "read": f"return open({str(target)!r}).read()",
        "write": f"return open({str(target)!r}, 'w').write('corrupt')",
        "truncate": f"import os\nos.truncate({str(target)!r}, 0)\nreturn []",
        "unlink": f"import os\nos.unlink({str(target)!r})\nreturn []",
    }
    with pytest.raises(SandboxError, match="PermissionError"):
        execute(bodies[operation])
    assert target.read_text() == "protected data"


def test_accessible_host_files_are_denied_by_kernel_sandbox():
    # Avoid relying only on /root or pytest's mode-0700 directories: these files
    # would be readable/writable by the dropped UID without the kernel policy.
    with tempfile.TemporaryDirectory(prefix="commerce-rsi-isolation-", dir="/tmp") as directory:
        os.chmod(directory, 0o777)
        target = Path(directory) / "world_accessible_fixture"
        target.write_text("must remain private to the trainer")
        target.chmod(0o666)
        for body in (f"return open({str(target)!r}).read()",
                     f"return open({str(target)!r}, 'w').write('corrupt')"):
            with pytest.raises(SandboxError, match="PermissionError"):
                execute(body)
        assert target.read_text() == "must remain private to the trainer"


def test_network_and_child_processes_are_blocked():
    # ctypes is already imported by the trusted worker. Direct syscalls exercise
    # the kernel policy even if ordinary import attempts would also be denied.
    for body in (
        "import ctypes\nc = ctypes.CDLL(None, use_errno=True)\nr = c.syscall(41, 2, 1, 0)\nreturn [r, ctypes.get_errno()]",
        "import ctypes\nc = ctypes.CDLL(None, use_errno=True)\nr = c.syscall(57)\nreturn [r, ctypes.get_errno()]",
        "import ctypes\nc = ctypes.CDLL(None, use_errno=True)\nr = c.syscall(62, 1, 0)\nreturn [r, ctypes.get_errno()]",
    ):
        assert execute(body) == [-1, 1]


def test_subprocess_exec_is_blocked():
    with pytest.raises(SandboxError, match="PermissionError"):
        execute("import subprocess\nreturn subprocess.run(['/bin/true']).returncode")


def test_wall_timeout_and_output_limit():
    with pytest.raises(SandboxError, match="timeout"):
        execute("while True:\n    pass", limits=SandboxLimits(wall_seconds=0.3))
    with pytest.raises(SandboxError, match="output_limit"):
        execute("print('x' * 100000)\nreturn []", limits=SandboxLimits(output_bytes=1024))


def test_invalid_candidate_never_silently_falls_back():
    with pytest.raises(SandboxError, match="ZeroDivisionError"):
        execute("return 1 / 0")
    with pytest.raises(SandboxError, match="invalid_output"):
        execute("return {1, 2}")


def test_credentials_and_parent_descriptors_not_exposed(monkeypatch, tmp_path):
    monkeypatch.setenv("COMMERCE_RSI_TEST_SECRET", "hidden")
    secret = tmp_path / "open_private"
    secret.write_text("hidden answer")
    with secret.open() as parent_file:
        os.set_inheritable(parent_file.fileno(), True)
        result = execute("import os\nreturn dict(os.environ)")
        assert "COMMERCE_RSI_TEST_SECRET" not in result
        body = f"import os\nreturn os.read({max(3, parent_file.fileno())}, 10).decode()"
        with pytest.raises(SandboxError):
            execute(body)


def test_trainer_bridge_whitelists_data_and_returns_checked_shape():
    import numpy as np
    from commerce_rsi.openrsi_adapter.sandbox import SandboxedSampler
    from commerce_rsi.sampling.core import CandidateView, validate_negatives

    view = CandidateView.from_rows([
        {"item_id": item_id, "text": "x" * 300, "category": "public"}
        for item_id in ("p", "a", "b")
    ])
    batch = [{"qid": "q", "query": "request", "item_id": "p", "ori_review": "forbidden"}]
    source = '''def sample_negatives(batch, candidate_view, model_view, rng, num_negatives):
    assert "ori_review" not in batch[0]
    assert "hidden_answers" not in model_view
    assert len(candidate_view["texts"][0]) == 256
    return [rng.sample([x for x in candidate_view["item_ids"] if x not in set(row["known_positive_ids"]) | {row["item_id"]}], num_negatives) for row in batch]
'''
    known = {"q": frozenset({"p"})}
    sampler = SandboxedSampler(source, known)
    result = sampler.sample_negatives(batch, view, {"step": 0, "hidden_answers": "secret"},
                                      np.random.default_rng(12), 2)
    assert validate_negatives(result, batch, view, 2, known) == result
