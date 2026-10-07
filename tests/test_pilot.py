"""Controller contracts with fake models/rewards; no capability claims or GPU use."""

import json
from pathlib import Path

import pytest

from commerce_rsi.openrsi_adapter.upstream import SearchRequest
from commerce_rsi.recursion import pilot


SOURCE = "def sample_negatives(batch, candidate_view, model_view, rng, num_negatives):\n    return [['negative']]\n"


class FakeSearch:
    def __init__(self, *args, **kwargs):
        self.recorded = []

    def next_prompt(self, task_id, task_name, **kwargs):
        return SearchRequest(task_id=task_id, task_name=task_name, operator="draft",
                             messages=[{"role": "system", "content": "trusted contract"},
                                       {"role": "user", "content": "task description"}],
                             parent_ids=(), selection_metadata={"operator": "draft"})

    def record(self, request, source, **kwargs):
        self.recorded.append({"request": request, "source": source, **kwargs})
        return len(self.recorded)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def fixture_record(role="meta_train"):
    return {"metadata": {"uuid": "task-uuid", "task_name": "test_task", "outer_split": role,
                         "task_description": "Test fixture only", "data_description": "Training only",
                         "data_dir": "/unused/prepared/tasks/test_task",
                         "prepared_manifest_sha256": "manifest-hash"}}


def test_source_extraction_does_not_rewrite_program_or_accept_multiple_programs():
    assert pilot.extract_source("```python\n" + SOURCE + "```\n") == SOURCE
    assert pilot.extract_source(SOURCE) == SOURCE
    with pytest.raises(ValueError, match="one complete"):
        pilot.extract_source("```python\n" + SOURCE + "```\n```python\npass\n```")
    with pytest.raises(ValueError, match="Missing"):
        pilot.extract_source("print('no sampler')")


def test_group_failures_have_zero_reward_and_truncated_program_is_never_executed(monkeypatch, tmp_path):
    completions = [
        {"index": 0, "text": "```python\n" + SOURCE + "```", "finish_reason": "eos", "token_ids": [1, 2]},
        {"index": 1, "text": "this is not valid python!!!", "finish_reason": "eos", "token_ids": [3, 4]},
        {"index": 2, "text": SOURCE, "finish_reason": "length", "token_ids": [5, 6]},
        {"index": 3, "text": SOURCE.replace("'negative'", "'raises'"), "finish_reason": "eos", "token_ids": [7, 8]},
    ]
    group = {"checkpoint_id": "A0", "completions": completions,
             "cost": {"completion_tokens": 8, "agent_generation_wall_seconds": 1}}
    released = []

    class FakeAgent:
        def generate_group(self, prompt, **kwargs):
            assert "trusted contract" in prompt
            assert kwargs["count"] == 4
            return group

        def release(self):
            released.append(True)

    monkeypatch.setattr(pilot, "load_agent", lambda *args, **kwargs: FakeAgent())
    business_calls = []

    def fake_business(prepared, task, strategy, output, **kwargs):
        business_calls.append(kwargs)
        assert strategy == "isolated" and task == "test_task"
        assert kwargs["agent_checkpoint"] == "A0"
        assert kwargs["config"].shared_towers
        if len(business_calls) == 2:
            Path(output).mkdir(parents=True)
            (Path(output) / "run.json").write_text(json.dumps({"cost": {"business_total_wall_seconds": 2}}))
            raise RuntimeError("sandbox execution failed")
        return {"metrics": {"mrr": 0.2, "mrr@10": 0.1},
                "cost": {"business_total_wall_seconds": 3}, "business_initialization_sha256": "init"}

    monkeypatch.setattr(pilot, "run_business", fake_business)
    search = FakeSearch()
    saved, report = pilot.run_group(fixture_record(), search, "unused-model", None, tmp_path / "group",
                                   generation_seed=17, model_seed=17, business_seed=17,
                                   business_steps=2, group_size=4, max_new_tokens=100)
    assert saved == group and released == [True]
    assert len(business_calls) == 2  # Neither syntax failure nor truncated code reaches business execution.
    assert report["rewards"] == [0.2, 0.0, 0.0, 0.0]
    assert [entry["score"] for entry in search.recorded] == [0.2, None, None, None]
    assert [entry["raw_response"] for entry in search.recorded] == [entry["text"] for entry in completions]
    assert report["results"][3]["business_cost"]["business_total_wall_seconds"] == 2
    assert (tmp_path / "group" / "candidate_0" / "sampler.py").read_text() == SOURCE
    assert not (tmp_path / "group" / "candidate_1").exists()
    assert not (tmp_path / "group" / "candidate_2").exists()


def setup_pilot(monkeypatch, tmp_path, role="meta_train"):
    package = tmp_path / "package.parquet"
    package.write_bytes(b"synthetic package fixture; reader mocked")
    monkeypatch.setattr(pilot, "read_task_package", lambda _: [fixture_record(role)])
    monkeypatch.setattr(pilot, "upstream_manifest", lambda: {"commit": "test-only"})
    monkeypatch.setattr(pilot, "SamplerSearch", FakeSearch)
    return package


@pytest.mark.parametrize("mode,role", [("recursive", "meta_dev"), ("fixed", "meta_dev"),
                                      ("evaluate", "meta_train")])
def test_role_rejection_occurs_before_generation_or_output(monkeypatch, tmp_path, mode, role):
    package = setup_pilot(monkeypatch, tmp_path, role)
    monkeypatch.setattr(pilot, "run_group", lambda *args, **kwargs: pytest.fail("must not generate"))
    with pytest.raises(ValueError, match="requires"):
        pilot.run_pilot(package, "model", tmp_path / "output", task_id="test_task", mode=mode)
    assert not (tmp_path / "output").exists()


def test_recursive_generations_only_advance_after_update_and_reuse_current_adapter(monkeypatch, tmp_path):
    package = setup_pilot(monkeypatch, tmp_path)
    adapters, update_inputs = [], []

    def fake_group(record, search, model_dir, adapter, output, **kwargs):
        adapters.append(adapter)
        group = {"checkpoint_id": str(adapter), "completions": [{}, {}]}
        report = {"checkpoint_id": str(adapter), "rewards": [0, 0] if len(adapters) == 1 else [0, 0.1],
                  "results": [], "generation_cost": {"completion_tokens": 2}}
        return group, report

    def fake_update(model_dir, adapter, group, rewards, output):
        assert group["checkpoint_id"] == str(adapter)
        update_inputs.append(adapter)
        return {"status": "no_update" if rewards[0] == rewards[1] else "updated",
                "parameters_changed": rewards[0] != rewards[1], "reload_verified": rewards[0] != rewards[1]}

    monkeypatch.setattr(pilot, "run_group", fake_group)
    monkeypatch.setattr(pilot, "update_from_group", fake_update)
    report = pilot.run_pilot(package, "model", tmp_path / "recursive", task_id="test_task",
                             rounds=2, max_attempts=3)
    first_updated = (tmp_path / "recursive" / "A1_attempt1").resolve()
    assert adapters == update_inputs == [None, None, first_updated]
    assert report["status"] == "completed"
    assert len(report["groups"]) == 3
    assert [entry["status"] for entry in report["lineage"]] == ["no_update", "updated", "updated"]
    assert report["final_adapter"] == str((tmp_path / "recursive" / "A2_attempt0").resolve())


@pytest.mark.parametrize("mode,role", [("fixed", "meta_train"), ("evaluate", "meta_dev")])
def test_frozen_modes_spend_exact_requested_groups_without_updating(monkeypatch, tmp_path, mode, role):
    package = setup_pilot(monkeypatch, tmp_path, role)
    adapters = []
    frozen = tmp_path / "frozen-adapter"

    def fake_group(record, search, model_dir, adapter, output, **kwargs):
        adapters.append(adapter)
        return {}, {"rewards": [0.0, 0.1], "results": [], "generation_cost": {}}

    monkeypatch.setattr(pilot, "run_group", fake_group)
    monkeypatch.setattr(pilot, "update_from_group", lambda *args: pytest.fail("frozen mode must not update"))
    report = pilot.run_pilot(package, "model", tmp_path / mode, task_id="test_task", mode=mode,
                             input_adapter=frozen, rounds=2, max_attempts=2)
    assert adapters == [frozen.resolve()] * 4
    assert len(report["groups"]) == 4 and report["lineage"] == []
    assert report["final_adapter"] == str(frozen.resolve())


def test_no_reward_difference_stops_at_budget_and_does_not_claim_generation(monkeypatch, tmp_path):
    package = setup_pilot(monkeypatch, tmp_path)
    calls = []

    def fake_group(*args, **kwargs):
        calls.append(True)
        return {}, {"rewards": [0.0, 0.0], "results": [], "generation_cost": {}}

    monkeypatch.setattr(pilot, "run_group", fake_group)
    monkeypatch.setattr(pilot, "update_from_group", lambda *args: {"status": "no_update", "parameters_changed": False})
    destination = tmp_path / "no-updates"
    with pytest.raises(RuntimeError, match="No differential execution rewards"):
        pilot.run_pilot(package, "model", destination, task_id="test_task", rounds=2, max_attempts=2)
    report = json.loads((destination / "pilot.json").read_text())
    assert len(calls) == 2
    assert report["status"] == "failed"
    assert "final_adapter" not in report
    assert all(entry["status"] == "no_update" for entry in report["lineage"])
