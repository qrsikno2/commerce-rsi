import json

import pytest

from commerce_rsi.openrsi_adapter.upstream import (
    SamplerSearch, UPSTREAM_COMMIT, export_task_package, read_task_package, upstream_manifest,
)


def test_actual_upstream_draft_improve_debug_and_crossover(tmp_path):
    with SamplerSearch(tmp_path / "programs.sqlite") as search:
        draft = search.next_prompt("id", "task")
        assert draft.operator == "draft"
        assert "sample_negatives" in draft.messages[0]["content"]
        first = search.record(draft, "def sample_negatives(*args): return []", score=0.2, success=True)
        search.search._choose_rl_mixed_mode = lambda: "improve"
        improved = search.next_prompt("id", "task")
        assert improved.operator == "improve" and improved.parent_ids == (first,)
        assert "0.2" in improved.messages[1]["content"]
        assert "reward_validation_mrr\"" in improved.messages[1]["content"]
        assert "untruncated reciprocal rank" in improved.messages[0]["content"]
        assert search.database.get_by_id(first).metadata["primary_metric"] == "mrr"
        second = search.record(improved, "improved", score=0.3, success=True)
        search.search._choose_rl_mixed_mode = lambda: "crossover"
        crossover = search.next_prompt("id", "task")
        assert crossover.operator == "crossover"
        assert set(crossover.parent_ids) == {first, second}
        bad = search.record(crossover, "bad", score=None, success=False, feedback="invalid IDs")
        search.search._choose_rl_mixed_mode = lambda: "debug"
        debug = search.next_prompt("id", "task")
        assert debug.operator == "debug" and debug.parent_ids == (bad,)
        assert "invalid IDs" in debug.messages[1]["content"]
        assert "submission.csv" not in debug.messages[0]["content"]
        assert search.database.get_by_id(bad).metadata["airaevo_experience_card"]["is_buggy"]


def test_independent_database_instances_and_reload(tmp_path):
    with SamplerSearch(tmp_path / "a.sqlite") as a, SamplerSearch(tmp_path / "b.sqlite") as b:
        request = a.next_prompt("id", "task")
        a.record(request, "example", score=0.0, success=True)
        assert a.database.size("task") == 1
        assert b.database.size("task") == 0
    with SamplerSearch(tmp_path / "a.sqlite") as reopened:
        assert reopened.database.size("task") == 1
        assert reopened.database.get_all("task")[0].score == 0.0


def test_recorded_upstream_version_includes_modified_file_hashes():
    manifest = upstream_manifest()
    assert manifest["commit"] == UPSTREAM_COMMIT
    assert all(len(value) == 64 for value in manifest["file_sha256"].values())


def _prepare_test_inputs(tmp_path):
    pytest.importorskip("pyarrow")
    tasks = [{"task_id": name, "category": "category " + name, "outer_split": split}
             for name, split in (("train_task", "meta_train"), ("dev_task", "meta_dev"))]
    for task in tasks:
        directory = tmp_path / "tasks" / task["task_id"]
        directory.mkdir(parents=True)
        (directory / "train.jsonl").write_text("")
        (directory / "reward_validation.jsonl").write_text("")
    manifest = {"protocol": "test-only", "tasks": tasks, "catalog_sha256": "a" * 64}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    train_path, dev_path = tmp_path / "train.parquet", tmp_path / "dev.parquet"
    export_task_package(tmp_path, train_path, "meta_train")
    export_task_package(tmp_path, dev_path, "meta_dev")
    return manifest_path, manifest, train_path, dev_path


def test_task_parquet_roundtrip_enforces_separate_inputs_and_manifest(tmp_path):
    manifest_path, manifest, train_path, dev_path = _prepare_test_inputs(tmp_path)
    train, dev = read_task_package(train_path), read_task_package(dev_path)
    assert len(train) == len(dev) == 1
    assert train[0]["metadata"]["uuid"] != dev[0]["metadata"]["uuid"]
    assert train[0]["metadata"]["task_name"] == "train_task"
    assert train[0]["metadata"]["primary_metric"] == "mrr"
    with pytest.raises(ValueError, match="only meta_train or meta_dev"):
        export_task_package(tmp_path, tmp_path / "test.parquet", "meta_test")
    manifest_path.write_text(json.dumps({**manifest, "modified": True}))
    with pytest.raises(ValueError, match="manifest changed"):
        read_task_package(train_path)


@pytest.mark.parametrize("field,value,message", [
    ("outer_split", "meta_train", "outer_split differs"),
    ("category", "forged category", "category differs"),
    ("catalog_sha256", "b" * 64, "catalog_sha256 differs"),
    ("protocol", "forged protocol", "protocol differs"),
    ("uuid", "forged UUID", "UUID differs"),
    ("task_name", "invented_task", "not uniquely present"),
    ("primary_metric", "mrr_at_10", "primary metric"),
    ("prompt_schema_version", 1, "prompt schema version"),
])
def test_task_metadata_cannot_override_authoritative_manifest(tmp_path, field, value, message):
    _, _, _, dev_path = _prepare_test_inputs(tmp_path)
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = pq.read_table(dev_path).to_pylist()
    records[0]["metadata"][field] = value
    if field == "task_name":
        records[0]["metadata"]["data_dir"] = str(tmp_path / "tasks" / value)
    tampered_path = tmp_path / "tampered.parquet"
    pq.write_table(pa.Table.from_pylist(records), tampered_path)
    with pytest.raises(ValueError, match=message):
        read_task_package(tampered_path)


def test_task_data_directory_cannot_escape_through_symlink(tmp_path):
    _, _, _, dev_path = _prepare_test_inputs(tmp_path)
    directory = tmp_path / "tasks" / "dev_task"
    outside = tmp_path / "outside_tasks"
    directory.rename(outside)
    directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="resolve exactly inside"):
        read_task_package(dev_path)
