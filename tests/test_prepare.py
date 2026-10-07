"""Protocol invariants on synthetic fixtures; these are not experiment results."""

import csv
import json
from collections import defaultdict

import pytest

from commerce_rsi.data.audit import normalize_query
from commerce_rsi.data.download import sha256_file
from commerce_rsi.data.prepare import prepare_dataset


@pytest.fixture
def source(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    queries, products = [], {}
    for category in ("Kitchen", "Books", "Tools"):
        for index in range(12):
            item_id = f"{category}-p{0 if index == 1 else index}"
            user_id = f"{category}-u{2 if index == 3 else index}"
            query = f"{category} needs {4 if index == 5 else index} useful special item description"
            if index == 5:
                query = query.upper() + "!"
            queries.append({"qid": f"{category}-q{index}", "query": query,
                            "item_id": item_id, "user_id": user_id,
                            "ori_review": "FORBIDDEN_REVIEW_VALUE", "ori_rating": "FORBIDDEN_RATING_VALUE"})
            products[item_id] = {"item_id": item_id, "category": category,
                                 "metadata": {"title": f"{category} product {index}",
                                              "description": "Public description",
                                              "ori_review": "FORBIDDEN_PRODUCT_REVIEW_VALUE"}}
    # A component spanning two categories must be quarantined intact.
    for category in ("Kitchen", "Books"):
        item_id = f"{category}-cross"
        queries.append({"qid": f"{category}-cross", "query": f"cross-category {category} requirement",
                        "item_id": item_id, "user_id": "cross-category-user",
                        "ori_review": "FORBIDDEN_REVIEW_VALUE", "ori_rating": "FORBIDDEN_RATING_VALUE"})
        products[item_id] = {"item_id": item_id, "category": category, "metadata": f"{category} cross product"}
    for index in range(15):
        item_id = f"distractor-{index}"
        products[item_id] = {"item_id": item_id, "category": "Distractors", "metadata": f"distractor text {index}"}
    with (raw / "test.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["qid", "query", "item_id", "user_id", "ori_rating", "ori_review"])
        writer.writeheader()
        writer.writerows(queries)
    (raw / "sampled_item_metadata_1M.jsonl").write_text("".join(json.dumps(row) + "\n" for row in products.values()))
    manifest = {"revision": "synthetic-test-fixture", "files": [
        {"name": filename, "sha256": sha256_file(raw / filename)}
        for filename in ("test.csv", "sampled_item_metadata_1M.jsonl")
    ]}
    (raw / "source_manifest.json").write_text(json.dumps(manifest))
    return raw, queries


def test_both_split_levels_isolate_users_targets_queries_and_keep_whole_components(source, tmp_path):
    raw, source_queries = source
    output = tmp_path / "prepared"
    manifest = prepare_dataset(raw, output, candidate_count=40, minimum_task_queries=6)
    assert {task["outer_split"] for task in manifest["tasks"]} == {"meta_train", "meta_dev", "meta_test"}
    assert manifest["excluded_query_counts"]["cross_category_component_queries"] == 2
    assert manifest["usable_query_count"] == 36
    assignments = [json.loads(line) for line in (output / "private" / "split_assignments.jsonl").read_text().splitlines()]
    queries = {row["qid"]: row for row in source_queries}
    owners = defaultdict(set)
    by_qid = {}
    for assignment in assignments:
        owner = (assignment["outer_split"], assignment["task_id"], assignment["inner_split"])
        row = queries[assignment["qid"]]
        by_qid[row["qid"]] = owner
        for kind, value in (("target", row["item_id"]), ("user", row["user_id"]),
                            ("query", normalize_query(row["query"]))):
            owners[kind, value].add(owner)
    assert all(len(owner_set) == 1 for owner_set in owners.values())
    assert len(by_qid) == 36
    for category in ("Kitchen", "Books", "Tools"):
        assert by_qid[f"{category}-q0"] == by_qid[f"{category}-q1"]
        assert by_qid[f"{category}-q2"] == by_qid[f"{category}-q3"]
        assert by_qid[f"{category}-q4"] == by_qid[f"{category}-q5"]
    assert "Kitchen-cross" not in by_qid and "Books-cross" not in by_qid


def test_private_placement_and_export_field_whitelists(source, tmp_path):
    raw, _ = source
    output = tmp_path / "prepared"
    manifest = prepare_dataset(raw, output, candidate_count=40, minimum_task_queries=6)
    for task in manifest["tasks"]:
        entries = [entry for entry in manifest["query_files"] if task["task_id"] in entry["path"]]
        assert len(entries) == 3
        for entry in entries:
            relative = entry["path"]
            if task["outer_split"] == "meta_test":
                assert relative.startswith("private/meta_test/")
            elif relative.endswith("final_evaluation.jsonl"):
                assert relative.startswith("private/final_evaluation/")
            else:
                assert relative.startswith("tasks/")
            for line in (output / relative).read_text().splitlines():
                assert set(json.loads(line)) == {"qid", "query", "item_id"}
    public_catalog = output / "public" / "catalog.jsonl"
    for line in public_catalog.read_text().splitlines():
        assert set(json.loads(line)) == {"item_id", "category", "text"}
    for path in output.rglob("*.json*"):
        assert "FORBIDDEN_" not in path.read_text()
    assert manifest["catalog_sha256"] == sha256_file(public_catalog)


def test_pool_is_fixed_for_same_seed_and_protocol_cannot_be_overwritten(source, tmp_path):
    raw, _ = source
    first = prepare_dataset(raw, tmp_path / "first", candidate_count=40, seed=19, minimum_task_queries=6)
    second = prepare_dataset(raw, tmp_path / "second", candidate_count=40, seed=19, minimum_task_queries=6)
    assert first["catalog_sha256"] == second["catalog_sha256"]
    assert first["tasks"] == second["tasks"]
    assert first["query_files"] == second["query_files"]
    with pytest.raises(ValueError, match="overwrite"):
        prepare_dataset(raw, tmp_path / "first", candidate_count=40, seed=19, minimum_task_queries=6)


def test_too_small_candidate_pool_fails(source, tmp_path):
    raw, _ = source
    with pytest.raises(ValueError, match="smaller than"):
        prepare_dataset(raw, tmp_path / "small", candidate_count=1, minimum_task_queries=6)


def test_source_tampering_fails_before_preparation(source, tmp_path):
    raw, _ = source
    query_path = raw / "test.csv"
    query_path.write_text(query_path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed since acquisition"):
        prepare_dataset(raw, tmp_path / "tampered", candidate_count=40, minimum_task_queries=6)
