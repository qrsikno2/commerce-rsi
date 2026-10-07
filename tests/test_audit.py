import csv
import json
from pathlib import Path

import pytest

from commerce_rsi.data.audit import (
    audit_dataset,
    detect_near_duplicates,
    find_near_duplicates,
    normalize_query,
    product_text,
    query_components,
    read_safe_queries,
)


def write_queries(path, rows, header=None):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header or ["qid", "query", "item_id", "user_id", "ori_rating", "ori_review"])
        writer.writerows(rows)


def test_normalization_and_public_text_whitelist():
    assert normalize_query("  ＵＳＢ–C_Cable! ") == "usb c cable"
    assert normalize_query("Straße") == "strasse"
    assert product_text({"title": "A", "description": ["B", {"ori_review": "SECRET"}],
                         "ori_review": "SECRET", "ori_rating": 98765}) == "A\nB"
    assert product_text(["unsupported"]) == ""


def test_components_transitive_shared_user_target_and_normalized_query():
    rows = [
        {"qid": "1", "user_id": "u1", "item_id": "i1", "query": "red SHOES!"},
        {"qid": "2", "user_id": "u1", "item_id": "i2", "query": "boots"},
        {"qid": "3", "user_id": "u3", "item_id": "i2", "query": "bag"},
        {"qid": "4", "user_id": "u4", "item_id": "i4", "query": "Red shoes"},
        {"qid": "5", "user_id": "", "item_id": "", "query": "other"},
        {"qid": "6", "user_id": "", "item_id": "", "query": "distinct"},
    ]
    assert query_components(rows) == [[0, 1, 2, 3], [4], [5]]
    assert query_components(rows, near_pairs=[["2", "5"]]) == [[0, 1, 2, 3, 4], [5]]
    with pytest.raises(ValueError, match="unknown qid"):
        query_components(rows, near_pairs=[["2", "missing"]])
    assert query_components([]) == []


def test_audit_actual_linkage_anomalies_and_no_review_export(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    secret = "PRIVATE_REVIEW_SENTINEL_863693"
    write_queries(raw / "test.csv", [
        ["q1", "Red Shoes!", "p1", "u1", "PRIVATE_RATING", secret],
        ["q1", "red shoes", "p2", "u2", "PRIVATE_RATING", secret],
        ["q3", "a different query", "p2", "u3", "PRIVATE_RATING", secret],
        ["q4", "missing target", "absent", "u3", "PRIVATE_RATING", secret],
        ["q5", "no product text", "p3", "u5", "PRIVATE_RATING", secret],
        ["q6", "", "p1", "u1", "PRIVATE_RATING", secret],
        ["q7", "truncated row", "p1"],
    ])
    products = [
        {"item_id": "p1", "category": "Shoes", "metadata": {"title": "shoe", "ori_review": secret}},
        {"item_id": "p2", "category": "Shoes", "metadata": "boot"},
        {"item_id": "p3", "category": "Other", "metadata": {"ori_review": secret}},
        {"item_id": "p1", "category": "ignored", "metadata": secret},
        {"category": "Other", "metadata": "missing identifier"},
        ["not", "an", "object"],
    ]
    (raw / "sampled_item_metadata_1M.jsonl").write_text(
        "\n".join(json.dumps(product) for product in products) + '\n{"broken":\n', encoding="utf-8")
    destination = tmp_path / "report"
    report = audit_dataset(raw, destination)
    assert report["queries"]["complete_rows"] == 5
    assert report["queries"]["incomplete_rows"] == 1
    assert report["queries"]["malformed_records"] == 1
    assert report["queries"]["qid_repetition"]["excess_rows"] == 1
    assert report["queries"]["normalized_query_repetition"]["excess_rows"] == 1
    assert report["queries"]["target_repetition"]["repeated_keys"] == 1
    assert report["queries"]["user_repetition"]["repeated_keys"] == 1
    assert report["catalog"]["duplicate_item_id_records"] == 1
    assert report["catalog"]["malformed_records"] == 2
    assert report["catalog"]["target_coverage"]["missing_unique_targets"] == 1
    assert report["catalog"]["target_coverage"]["present_targets_without_text"] == 1
    assert report["catalog"]["target_coverage"]["fraction_query_rows_with_usable_target"] == pytest.approx(0.6)
    assert report["catalog"]["category_distribution_query_targets"] == {"Other": 1, "Shoes": 3}
    assert report["grouping_feasibility"]["largest_component_rows"] == 4
    assert report["grouping_feasibility"]["giant_component_warning"]
    serialized = (destination / "audit.json").read_text()
    assert secret not in serialized
    assert "PRIVATE_RATING" not in serialized
    assert "Red Shoes" not in serialized
    assert not list(destination.glob("catalog-audit-*"))
    assert json.loads(serialized) == report


def test_missing_headers_terminal_csv_and_safe_records(tmp_path):
    path = tmp_path / "queries.csv"
    write_queries(path, [["q", "thing", "p", "u"]], header=["qid", "query", "item_id", "user_id"])
    rows, report = read_safe_queries(path)
    assert set(rows[0]) == {"qid", "query", "item_id", "user_id"}
    assert report["missing_header_fields"] == ["ori_rating", "ori_review"]
    path.write_text('qid,query,item_id,user_id\nq,"unterminated,p,u\n')
    rows, report = read_safe_queries(path)
    assert rows == []
    assert not report["parse_complete"]
    assert report["malformed_records"] == 1


def test_near_duplicate_pairs_are_verified_deterministic_and_bounded():
    base = " ".join(f"term{index}" for index in range(80))
    rows = [
        {"qid": "a", "query": base},
        {"qid": "a2", "query": base.upper()},
        {"qid": "b", "query": base + " extra"},
        {"qid": "c", "query": "independent completely different words"},
    ]
    result = detect_near_duplicates(rows)
    assert result["pairs"] == [["a", "b"]]
    assert result == detect_near_duplicates(rows)
    assert result["unique_normalized_queries"] == 3
    duplicates = [{"qid": str(index), "query": base + f" tail{index}"} for index in range(30)]
    bounded = find_near_duplicates(duplicates, max_candidates=3)
    assert bounded["candidate_pairs_checked"] == 3
    assert bounded["candidate_budget_exhausted"]
    capped = find_near_duplicates(duplicates, bucket_limit=1)
    assert capped["capped_bucket_insertions"] > 0
    assert find_near_duplicates([])["verified_representative_pairs"] == 0


def test_giant_component_is_preserved(tmp_path):
    rows = [{"qid": str(index), "query": f"query {index}", "item_id": f"p{index}", "user_id": "one_user"}
            for index in range(1000)]
    assert query_components(rows) == [list(range(1000))]
