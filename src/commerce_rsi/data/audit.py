"""Streaming Amazon-C4 audit; exported reports never contain source reviews/ratings.

Only query records and their bounded near-duplicate index are held in memory.
The million-item catalogue is indexed in a temporary SQLite database, not a
million-element Python dictionary. Reports deliberately contain no query text.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

SAFE_QUERY_FIELDS = ("qid", "query", "item_id", "user_id")
RAW_QUERY_FIELDS = SAFE_QUERY_FIELDS + ("ori_rating", "ori_review")
EXAMPLE_LIMIT = 20


def normalize_query(text: str) -> str:
    """NFKC + Unicode casefold + alphanumeric tokens; punctuation is a separator."""
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


def product_text(metadata: object) -> str:
    """Read only the documented public text, including a strict dictionary whitelist."""
    if isinstance(metadata, str):
        return metadata.strip()
    if not isinstance(metadata, dict):
        return ""
    pieces: list[str] = []
    for key in ("title", "description"):
        value = metadata.get(key)
        if isinstance(value, str):
            pieces.append(value)
        elif isinstance(value, list):
            pieces.extend(piece for piece in value if isinstance(piece, str))
    return "\n".join(pieces).strip()


def query_components(records: list[Mapping[str, str]],
                     near_pairs: Iterable[Iterable[str]] | None = None) -> list[list[int]]:
    """Connect row indices by shared user, target item, or normalized query.

    Empty values never form an edge. Callers must reject incomplete records before
    declaring a leakage-free split. Duplicate qids should separately be rejected.
    Optional verified near pairs are qid pairs and add further indivisible edges.
    """
    parent = list(range(len(records)))
    size = [1] * len(records)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            if size[left] < size[right]:
                left, right = right, left
            parent[right] = left
            size[left] += size[right]

    seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for field in ("user_id", "item_id", "query"):
            value = record.get(field, "")
            if field == "query":
                value = normalize_query(value)
            if not value:
                continue
            key = (field, value)
            previous = seen.setdefault(key, index)
            union(index, previous)
    if near_pairs is not None:
        qid_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            qid_to_indices[record["qid"]].append(index)
        for pair in near_pairs:
            qid_a, qid_b = pair
            if qid_a not in qid_to_indices or qid_b not in qid_to_indices:
                raise ValueError("Near-duplicate pair refers to an unknown qid")
            indices = qid_to_indices[qid_a] + qid_to_indices[qid_b]
            for index in indices[1:]:
                union(indices[0], index)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[find(index)].append(index)
    return sorted(groups.values(), key=lambda group: (-len(group), group[0]))


class _Lengths:
    def __init__(self) -> None:
        self.histogram: Counter[int] = Counter()

    def add(self, value: int) -> None:
        self.histogram[value] += 1

    def report(self) -> dict:
        count = self.histogram.total()
        if not count:
            return {"count": 0}
        ordered = sorted(self.histogram.items())
        quantiles = {}
        for name, fraction in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99)):
            rank = max(1, math.ceil(fraction * count))
            accumulated = 0
            for value, occurrences in ordered:
                accumulated += occurrences
                if accumulated >= rank:
                    quantiles[name] = value
                    break
        return {"count": count, "min": ordered[0][0], "max": ordered[-1][0],
                "mean": sum(value * frequency for value, frequency in ordered) / count,
                **quantiles}


def _example(examples: dict, kind: str, *, line: int | None = None, **ids: str) -> None:
    bucket = examples.setdefault(kind, [])
    if len(bucket) < EXAMPLE_LIMIT:
        record = {key: value for key, value in ids.items()
                  if key in ("qid", "item_id", "user_id") and isinstance(value, str)}
        if line is not None:
            record["line"] = line
        bucket.append(record)


def _repetition(counter: Counter[str], *, include_ids: bool = False) -> dict:
    repeated = [(value, count) for value, count in counter.items() if count > 1]
    report = {"unique": len(counter), "repeated_keys": len(repeated),
              "rows_in_repeated_keys": sum(count for _, count in repeated),
              "excess_rows": sum(count - 1 for _, count in repeated),
              "max_rows_per_key": max(counter.values(), default=0)}
    if include_ids:
        report["examples"] = [{"id": value, "count": count}
                              for value, count in sorted(repeated, key=lambda pair: (-pair[1], pair[0]))[:EXAMPLE_LIMIT]]
    return report


def read_safe_queries(path: Path) -> tuple[list[dict[str, str]], dict]:
    """Load only complete whitelisted records and return privacy-safe diagnostics.

    Forbidden columns are checked for header presence only; their values are not
    inspected, copied into records, or included in exceptions/reports.
    """
    csv.field_size_limit(32 * 1024 * 1024)
    rows: list[dict[str, str]] = []
    missing: Counter[str] = Counter()
    examples: dict = {}
    chars, words = _Lengths(), _Lengths()
    total = malformed = incomplete = 0
    parse_complete = True
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        header = reader.fieldnames or []
        missing_header = [field for field in RAW_QUERY_FIELDS if field not in header]
        duplicate_header = sorted(field for field, count in Counter(header).items() if count > 1)
        if duplicate_header:
            raise ValueError("Duplicate CSV header columns: input is ambiguous")
        try:
            for raw in reader:
                total += 1
                if None in raw or any(raw.get(field) is None for field in header):
                    malformed += 1
                    _example(examples, "malformed_csv_record", line=reader.line_num)
                    continue
                record = {field: (raw.get(field) or "").strip() for field in SAFE_QUERY_FIELDS}
                absent = [field for field, value in record.items() if not value]
                if absent:
                    incomplete += 1
                    missing.update(absent)
                    _example(examples, "missing_required_value", line=reader.line_num,
                             **{field: record[field] for field in ("qid", "item_id", "user_id")})
                    continue
                if not normalize_query(record["query"]):
                    incomplete += 1
                    missing["normalized_query"] += 1
                    _example(examples, "empty_normalized_query", line=reader.line_num, qid=record["qid"])
                    continue
                rows.append(record)
                chars.add(len(record["query"]))
                words.add(len(record["query"].split()))
                if len(record["query"]) > 10000:
                    _example(examples, "query_over_10000_characters", line=reader.line_num, qid=record["qid"])
        except csv.Error:
            malformed += 1
            parse_complete = False
            _example(examples, "terminal_csv_parse_error", line=reader.line_num)
    return rows, {"records_read": total, "complete_rows": len(rows), "incomplete_rows": incomplete,
                  "malformed_records": malformed, "parse_complete": parse_complete,
                  "missing_header_fields": missing_header, "missing_safe_values": dict(missing),
                  "forbidden_columns": {field: {"header_present": field in header, "values_inspected": False}
                                        for field in ("ori_review", "ori_rating")},
                  "length_characters": chars.report(), "length_whitespace_tokens": words.report(),
                  "abnormal_samples": examples}


def find_near_duplicates(records: list[Mapping[str, str]], *, threshold: float = 0.85,
                         max_candidates: int = 250000, bucket_limit: int = 256,
                         include_pairs: bool = False) -> dict:
    """Bounded deterministic MinHash LSH candidates verified by exact Jaccard.

    Exact normalized duplicates are collapsed first. Returned pairs contain only
    representative qids and similarity, never query text. This is an approximate
    audit, not a proof that all near duplicates are absent.
    """
    import numpy as np

    if not 0 < threshold <= 1 or max_candidates < 1 or bucket_limit < 1:
        raise ValueError("Invalid near-duplicate audit limits")
    representatives: dict[str, Mapping[str, str]] = {}
    for record in records:
        normalized = normalize_query(record["query"])
        if normalized:
            representatives.setdefault(normalized, record)
    prime = np.uint64(2147483647)
    generator = np.random.default_rng(20261007)
    multiplier = generator.integers(1, int(prime), size=64, dtype=np.uint64)
    offset = generator.integers(0, int(prime), size=64, dtype=np.uint64)
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    shingles: list[set[str]] = []
    qids: list[str] = []
    candidates_seen = verified = capped_insertions = 0
    examples: list[dict] = []
    pairs: list[list[str]] = []
    exhausted = False
    for index, (normalized, record) in enumerate(representatives.items()):
        tokens = normalized.split()
        grams = {" ".join(tokens[start:start + 3]) for start in range(max(1, len(tokens) - 2))}
        hashed = np.array([int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "little") % int(prime)
                           for gram in sorted(grams)], dtype=np.uint64)
        signature = ((multiplier[:, None] * hashed[None, :] + offset[:, None]) % prime).min(axis=1)
        keys = [(band, signature[band * 4:(band + 1) * 4].tobytes()) for band in range(16)]
        candidates = {previous for key in keys for previous in buckets.get(key, [])}
        for previous in sorted(candidates):
            if candidates_seen >= max_candidates:
                exhausted = True
                break
            candidates_seen += 1
            other = shingles[previous]
            similarity = len(grams & other) / len(grams | other)
            if similarity >= threshold:
                verified += 1
                if include_pairs:
                    pairs.append([qids[previous], record["qid"]])
                if len(examples) < EXAMPLE_LIMIT:
                    examples.append({"qid_a": qids[previous], "qid_b": record["qid"], "jaccard": similarity})
        shingles.append(grams)
        qids.append(record["qid"])
        for key in keys:
            if len(buckets[key]) < bucket_limit:
                buckets[key].append(index)
            else:
                capped_insertions += 1
        if exhausted:
            break
    report = {"method": "normalized word-trigram MinHash LSH; 64 permutations, 16 bands of 4; exact Jaccard verification",
            "normalization": "Unicode NFKC, casefold, alphanumeric tokens joined by spaces",
            "seed": 20261007, "threshold": threshold, "unique_normalized_queries": len(representatives),
            "unique_queries_scanned": len(shingles), "candidate_pairs_checked": candidates_seen,
            "verified_representative_pairs": verified, "max_candidate_pairs": max_candidates,
            "bucket_limit": bucket_limit, "capped_bucket_insertions": capped_insertions,
            "candidate_budget_exhausted": exhausted, "examples": examples,
            "limitations": ["Approximate candidates can miss near duplicates; counts are lower bounds.",
                            "Exact duplicate clusters use one representative; pair count is not expanded.",
                            "Bucket and global candidate caps bound cost and may further reduce recall.",
                            "Near duplicates are reported but are not edges in the required exact grouping."]}
    if include_pairs:
        report["pairs"] = pairs
    return report


def detect_near_duplicates(records: list[Mapping[str, str]], *, threshold: float = 0.9,
                           max_candidates: int = 250000, bucket_limit: int = 256) -> dict:
    """Return bounded qid pairs for conservative merging before data preparation.

    Pairs connect normalized-query representatives. Exact duplicate groups must
    also be merged using ``query_components``. A cap hit is an explicit audit
    limitation; absence of pairs is never a proof of semantic independence.
    """
    return find_near_duplicates(records, threshold=threshold, max_candidates=max_candidates,
                                bucket_limit=bucket_limit, include_pairs=True)


def _category(value: object) -> str:
    if isinstance(value, str):
        return value.strip() or "[missing]"
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return " / ".join(value) or "[missing]"
    return "[missing]"


def _audit_catalog(path: Path, records: list[dict[str, str]], working_dir: Path) -> dict:
    examples: dict = {}
    missing: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    chars, words = _Lengths(), _Lengths()
    total = malformed = invalid_ids = duplicates = empty_text = 0
    with tempfile.TemporaryDirectory(prefix="catalog-audit-", dir=working_dir) as temporary:
        connection = sqlite3.connect(str(Path(temporary) / "catalog.sqlite3"))
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA cache_size=-16384")
            connection.execute("CREATE TABLE items (item_id TEXT PRIMARY KEY, category TEXT, usable INTEGER)")
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    total += 1
                    try:
                        raw = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        malformed += 1
                        _example(examples, "malformed_json", line=line_number)
                        continue
                    if not isinstance(raw, dict):
                        malformed += 1
                        _example(examples, "non_object_json", line=line_number)
                        continue
                    for field in ("item_id", "category", "metadata"):
                        if field not in raw or raw[field] is None or raw[field] == "":
                            missing[field] += 1
                    item_id = raw.get("item_id")
                    if not isinstance(item_id, str) or not item_id.strip():
                        invalid_ids += 1
                        _example(examples, "invalid_item_id", line=line_number)
                        continue
                    item_id = item_id.strip()
                    category = _category(raw.get("category"))
                    text = product_text(raw.get("metadata"))
                    inserted = connection.execute("INSERT OR IGNORE INTO items VALUES (?, ?, ?)",
                                                  (item_id, category, int(bool(text)))).rowcount
                    if not inserted:
                        duplicates += 1
                        _example(examples, "duplicate_item_id", line=line_number, item_id=item_id)
                        continue
                    categories[category] += 1
                    chars.add(len(text))
                    words.add(len(text.split()))
                    if not text:
                        empty_text += 1
                        _example(examples, "empty_product_text", line=line_number, item_id=item_id)
                    elif len(text) > 100000:
                        _example(examples, "product_over_100000_characters", line=line_number, item_id=item_id)
                    if total % 10000 == 0:
                        connection.commit()
            connection.commit()
            unique_products = connection.execute("SELECT count(*) FROM items").fetchone()[0]
            target_counts = Counter(record["item_id"] for record in records)
            target_categories: Counter[str] = Counter()
            absent, unusable = [], []
            for item_id in sorted(target_counts):
                match = connection.execute("SELECT category, usable FROM items WHERE item_id = ?", (item_id,)).fetchone()
                if match is None:
                    absent.append(item_id)
                    _example(examples, "target_missing_from_catalog", item_id=item_id)
                else:
                    target_categories[match[0]] += target_counts[item_id]
                    if not match[1]:
                        unusable.append(item_id)
                        _example(examples, "target_has_no_product_text", item_id=item_id)
            absent_rows = sum(target_counts[item_id] for item_id in absent)
            unusable_rows = sum(target_counts[item_id] for item_id in unusable)
        finally:
            connection.close()
    return {"records_read": total, "unique_item_ids": unique_products,
            "malformed_records": malformed, "invalid_item_id_records": invalid_ids,
            "duplicate_item_id_records": duplicates, "missing_values": dict(missing),
            "unique_products_with_empty_text": empty_text,
            "category_distribution_unique_products": dict(sorted(categories.items())),
            "category_distribution_query_targets": dict(sorted(target_categories.items())),
            "length_characters": chars.report(), "length_whitespace_tokens": words.report(),
            "target_coverage": {"unique_query_targets": len(target_counts), "missing_unique_targets": len(absent),
                                "missing_target_query_rows": absent_rows, "present_targets_without_text": len(unusable),
                                "target_without_text_query_rows": unusable_rows,
                                "fraction_query_rows_with_target": (len(records) - absent_rows) / len(records) if records else None,
                                "fraction_query_rows_with_usable_target": (len(records) - absent_rows - unusable_rows) / len(records) if records else None},
            "abnormal_samples": examples,
            "duplicate_policy": "First occurrence used for statistics; duplicate IDs require resolution before preparation."}


def audit_dataset(raw_dir: Path, output_dir: Path) -> dict:
    """Audit official raw filenames and persist auditable, privacy-safe reports."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    query_path, catalog_path = raw_dir / "test.csv", raw_dir / "sampled_item_metadata_1M.jsonl"
    for path in (query_path, catalog_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Amazon-C4 source file missing: {path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records, queries = read_safe_queries(query_path)
    qid_counts = Counter(record["qid"] for record in records)
    normalized_counts = Counter(normalize_query(record["query"]) for record in records)
    users = Counter(record["user_id"] for record in records)
    targets = Counter(record["item_id"] for record in records)
    queries.update({"qid_repetition": _repetition(qid_counts, include_ids=True),
                    "raw_query_repetition": _repetition(Counter(record["query"] for record in records)),
                    "normalized_query_repetition": _repetition(normalized_counts),
                    "target_repetition": _repetition(targets, include_ids=True),
                    "user_repetition": _repetition(users, include_ids=True),
                    "near_duplicates": find_near_duplicates(records)})
    groups = query_components(records)
    sizes = [len(group) for group in groups]
    largest = max(sizes, default=0)
    graph = {"edge_fields": ["user_id", "item_id", "normalized_query"],
             "components": len(groups), "singletons": sizes.count(1), "largest_component_rows": largest,
             "largest_component_fraction": largest / len(records) if records else 0,
             "largest_50_component_sizes": sizes[:50],
             "size_distribution": {str(size): count for size, count in sorted(Counter(sizes).items())},
             "giant_component_warning": largest / len(records) > 0.1 if records else False,
             "warning_threshold_fraction": 0.1,
             "note": "Components are indivisible for splitting. A giant component can prevent desired task counts/ratios; do not break edges to force ratios."}
    catalog = _audit_catalog(catalog_path, records, output_dir)
    report = {"schema_version": 1, "dataset": "McAuley-Lab/Amazon-C4", "protocol": "custom training protocol; not the original evaluation protocol",
              "source_files": {path.name: {"bytes": path.stat().st_size} for path in (query_path, catalog_path)},
              "queries": queries, "catalog": catalog, "grouping_feasibility": graph,
              "privacy": {"query_record_whitelist": list(SAFE_QUERY_FIELDS), "product_dict_text_whitelist": ["title", "description"],
                          "forbidden_values_inspected_or_exported": False, "abnormal_samples": "IDs and line numbers only; no source text"},
              "limitations": ["Queries are synthetic rewrites of positive reviews, not online search logs.",
                              "One labeled parent-ASIN target does not establish complete relevance or variant-level correctness.",
                              "User repetition measures potential overlap before splits, not actual split leakage.",
                              "Near-duplicate search is approximate; exact graph isolation does not certify semantic isolation.",
                              "Invalid/incomplete queries are excluded from linkage and group statistics; counts are reported."]}
    (output_dir / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = ["# Amazon-C4 data audit", "", "This is an audit for a custom training protocol, not a reproduction of the original evaluation protocol.", "",
               f"- Query records read: {queries['records_read']}; complete records: {len(records)}.",
               f"- Malformed query records: {queries['malformed_records']}; incomplete: {queries['incomplete_rows']}.",
               f"- Repeated qids: {queries['qid_repetition']['repeated_keys']}; repeated normalized queries: {queries['normalized_query_repetition']['repeated_keys']}.",
               f"- Verified near-duplicate representative pairs (approximate lower bound): {queries['near_duplicates']['verified_representative_pairs']}.",
               f"- Unique catalog items: {catalog['unique_item_ids']}; duplicate item records: {catalog['duplicate_item_id_records']}.",
               f"- Missing unique targets: {catalog['target_coverage']['missing_unique_targets']}; present targets without text: {catalog['target_coverage']['present_targets_without_text']}.",
               f"- Components: {len(groups)}; largest: {largest} ({graph['largest_component_fraction']:.2%} of complete queries).", "",
               "The JSON report contains complete counts, category/length distributions, ID-only anomaly examples, and method limitations. No review/rating values or query text are exported.", "",
               "Shared target, user, and normalized-query edges must remain intact in both levels of splitting. Near-duplicate candidates require further split-boundary checks. Parent-ASIN labels do not justify color/size accuracy claims.", ""]
    (output_dir / "AUDIT.md").write_text("\n".join(summary), encoding="utf-8")
    return report
