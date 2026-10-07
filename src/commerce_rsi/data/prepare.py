"""Custom category-task protocol, with grouped splits and a fixed public corpus."""

from __future__ import annotations

import collections
import csv
import hashlib
import heapq
import json
import re
from pathlib import Path

from .download import sha256_file


def _hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def split_groups(groups, seed: int, ratios=(0.7, 0.15, 0.15)):
    """Greedy size allocation of indivisible groups, with deterministic tie breaks."""
    if len(groups) < len(ratios):
        raise ValueError("Not enough independent components for nonempty splits")
    ordered = sorted(groups, key=lambda group: (-len(group), _hash(str(min(group)), seed)))
    total = sum(map(len, groups))
    result = [[] for _ in ratios]
    sizes = [0] * len(ratios)
    for group in ordered:
        slot = max(range(len(ratios)), key=lambda i: (total * ratios[i] - sizes[i], -i))
        result[slot].extend(group)
        sizes[slot] += len(group)
    if any(not group for group in result):
        raise ValueError("Components are too large for three nonempty splits")
    return result


def assert_isolation(records, assignments):
    """Every user/target/normalized query belongs to exactly one inner split."""
    from .audit import normalize_query

    seen = {}
    for index, assignment in assignments.items():
        row = records[index]
        for kind, value in (("user", row["user_id"]), ("target", row["item_id"]),
                            ("query", normalize_query(row["query"]))):
            if not value:
                continue
            key = (kind, value)
            if key in seen and seen[key] != assignment:
                raise ValueError(f"{kind} leaks across task/split boundaries")
            seen[key] = assignment


def prepare_dataset(raw_dir: Path, output_dir: Path, candidate_count: int = 30000,
                    seed: int = 20261007, minimum_task_queries: int = 150) -> dict:
    from .audit import detect_near_duplicates, normalize_query, product_text, query_components

    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Prepared output must be empty; never overwrite a protocol")
    if minimum_task_queries < 3:
        raise ValueError("minimum_task_queries must be at least 3")
    source_path = raw_dir / "source_manifest.json"
    source = json.loads(source_path.read_text())
    for entry in source["files"]:
        if sha256_file(raw_dir / entry["name"]) != entry["sha256"]:
            raise ValueError(f"Raw file changed since acquisition: {entry['name']}")
    with (raw_dir / "test.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = ("qid", "query", "item_id", "user_id")
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError("Missing required query columns")
        # Banned source fields do not cross this whitelist.
        records = [{key: (row.get(key) or "").strip() for key in required} for row in reader]
    raw_count = len(records)
    qids = collections.Counter(row["qid"] for row in records)
    rejected = collections.Counter()
    clean = []
    for row in records:
        if not all(row.values()) or not normalize_query(row["query"]):
            rejected["missing_required_or_empty_query"] += 1
        elif qids[row["qid"]] > 1:
            rejected["ambiguous_duplicate_qid"] += 1
        else:
            clean.append(row)
    records = clean
    targets = {row["item_id"] for row in records}
    target_catalog = {}
    metadata_path = raw_dir / "sampled_item_metadata_1M.jsonl"

    def catalog_rows():
        with metadata_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                item_id = row.get("item_id")
                if not isinstance(item_id, str) or not item_id:
                    raise ValueError(f"Invalid product identifier at line {line_number}")
                text = product_text(row.get("metadata"))
                category = row.get("category")
                if not isinstance(category, str) or not category.strip() or not text.strip():
                    continue
                yield {"item_id": item_id, "category": category.strip(), "text": text}

    for row in catalog_rows():
        if row["item_id"] in targets:
            old = target_catalog.get(row["item_id"])
            if old is not None and old != row:
                raise ValueError("Conflicting duplicate target metadata")
            target_catalog[row["item_id"]] = row
    clean = []
    for row in records:
        if row["item_id"] not in target_catalog:
            rejected["missing_or_empty_target_metadata"] += 1
        else:
            clean.append(row)
    records = clean
    targets = {row["item_id"] for row in records}
    target_catalog = {key: value for key, value in target_catalog.items() if key in targets}
    if candidate_count < len(targets):
        raise ValueError(f"candidate_count={candidate_count} is smaller than {len(targets)} targets")

    # Include all usable targets regardless of their eventual task/split. Nothing in
    # the public catalog marks which items are targets or which split they belong to.
    target_texts = {normalize_query(row["text"]) for row in target_catalog.values()}
    distractor_count = candidate_count - len(target_catalog)
    heap, selected = [], {}
    for row in catalog_rows():
        if row["item_id"] in targets or not distractor_count:
            continue
        text_key = normalize_query(row["text"])
        if text_key in target_texts:
            continue
        rank = int(_hash(row["item_id"], seed), 16)
        existing = selected.get(text_key)
        if existing is not None:
            if rank >= existing[0]:
                continue
            selected[text_key] = (rank, row)
            heapq.heappush(heap, (-rank, text_key, row["item_id"]))
        elif len(selected) < distractor_count:
            selected[text_key] = (rank, row)
            heapq.heappush(heap, (-rank, text_key, row["item_id"]))
        else:
            while heap and (heap[0][1] not in selected or selected[heap[0][1]][0] != -heap[0][0]):
                heapq.heappop(heap)
            if rank < -heap[0][0]:
                _, worst_text, _ = heapq.heappop(heap)
                del selected[worst_text]
                selected[text_key] = (rank, row)
                heapq.heappush(heap, (-rank, text_key, row["item_id"]))
    catalog = sorted(list(target_catalog.values()) + [value[1] for value in selected.values()],
                     key=lambda row: row["item_id"])
    if len(catalog) != candidate_count or len({row["item_id"] for row in catalog}) != len(catalog):
        raise ValueError("Cannot construct requested unique candidate pool")

    near = detect_near_duplicates(records, threshold=0.9)
    if near["candidate_budget_exhausted"] or near["capped_bucket_insertions"]:
        raise ValueError("Near-duplicate audit hit its cap; review grouping before preparation")
    components = query_components(records, near_pairs=near["pairs"])
    categories = collections.defaultdict(list)
    excluded = []
    for component in components:
        component_categories = {target_catalog[records[i]["item_id"]]["category"] for i in component}
        if len(component_categories) != 1:
            excluded.extend(component)
            rejected["cross_category_component_queries"] += len(component)
        else:
            categories[next(iter(component_categories))].append(component)
    eligible = {}
    for category, groups in categories.items():
        if sum(map(len, groups)) < minimum_task_queries or len(groups) < 3:
            excluded.extend(i for group in groups for i in group)
            rejected["insufficient_task_queries"] += sum(map(len, groups))
        else:
            eligible[category] = groups
    category_order = sorted(eligible, key=lambda c: _hash(c, seed))
    if len(category_order) < 3:
        raise ValueError(f"Only {len(category_order)} eligible categories; cannot claim three outer splits")
    # Whole categories are held out. Seed changes do not create additional tasks.
    n_dev = max(1, round(len(category_order) * 0.2))
    n_test = max(1, round(len(category_order) * 0.2))
    n_train = len(category_order) - n_dev - n_test
    roles = ["meta_train"] * n_train + ["meta_dev"] * n_dev + ["meta_test"] * n_test
    assignments, tasks = {}, []
    for category, role in zip(category_order, roles):
        task_id = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_") + "_" + _hash(category, seed)[:6]
        inner = split_groups(eligible[category], seed)
        counts = {}
        for split_name, indices in zip(("train", "reward_validation", "final_evaluation"), inner):
            for i in indices:
                assignments[i] = (role, task_id, split_name)
            counts[split_name] = len(indices)
        tasks.append({"task_id": task_id, "category": category, "outer_split": role,
                      "query_counts": counts, "components": len(eligible[category])})
    assert_isolation(records, assignments)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = output_dir / "public" / "catalog.jsonl"
    _jsonl(catalog_path, catalog)
    query_files = []
    for task in tasks:
        for split_name in ("train", "reward_validation", "final_evaluation"):
            if task["outer_split"] == "meta_test":
                base = output_dir / "private" / "meta_test" / task["task_id"]
            elif split_name == "final_evaluation":
                base = output_dir / "private" / "final_evaluation" / task["task_id"]
            else:
                base = output_dir / "tasks" / task["task_id"]
            path = base / f"{split_name}.jsonl"
            indices = sorted((i for i, a in assignments.items()
                              if a == (task["outer_split"], task["task_id"], split_name)),
                             key=lambda i: records[i]["qid"])
            _jsonl(path, ({key: records[i][key] for key in ("qid", "query", "item_id")} for i in indices))
            query_files.append({"path": str(path.relative_to(output_dir)), "sha256": sha256_file(path)})
    private_map = [{"qid": records[i]["qid"], "user_id": records[i]["user_id"],
                    "item_id": records[i]["item_id"], "outer_split": a[0],
                    "task_id": a[1], "inner_split": a[2]} for i, a in assignments.items()]
    _jsonl(output_dir / "private" / "split_assignments.jsonl", private_map)
    manifest = {
        "protocol": "amazon-c4-custom-category-tasks-v1",
        "status": "pilot; not an original Amazon-C4 benchmark reproduction",
        "dataset_revision": source["revision"], "source_manifest_sha256": sha256_file(source_path),
        "seed": seed, "raw_query_count": raw_count, "usable_query_count": len(assignments),
        "excluded_query_counts": dict(rejected), "candidate_count": len(catalog),
        "catalog_sha256": sha256_file(catalog_path),
        "candidate_rule": "All valid targets; smallest SHA256(seed:item_id) distractors, unique normalized metadata texts; target text aliases retained and excluded by sampler validation",
        "component_rule": "Shared target item, nonempty user, normalized query, or detected near-query pair (Jaccard>=0.9); cross-category components quarantined intact",
        "near_duplicate_check": {key: value for key, value in near.items() if key != "pairs"},
        "component_count": len(components), "largest_component_queries": max(map(len, components), default=0),
        "inner_ratios_requested": [0.7, 0.15, 0.15], "tasks": tasks, "query_files": query_files,
        "public_overlap": "All tasks share the same item texts; hidden target mappings and split memberships are never sampler inputs",
        "evaluation_lock": "meta_test and all final_evaluation files are private; no final evaluation before protocol and checkpoints are frozen",
    }
    _dump(output_dir / "manifest.json", manifest)
    _dump(output_dir / "public" / "catalog_manifest.json", {key: manifest[key] for key in
          ("protocol", "candidate_count", "catalog_sha256", "candidate_rule")})
    return manifest
