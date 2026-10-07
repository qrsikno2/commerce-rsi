#!/usr/bin/env python3
"""Regenerate factual pilot reports from saved artifacts, including failed attempts.

No GPU, model, raw data, private labels, or external services are accessed. Missing
measurements remain null; zero is used only for explicitly recorded zero or an
empty collection of events. In-progress artifacts are a best-effort snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


METRICS = ("mrr@10", "mrr", "hit@10", "ndcg@10")


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def aggregate(values, operation="sum"):
    values = list(values)
    known = [value for value in values if numeric(value)]
    result = None
    if known:
        result = sum(known) if operation == "sum" else max(known)
    elif not values and operation == "sum":
        result = 0
    return {"value": result, "recorded": len(known), "expected": len(values),
            "complete": len(known) == len(values)}


def descriptive(values):
    values = list(values)
    known = [value for value in values if numeric(value)]
    return {"n": len(known), "missing": len(values) - len(known),
            "mean": statistics.mean(known) if known else None,
            "sample_std": statistics.stdev(known) if len(known) >= 2 else None,
            "min": min(known) if known else None, "max": max(known) if known else None}


class Snapshot:
    def __init__(self, root):
        self.root = root.resolve()
        self.inventory = {}
        self.warnings = []
        self.cache = {}

    def relative(self, path):
        try:
            return str(Path(path).resolve().relative_to(self.root))
        except ValueError:
            return str(path)

    def resolve_recorded_path(self, value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def read(self, path):
        path = Path(path).resolve()
        if path in self.cache:
            return self.cache[path]
        if not path.exists():
            return None
        try:
            data = path.read_bytes()
            value = json.loads(data)
            self.inventory[self.relative(path)] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        except (OSError, ValueError) as error:
            self.warnings.append({"path": self.relative(path), "reason": type(error).__name__,
                                  "note": "Unreadable or concurrently written artifact; no values imputed."})
            value = None
        self.cache[path] = value
        return value


def business_runs(snapshot):
    base = snapshot.root / "artifacts" / "business_pilot"
    rows = []
    for path in sorted(base.rglob("run.json")):
        raw = snapshot.read(path)
        if not isinstance(raw, dict):
            continue
        parts = path.relative_to(base).parts
        rows.append({
            "path": snapshot.relative(path), "series": parts[0] if len(parts) >= 3 else "standalone",
            **{key: raw.get(key) for key in ("status", "task_id", "strategy", "seed", "steps", "completed_steps",
                                             "batch_size", "num_negatives", "config", "evaluation_split", "outer_split",
                                             "business_initialization_sha256", "batch_schedule_sha256", "business_code_sha256",
                                             "prepared_manifest_sha256", "parameters_changed", "training_tokens", "failure")},
            "data_sha256": raw.get("data_sha256", {}), "metrics": raw.get("metrics", {}),
            "initial_metrics": raw.get("initial_metrics", {}), "cost": raw.get("cost", {}),
        })
    return rows


def configuration_key(row, include_strategy=False):
    identity = {key: row.get(key) for key in ("series", "task_id", "steps", "batch_size", "num_negatives", "config",
                                             "data_sha256", "business_code_sha256", "evaluation_split")}
    if include_strategy:
        identity["strategy"] = row.get("strategy")
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def business_tables(rows):
    groups, pair_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[configuration_key(row, True)].append(row)
        if row["strategy"] in {"uniform", "lexical_hard"}:
            pair_groups[configuration_key(row)].append(row)
    summaries = []
    for key, records in sorted(groups.items()):
        identity = json.loads(key)
        completed = [row for row in records if row["status"] == "completed"]
        summaries.append({"id": hashlib.sha256(key.encode()).hexdigest()[:12], **identity,
                          "runs": len(records), "completed": len(completed),
                          "status_counts": dict(Counter(row["status"] or "unknown" for row in records)),
                          "seeds": sorted({row["seed"] for row in records if row["seed"] is not None}),
                          "metrics": {metric: descriptive(row["metrics"].get(metric) for row in completed) for metric in METRICS},
                          "business_total_wall_seconds": aggregate(row["cost"].get("business_total_wall_seconds") for row in records),
                          "gpu_peak_allocated_bytes": aggregate((row["cost"].get("gpu_peak_allocated_bytes") for row in records), "max"),
                          "source_paths": [row["path"] for row in records]})
    paired = []
    for key, records in sorted(pair_groups.items()):
        by_seed = defaultdict(lambda: defaultdict(list))
        for row in records:
            if row["status"] == "completed":
                by_seed[row["seed"]][row["strategy"]].append(row)
        pairs, rejected = [], []
        for seed, strategies in sorted(by_seed.items(), key=lambda value: str(value[0])):
            if len(strategies["uniform"]) != 1 or len(strategies["lexical_hard"]) != 1:
                rejected.append({"seed": seed, "reason": "Missing or ambiguous strategy replicate"})
                continue
            uniform, hard = strategies["uniform"][0], strategies["lexical_hard"][0]
            checks = {field: bool(uniform.get(field)) and uniform[field] == hard.get(field)
                      for field in ("business_initialization_sha256", "batch_schedule_sha256")}
            if not all(checks.values()):
                rejected.append({"seed": seed, "reason": "Initialization or batch schedule not verifiably paired", "checks": checks})
                continue
            deltas = {metric: hard["metrics"][metric] - uniform["metrics"][metric]
                      if numeric(hard["metrics"].get(metric)) and numeric(uniform["metrics"].get(metric)) else None
                      for metric in METRICS}
            pairs.append({"seed": seed, "uniform_path": uniform["path"], "lexical_hard_path": hard["path"],
                          "delta_lexical_minus_uniform": deltas, "pairing_checks": checks})
        paired.append({**json.loads(key), "pairs": pairs, "excluded_seed_pairs": rejected,
                       "delta_metrics": {metric: descriptive(pair["delta_lexical_minus_uniform"][metric] for pair in pairs)
                                         for metric in METRICS}})
    return summaries, paired


def summarize_pilot(snapshot, path):
    manifest = snapshot.read(path)
    if not isinstance(manifest, dict):
        return None
    directory = path.parent.resolve()
    embedded_groups = {snapshot.resolve_recorded_path(row["path"]): row for row in manifest.get("groups", []) if row.get("path")}
    group_dirs = set(embedded_groups)
    for filename in ("search_request.json", "rollout.json", "group_report.json", "executions.json"):
        group_dirs.update(file.parent.resolve() for file in directory.rglob(filename))
    group_rows, all_results, business_records, generation_costs = [], [], [], []
    for group_dir in sorted(group_dirs):
        embedded = embedded_groups.get(group_dir, {})
        snapshot.read(group_dir / "search_request.json")
        rollout = snapshot.read(group_dir / "rollout.json") or {}
        group_report = snapshot.read(group_dir / "group_report.json") or embedded
        executions = snapshot.read(group_dir / "executions.json")
        results = executions if isinstance(executions, list) else group_report.get("results", [])
        results = results if isinstance(results, list) else []
        cost = rollout.get("cost", group_report.get("generation_cost", {}))
        generation_costs.append(cost)
        all_results.extend(results)
        candidate_runs = {}
        for run_path in sorted(group_dir.glob("candidate_*/business/run.json")):
            run = snapshot.read(run_path)
            if isinstance(run, dict):
                candidate_runs[run_path.parent.parent.name] = {"path": snapshot.relative(run_path), "status": run.get("status"),
                                                             "cost": run.get("cost", {}), "failure": run.get("failure")}
        for business_dir in sorted(group_dir.glob("candidate_*/business")):
            if business_dir.is_dir() and business_dir.parent.name not in candidate_runs:
                candidate_runs[business_dir.parent.name] = {"path": snapshot.relative(business_dir / "run.json"),
                                                            "status": "running_or_incomplete", "cost": {}}
        for result in results:
            candidate = f"candidate_{result.get('completion_index')}"
            if candidate not in candidate_runs and isinstance(result.get("business_cost"), dict):
                candidate_runs[candidate] = {"path": None, "status": "unknown; execution-report cost fallback",
                                             "cost": result["business_cost"]}
        business_records.extend(candidate_runs.values())
        generated = len(rollout["completions"]) if isinstance(rollout.get("completions"), list) else None
        group_rows.append({"path": snapshot.relative(group_dir),
                           "checkpoint_id": rollout.get("checkpoint_id", group_report.get("checkpoint_id")),
                           "generation_seed": rollout.get("generation_seed"), "generated_completions": generated,
                           "recorded_executions": len(results), "valid": sum(row.get("success") is True for row in results),
                           "failed": sum(row.get("success") is False for row in results),
                           "unresolved_completions": max(0, generated - len(results)) if generated is not None else None,
                           "group_report_present": (group_dir / "group_report.json").exists(),
                           "generation_cost": cost, "results": results})
    embedded_updates = {snapshot.resolve_recorded_path(row["path"]): row for row in manifest.get("lineage", []) if row.get("path")}
    update_dirs = (set(embedded_updates) | {file.parent.resolve() for file in directory.rglob("update.json")}
                   | {folder.resolve() for folder in directory.glob("A*_attempt*") if folder.is_dir()})
    updates = []
    for update_dir in sorted(update_dirs):
        raw = snapshot.read(update_dir / "update.json") or embedded_updates.get(update_dir, {})
        before, after, reloaded = (raw.get(field) for field in ("adapter_before_sha256", "adapter_after_sha256", "adapter_reloaded_sha256"))
        row = {**raw, "path": snapshot.relative(update_dir),
               "adapter_hash_changed_verified": before != after if before and after else None,
               "reload_hash_match_verified": after == reloaded if after and reloaded else None}
        updates.append(row)
    updated = [row for row in updates if row.get("status") == "updated"]
    pending_execution_costs = []
    for row in group_rows:
        pending_execution_costs.extend([None] * (row["unresolved_completions"] if row["unresolved_completions"] is not None else 1))
    costs = {"generation_load_seconds": aggregate(row.get("agent_load_seconds") for row in generation_costs),
             "generation_wall_seconds": aggregate(row.get("agent_generation_wall_seconds") for row in generation_costs),
             "generation_prompt_tokens": aggregate(row["generation_cost"]["prompt_tokens_per_completion"] * row["generated_completions"]
                                                     if numeric(row["generation_cost"].get("prompt_tokens_per_completion"))
                                                     and numeric(row["generated_completions"]) else None for row in group_rows),
             "completion_tokens": aggregate(row.get("completion_tokens") for row in generation_costs),
             "truncated_completions": aggregate(row.get("truncated_completions") for row in generation_costs),
             "execution_wall_seconds": aggregate([row.get("execution_wall_seconds") for row in all_results] + pending_execution_costs),
             "business_total_wall_seconds": aggregate(row["cost"].get("business_total_wall_seconds") for row in business_records),
             "business_training_wall_seconds": aggregate(row["cost"].get("business_training_wall_seconds") for row in business_records),
             "business_cuda_event_seconds": aggregate(row["cost"].get("business_training_cuda_event_seconds") for row in business_records),
             "sampler_seconds": aggregate(row["cost"].get("sampler_seconds") for row in business_records),
             "agent_update_total_wall_seconds": aggregate(row.get("agent_update_total_wall_seconds") for row in updates),
             "update_load_seconds": aggregate(row.get("agent_load_seconds") for row in updates),
             "optimizer_step_seconds": aggregate(row.get("optimizer_step_wall_seconds") for row in updates),
             "update_training_tokens": aggregate(row.get("training_tokens") for row in updates),
             "pilot_wall_seconds": manifest.get("total_wall_seconds")}
    gpu = {phase: {field: aggregate((row.get(field) for row in phase_rows), "max")
                   for field in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes")}
           for phase, phase_rows in (("generation", generation_costs),
                                     ("business", [row["cost"] for row in business_records]), ("update", updates))}
    lineage_checks = []
    for update in updated:
        output_checkpoint = update.get("output_checkpoint_id")
        subsequent_groups = [row["path"] for row in group_rows if output_checkpoint and row["checkpoint_id"] == output_checkpoint]
        lineage_checks.append({"adapter_path": update["path"], "output_checkpoint_id": output_checkpoint,
                               "parameters_changed_flag": update.get("parameters_changed"),
                               "adapter_hash_changed": update["adapter_hash_changed_verified"],
                               "reload_verified_flag": update.get("reload_verified"),
                               "reload_hash_match": update["reload_hash_match_verified"],
                               "rollout_groups_using_output_checkpoint": subsequent_groups})
    return {"path": snapshot.relative(path), "name": directory.name,
            **{key: manifest.get(key) for key in ("status", "mode", "task_id", "outer_split", "primary_metric", "failure",
                                                 "rounds", "max_attempts_per_round", "group_size", "business_steps", "business_config",
                                                 "max_new_tokens", "seed", "package_sha256", "initial_adapter", "final_adapter")},
            "groups": group_rows, "group_count": len(group_rows),
            "generated_completions": aggregate(row["generated_completions"] for row in group_rows),
            "recorded_executions": len(all_results), "valid_executions": sum(row.get("success") is True for row in all_results),
            "failed_executions": sum(row.get("success") is False for row in all_results),
            "unknown_execution_status": sum(row.get("success") not in (True, False) for row in all_results),
            "unresolved_completions": aggregate(row["unresolved_completions"] for row in group_rows),
            "failure_counts": dict(Counter(row.get("failure", "reason missing") for row in all_results if row.get("success") is False)),
            "reward_statistics_all_recorded_executions": descriptive(row.get("reward") for row in all_results),
            "valid_metrics": {metric: descriptive(row.get("metrics", {}).get(metric) for row in all_results if row.get("success") is True)
                              for metric in METRICS},
            "updates": updates, "update_status_counts": dict(Counter(row.get("status", "unknown") for row in updates)),
            "actual_updated_count": len(updated), "no_update_count": sum(row.get("status") == "no_update" for row in updates),
            "verified_parameter_change_and_reload_count": sum(row.get("parameters_changed") is True and row.get("reload_verified") is True
                                                               and row["adapter_hash_changed_verified"] is True and row["reload_hash_match_verified"] is True
                                                               for row in updated),
            "checkpoint_lineage_checks": lineage_checks, "costs": costs, "gpu_phase_maxima": gpu,
            "business_attempts": business_records,
            "snapshot_note": "Completed artifacts are counted once; active generation/update without a saved cost record remains unknown."}


def build_report(root):
    snapshot = Snapshot(root)
    business = business_runs(snapshot)
    summaries, pairs = business_tables(business)
    pilots = [value for path in sorted((root / "artifacts" / "recursion").rglob("pilot.json"))
              if (value := summarize_pilot(snapshot, path)) is not None]
    checkpoint_rollouts = defaultdict(list)
    for pilot in pilots:
        for group in pilot["groups"]:
            if group.get("checkpoint_id"):
                checkpoint_rollouts[group["checkpoint_id"]].append(group["path"])
    for pilot in pilots:
        for check in pilot["checkpoint_lineage_checks"]:
            check["rollout_groups_using_output_checkpoint"] = checkpoint_rollouts.get(check["output_checkpoint_id"], [])
    pilot_cost_totals = {}
    if pilots:
        for key in pilots[0]["costs"]:
            measurements = [pilot["costs"][key] for pilot in pilots]
            if key == "pilot_wall_seconds":
                pilot_cost_totals[key] = aggregate(measurements)
                continue
            total = aggregate(row["value"] for row in measurements)
            total.update(recorded=sum(row["recorded"] for row in measurements),
                         expected=sum(row["expected"] for row in measurements),
                         complete=all(row["complete"] for row in measurements))
            pilot_cost_totals[key] = total
    return {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "business_runs": business, "business_config_summaries": summaries, "paired_seed_comparisons": pairs,
            "pilots": pilots, "pilot_cost_totals": pilot_cost_totals,
            "business_total_wall_seconds": aggregate(row["cost"].get("business_total_wall_seconds") for row in business),
            "source_inventory": snapshot.inventory, "read_warnings": snapshot.warnings,
            "methods": {"std": "Sample standard deviation (ddof=1), unknown for one or zero observations.",
                        "missing": "No missing metric or measurement is filled with zero. Aggregates report observed/expected coverage.",
                        "pairing": "Same series, task, raw config, budget, data hashes, business-code hash, seed, initialization hash and batch schedule.",
                        "costs": "Generation load + generation wall + execution wall + update total are separate top-level phases; business/sampler/update-load/optimizer times are overlapping submeasurements, not additional costs.",
                        "progress": "Point-in-time file snapshot; partial attempts remain visible and incomplete measurements are flagged."},
            "limitations": ["Real business training and successful Agent parameter updates are distinct evidence levels.",
                            "Two updated/reloaded adapters prove a control path, not recursive capability improvement or RISE/RSI superiority.",
                            "This report does not perform final evaluation or unlock meta-test. It only reads saved development/pilot logs.",
                            "No equal-total-compute comparison is inferred; controller arguments alone do not match actual groups, tokens, or update costs.",
                            "Seed variability on one task is not independent-task generalization; paired standard deviations are descriptive.",
                            "Earlier independent-tower and later shared-tower business runs have different configurations and must not be pooled.",
                            "Full MRR and MRR@10 are distinct. Missing older full-MRR values remain unknown."]}


def fmt(value, digits=6):
    return f"{value:.{digits}f}" if numeric(value) else "unknown"


def measurement(value, digits=2):
    if not isinstance(value, dict):
        return fmt(value, digits)
    suffix = "" if value["complete"] else f" [partial {value['recorded']}/{value['expected']}]"
    return fmt(value["value"], digits) + suffix


def metric_cell(value):
    if not value["n"]:
        return "unknown (n=0)"
    return f"{fmt(value['mean'])} ± {fmt(value['sample_std'])} (n={value['n']})"


def markdown(report):
    lines = ["# Recorded experiments", "", f"Generated from saved artifacts at {report['generated_at_utc']}.", "",
             "Regenerate with `.venv/bin/python scripts/report_experiments.py`. The detailed JSON ledger is",
             "`artifacts/experiment_summary.json`; it includes every source-file hash, failed attempt, update record, and measurement coverage.", "",
             "These are development/control-path results. Final evaluation and meta-test remain locked. Unknown values are not zeros.", "",
             "## Business-model runs", "", "Metrics use completed runs only; costs include every saved run. Values are mean ± sample standard deviation across the listed paired-seed configuration; n=1 has unknown standard deviation.", "",
             "| Series / task | Strategy | Steps | Towers | Seeds | MRR@10 | Full MRR | Total business seconds |",
             "| --- | --- | ---: | --- | --- | --- | --- | ---: |"]
    for row in report["business_config_summaries"]:
        config = row.get("config") or {}
        towers = "shared" if config.get("shared_towers") is True else "independent (legacy field absent)" if "shared_towers" not in config else "independent"
        lines.append(f"| {row['series']} / {row['task_id']} | {row['strategy']} | {row['steps']} | {towers} | {', '.join(map(str, row['seeds']))} | {metric_cell(row['metrics']['mrr@10'])} | {metric_cell(row['metrics']['mrr'])} | {measurement(row['business_total_wall_seconds'])} |")
    if not report["business_config_summaries"]:
        lines.append("| No saved business runs | — | — | — | — | unknown | unknown | unknown |")
    lines += ["", "Configurations are separated by exact saved settings, data hashes, and code hash. Older run files without `shared_towers` came from the independent-tower implementation; they are not pooled with later shared-tower runs.", "",
              "## Paired lexical-hard minus uniform differences", "",
              "Only uniquely paired seeds with matching initialization and training-query schedules are included. Negative differences favor uniform for that metric.", "",
              "| Series / task | Steps | Paired seeds | Δ MRR@10 | Δ full MRR | Excluded seed pairs |",
              "| --- | ---: | --- | --- | --- | ---: |"]
    for row in report["paired_seed_comparisons"]:
        lines.append(f"| {row['series']} / {row['task_id']} | {row['steps']} | {', '.join(str(pair['seed']) for pair in row['pairs']) or 'none'} | {metric_cell(row['delta_metrics']['mrr@10'])} | {metric_cell(row['delta_metrics']['mrr'])} | {len(row['excluded_seed_pairs'])} |")
    lines += ["", "## Agent pilots", "",
              "An `updated` count requires a saved update record with that status. Verified counts additionally require changed adapter hashes and exact reload agreement. No-update attempts still consume load/update time and remain in all costs.", "",
              "| Pilot | Mode / task | Status | Groups | Valid / failed / pending completions | Updated / no-update | Hash + reload verified | Valid full MRR mean |",
              "| --- | --- | --- | ---: | --- | --- | ---: | ---: |"]
    for row in report["pilots"]:
        pending = row["unresolved_completions"]["value"]
        lines.append(f"| {row['name']} | {row['mode']} / {row['task_id']} | {row['status'] or 'unknown'} | {row['group_count']} | {row['valid_executions']} / {row['failed_executions']} / {pending if pending is not None else 'unknown'} | {row['actual_updated_count']} / {row['no_update_count']} | {row['verified_parameter_change_and_reload_count']} | {fmt(row['valid_metrics']['mrr']['mean'])} |")
    if not report["pilots"]:
        lines.append("| No saved Agent pilots | — | unknown | 0 | 0 / 0 / unknown | 0 / 0 | 0 | unknown |")
    lines += ["", "| Pilot | Generation load s | Generation s | Candidate execution s | Agent update total s | Update load s (included) | Optimizer s (included) | Completion tokens | Recorded wall s |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["pilots"]:
        costs = row["costs"]
        keys = ("generation_load_seconds", "generation_wall_seconds", "execution_wall_seconds", "agent_update_total_wall_seconds",
                "update_load_seconds", "optimizer_step_seconds", "completion_tokens", "pilot_wall_seconds")
        lines.append("| " + row["name"] + " | " + " | ".join(measurement(costs[key], 0 if key == "completion_tokens" else 2) for key in keys) + " |")
    if report["pilot_cost_totals"]:
        costs = report["pilot_cost_totals"]
        lines.append("| All saved pilots, including failures | " + " | ".join(measurement(costs[key], 0 if key == "completion_tokens" else 2) for key in keys) + " |")
    lines += ["", "Business total/training, sampler, CUDA-event, and update-token submeasurements are retained in the JSON. Execution seconds include business work; update load and optimizer seconds are subsets of update total. Do not sum overlapping columns.", "",
              "| Pilot | Generation peak allocated GiB | Business peak allocated GiB | Update peak allocated GiB |",
              "| --- | ---: | ---: | ---: |"]
    for row in report["pilots"]:
        values = []
        for phase in ("generation", "business", "update"):
            item = row["gpu_phase_maxima"][phase]["gpu_peak_allocated_bytes"]
            scaled = {**item, "value": item["value"] / 1024**3 if numeric(item["value"]) else None}
            values.append(measurement(scaled, 3))
        lines.append("| " + row["name"] + " | " + " | ".join(values) + " |")
    for row in report["pilots"]:
        if row.get("failure"):
            lines += ["", f"`{row['name']}` termination: {row['failure']}"]
        if row["failure_counts"]:
            lines += ["", f"`{row['name']}` recorded candidate failures:", ""]
            lines += [f"- {count} × {reason}" for reason, count in sorted(row["failure_counts"].items())]
    lines += ["", "## Limits of the evidence", ""]
    lines += [f"- {limitation}" for limitation in report["limitations"]]
    lines += ["- Running or incomplete attempts may have generated tokens or consumed GPU time before a cost artifact exists; that unrecorded work is unknown, not free.",
              "- Manual inspection found plausible unlabeled relevant alternatives and many Books/Kindle lexical negatives in the Toys baseline; see `REVIEW.md`. Single-target retrieval scores do not establish full relevance or user satisfaction.", ""]
    if report["read_warnings"]:
        lines += ["Snapshot read warnings (regenerate after writers finish):", ""]
        lines += [f"- `{warning['path']}`: {warning['reason']}" for warning in report["read_warnings"]]
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    report = build_report(root)
    destination = root / "artifacts" / "experiment_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    document = root / "docs" / "EXPERIMENTS.md"
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"business_runs": len(report["business_runs"]), "pilots": len(report["pilots"]),
                      "recorded_updated_count": sum(row["actual_updated_count"] for row in report["pilots"]),
                      "warnings": len(report["read_warnings"]), "json": str(destination), "markdown": str(document)}))


if __name__ == "__main__":
    main()
