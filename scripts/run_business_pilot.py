"""Paired-seed meta-dev calibration, never opens final/meta-test queries."""

import argparse
import json
from pathlib import Path

from commerce_rsi.retrieval.runner import run_business
from commerce_rsi.retrieval.model import BusinessConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/data/prepared"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/business_pilot/paired_1000"))
    parser.add_argument("--task-id", default="toys_7a17d0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--shared-towers", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new pilot output directory")
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    task = next(task for task in manifest["tasks"] if task["task_id"] == args.task_id)
    if task["outer_split"] != "meta_dev":
        raise ValueError("Business calibration uses meta-dev only")
    reports = []
    for strategy, seed in [("bm25", args.seeds[0])] + [
            (strategy, seed) for seed in args.seeds for strategy in ("uniform", "lexical_hard")]:
        report = run_business(args.prepared, args.task_id, strategy,
                              args.output / f"{strategy}_s{seed}", seed=seed, steps=args.steps,
                              config=BusinessConfig(shared_towers=args.shared_towers))
        reports.append(report)
        print(json.dumps({key: report[key] for key in ("strategy", "seed", "metrics", "cost")}), flush=True)
    by_seed = {}
    for report in reports:
        if report["strategy"] == "bm25":
            continue
        old = by_seed.setdefault(report["seed"], report)
        for key in ("business_initialization_sha256", "batch_schedule_sha256", "data_sha256"):
            if old[key] != report[key]:
                raise ValueError(f"Paired experiment mismatch: {key}")
    summary = {"purpose": "meta-dev business calibration only; no Agent update or RSI claim",
               "task_id": args.task_id, "steps": args.steps, "paired_seeds": args.seeds,
               "shared_towers": args.shared_towers,
               "reports": [{key: report[key] for key in ("strategy", "seed", "metrics", "cost")} for report in reports]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
