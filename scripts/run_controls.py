"""Matched proposal-count smoke controls, with learning costs kept separate."""

import json
from pathlib import Path

from commerce_rsi.recursion.pilot import run_pilot, write_json


root = Path(__file__).resolve().parents[1]
recursive_dir = root / "artifacts/recursion/pilot_v2"
recursive = json.loads((recursive_dir / "pilot.json").read_text())
updates = [update for update in recursive["lineage"] if update["status"] == "updated"]
if recursive["status"] != "completed" or len(updates) != 2:
    raise RuntimeError("Controls require two actual saved generations")
if len(recursive["groups"]) != 2 or any(len(group["results"]) != 2 for group in recursive["groups"]):
    raise RuntimeError("This fixed-control budget matches exactly two groups of two; inspect other budgets explicitly")
model = root / "artifacts/models/Qwen2.5-Coder-3B-Instruct"
task_dir = root / "artifacts/tasks/schema_v2"
fixed_dir = root / "artifacts/recursion/fixed_A0"
if not fixed_dir.exists():
    run_pilot(task_dir / "meta_train.parquet", model, fixed_dir,
              task_id="garden_03c966", mode="fixed", rounds=2, max_attempts=1,
              group_size=2, business_steps=100, seed=17)
fixed = json.loads((fixed_dir / "pilot.json").read_text())
if fixed["status"] != "completed" or len(fixed["groups"]) != len(recursive["groups"]):
    raise RuntimeError("Fixed control did not finish the matched proposal budget")
comparison = {
    "purpose": "Single-task meta-dev pilot; not final meta-test evidence",
    "training_controls": {"recursive": str(recursive_dir), "fixed_A0": str(fixed_dir),
                          "proposals_each": 4, "total_compute_matched": False},
    "evaluation_task": "toys_7a17d0", "evaluation_proposals_each": 4,
    "evaluation_groups_each": 2, "max_new_tokens_per_proposal": 768,
    "memory_condition": "Fresh separate program databases at start; each method retains its own execution feedback across the same two search groups",
    "learning_during_evaluation": False, "evaluations": {},
    "limitations": "One meta-dev task and one paired seed, unequal upfront Agent-training compute; no causal RSI improvement claim",
}
for label, adapter in [("A0", None), ("A1", updates[0]["path"]), ("A2", updates[1]["path"])]:
    output = root / "artifacts/recursion" / f"eval_{label}"
    if not output.exists():
        run_pilot(task_dir / "meta_dev.parquet", model, output,
                  task_id="toys_7a17d0", mode="evaluate", input_adapter=adapter,
                  rounds=2, max_attempts=1, group_size=2, business_steps=100, seed=17)
    report = json.loads((output / "pilot.json").read_text())
    if report["status"] != "completed" or len(report["groups"]) != 2 or report["lineage"]:
        raise RuntimeError(f"Evaluation budget or weight-freeze violation for {label}")
    checkpoint_ids = {group["checkpoint_id"] for group in report["groups"]}
    if len(checkpoint_ids) != 1:
        raise RuntimeError("Evaluation switched Agent weights")
    if adapter and checkpoint_ids != {updates[int(label[1:]) - 1]["output_checkpoint_id"]}:
        raise RuntimeError("Evaluation did not use the selected saved generation")
    results = [item for group in report["groups"] for item in group["results"]]
    comparison["evaluations"][label] = {
        "path": str(output), "checkpoint_id": next(iter(checkpoint_ids)),
        "valid_candidates": sum(item["success"] for item in results),
        "proposals": len(results),
        "best_search_reward": max(item["reward"] for item in results),
        "best_validation_mrr": max((item["reward"] for item in results if item["success"]), default=None),
        "total_wall_seconds": report["total_wall_seconds"],
    }
    write_json(root / "artifacts/controls/comparison.json", comparison)
    print(json.dumps({"event": "evaluation_completed", "agent": label,
                      **comparison["evaluations"][label]}), flush=True)
