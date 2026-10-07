"""Real OpenRSI search, sandboxed business feedback, and serial LoRA updates.

This is a control-path pilot. Meta-test is intentionally unavailable here; two
successful parameter updates alone do not establish recursive capability gains.
"""

import argparse
import ast
import hashlib
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

from commerce_rsi.agent_training.backend import load_agent, update_from_group
from commerce_rsi.data.download import sha256_file
from commerce_rsi.openrsi_adapter.sandbox import SandboxedSampler
from commerce_rsi.openrsi_adapter.upstream import SamplerSearch, read_task_package, upstream_manifest
from commerce_rsi.retrieval.model import BusinessConfig
from commerce_rsi.retrieval.runner import run_business


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def extract_source(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if len(blocks) > 1:
        raise ValueError("Expected one complete Python code block")
    source = blocks[0].strip() if blocks else text.strip()
    tree = ast.parse(source)
    if not any(isinstance(node, ast.FunctionDef) and node.name == "sample_negatives" for node in tree.body):
        raise ValueError("Missing sample_negatives function")
    return source + "\n"


def run_group(record, search, model_dir, adapter, output, *, generation_seed,
              model_seed, business_seed, business_steps, group_size, max_new_tokens):
    output.mkdir(parents=True, exist_ok=False)
    metadata = record["metadata"]
    request = search.next_prompt(metadata["uuid"], metadata["task_name"],
                                 task_description=metadata["task_description"],
                                 data_description=metadata["data_description"])
    # Prompts are native upstream selections with our injected sampler renderer.
    prompt = "\n\n".join(message["content"] for message in request.messages)
    prompt += "\nKeep the implementation concise. Return one complete Python code block, without a usage example."
    write_json(output / "search_request.json", asdict(request))
    agent = load_agent(model_dir, adapter, seed=model_seed)
    try:
        group = agent.generate_group(prompt, count=group_size, max_new_tokens=max_new_tokens,
                                     seed=generation_seed, max_prompt_tokens=4096)
    finally:
        agent.release()
    write_json(output / "rollout.json", group)
    results = []
    prepared = Path(metadata["data_dir"]).parent.parent
    for completion in group["completions"]:
        start = time.perf_counter()
        source = ""
        result = {"completion_index": completion["index"], "success": False, "reward": 0.0,
                  "checkpoint_id": group["checkpoint_id"], "task_id": metadata["task_name"],
                  "primary_metric": "mrr", "reward_split": "reward_validation"}
        candidate_dir = output / f"candidate_{completion['index']}"
        try:
            if completion["finish_reason"] == "length":
                raise ValueError("Generation hit token budget; truncated candidate rejected")
            source = extract_source(completion["text"])
            candidate_dir.mkdir()
            (candidate_dir / "sampler.py").write_text(source)
            source_hash = hashlib.sha256(source.encode()).hexdigest()
            sampler = SandboxedSampler(source)
            run = run_business(prepared, metadata["task_name"], "isolated", candidate_dir / "business",
                               seed=business_seed, steps=business_steps, config=BusinessConfig(shared_towers=True),
                               sampler_executor=sampler.sample_negatives, sampler_code_sha256=source_hash,
                               agent_checkpoint=group["checkpoint_id"], max_runtime_seconds=600)
            result.update(success=True, reward=run["metrics"]["mrr"], metrics=run["metrics"],
                          business_cost=run["cost"], sampler_code_sha256=source_hash,
                          business_initialization_sha256=run["business_initialization_sha256"])
            feedback = json.dumps({"metrics": run["metrics"], "cost": run["cost"]})
        except Exception as exc:
            feedback = f"{type(exc).__name__}: {exc}"
            result["failure"] = feedback
            run_path = candidate_dir / "business" / "run.json"
            if run_path.exists():
                failed = json.loads(run_path.read_text())
                result["business_cost"] = failed.get("cost", {})
        result["execution_wall_seconds"] = time.perf_counter() - start
        program_id = search.record(request, source, score=result["reward"] if result["success"] else None,
                                   success=result["success"], execution_seconds=result["execution_wall_seconds"],
                                   feedback=feedback, raw_response=completion["text"], metadata={
                                       "checkpoint_id": group["checkpoint_id"], "primary_metric": "mrr",
                                       "prepared_manifest_sha256": metadata["prepared_manifest_sha256"],
                                       "rollout_path": str((output / "rollout.json").resolve()),
                                       "completion_index": completion["index"]})
        result["upstream_program_id"] = program_id
        results.append(result)
        write_json(output / "executions.json", results)
        print(json.dumps({"event": "execution", **result}), flush=True)
    group_report = {"generation_cost": group["cost"], "checkpoint_id": group["checkpoint_id"],
                    "results": results, "rewards": [row["reward"] for row in results]}
    write_json(output / "group_report.json", group_report)
    return group, group_report


def run_pilot(package, model_dir, output, *, task_id, mode="recursive", input_adapter=None,
              rounds=2, max_attempts=3, group_size=2, business_steps=100,
              max_new_tokens=768, seed=17):
    if mode not in {"recursive", "fixed", "evaluate"}:
        raise ValueError("Unknown pilot mode")
    if not 1 <= rounds <= 3 or not 1 <= max_attempts <= 3 or not 2 <= group_size <= 4:
        raise ValueError("Pilot budget exceeds the bounded controller limits")
    records = read_task_package(package)
    matches = [row for row in records if row["metadata"]["task_name"] == task_id]
    if len(matches) != 1:
        raise ValueError("Task must occur once in the selected package")
    record = matches[0]
    role = record["metadata"]["outer_split"]
    if mode in {"recursive", "fixed"} and role != "meta_train":
        raise ValueError("Agent training/search control requires meta_train")
    if mode == "evaluate" and role != "meta_dev":
        raise ValueError("Pilot independent evaluation requires meta_dev")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new pilot output directory; checkpoints are never overwritten")
    output.mkdir(parents=True)
    adapter = Path(input_adapter).resolve() if input_adapter else None
    manifest = {"purpose": "control-path pilot; no final meta-test or RSI improvement claim",
                "mode": mode, "task_id": task_id, "outer_split": role,
                "package_sha256": sha256_file(Path(package)), "upstream": upstream_manifest(),
                "rounds": rounds, "max_attempts_per_round": max_attempts,
                "group_size": group_size, "business_steps": business_steps,
                "business_config": asdict(BusinessConfig(shared_towers=True)),
                "max_new_tokens": max_new_tokens, "seed": seed,
                "primary_metric": "untruncated MRR on reward_validation",
                "reward_failure": 0.0, "method": "on_policy_RLOO_single_step_per_successful_round",
                "initial_adapter": str(adapter) if adapter else None, "lineage": [], "groups": [],
                "status": "running"}
    write_json(output / "pilot.json", manifest)
    start = time.perf_counter()
    try:
        with SamplerSearch(output / "programs.sqlite", seed=seed) as search:
            for round_index in range(rounds):
                updated = False
                for attempt in range(max_attempts):
                    group_dir = output / f"round_{round_index}" / f"attempt_{attempt}"
                    group, report = run_group(record, search, model_dir, adapter, group_dir,
                                              generation_seed=seed + 1000 * round_index + 100 * attempt,
                                              model_seed=seed, business_seed=seed, business_steps=business_steps,
                                              group_size=group_size, max_new_tokens=max_new_tokens)
                    manifest["groups"].append({"path": str(group_dir), **report})
                    write_json(output / "pilot.json", manifest)
                    if mode != "recursive":
                        continue
                    next_adapter = output / f"A{round_index + 1}_attempt{attempt}"
                    update = update_from_group(model_dir, adapter, group, report["rewards"], next_adapter)
                    manifest["lineage"].append({"round": round_index, "attempt": attempt,
                                                "path": str(next_adapter), **update})
                    write_json(output / "pilot.json", manifest)
                    print(json.dumps({"event": "agent_update", "round": round_index,
                                      "status": update["status"],
                                      "parameters_changed": update.get("parameters_changed"),
                                      "reload_verified": update.get("reload_verified")}), flush=True)
                    if update["status"] == "updated":
                        adapter = next_adapter.resolve()
                        updated = True
                        break
                if mode == "recursive" and not updated:
                    raise RuntimeError("No differential execution rewards within the bounded round; no new generation claimed")
        manifest["status"] = "completed"
        manifest["final_adapter"] = str(adapter) if adapter else None
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["total_wall_seconds"] = time.perf_counter() - start
        write_json(output / "pilot.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--mode", choices=("recursive", "fixed", "evaluate"), default="recursive")
    parser.add_argument("--input-adapter", type=Path)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--business-steps", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    run_pilot(args.package, args.model, args.output, task_id=args.task_id, mode=args.mode,
              input_adapter=args.input_adapter, rounds=args.rounds, max_attempts=args.max_attempts,
              group_size=args.group_size, business_steps=args.business_steps,
              max_new_tokens=args.max_new_tokens, seed=args.seed)


if __name__ == "__main__":
    main()
