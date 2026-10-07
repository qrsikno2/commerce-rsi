"""Bounded trusted business runs. Untrusted generated programs are not loaded here."""

import hashlib
import inspect
import json
import os
import platform
import re
import time
import unicodedata
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from commerce_rsi.sampling import CandidateView, LexicalHardSampler, SamplingError, UniformSampler, validate_negatives

from .lexical import BM25
from .metrics import metrics_from_ranks, ranks_from_scores
from .model import BusinessConfig, encode_texts, initialize_model, model_hash


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _queries(path, catalog_ids):
    rows = _rows(path)
    if not rows:
        raise ValueError(f"Empty query split: {path}")
    output = []
    for row in rows:
        clean = {key: row[key] for key in ("qid", "query", "item_id")}
        if any(not isinstance(value, str) or not value.strip() for value in clean.values()):
            raise ValueError("Query fields must be nonempty strings")
        if clean["item_id"] not in catalog_ids:
            raise ValueError(f"Target missing from fixed candidate pool: {clean['item_id']}")
        output.append(clean)
    return output


def _normalize_text(text):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


def known_training_positives(rows, candidates=None):
    """Union observed train targets and duplicate catalog texts, never hidden labels."""
    by_text = {}
    by_qid = {}
    for row in rows:
        normalized = _normalize_text(row["query"])
        by_text.setdefault(normalized, set()).add(row["item_id"])
        by_qid.setdefault(row["qid"], set()).add(row["item_id"])
    result = {}
    for row in rows:
        normalized = _normalize_text(row["query"])
        result.setdefault(row["qid"], set()).update(by_text[normalized] | by_qid[row["qid"]])
    if candidates is not None:
        equivalent_items = {}
        item_texts = {}
        for item_id, text in zip(candidates.item_ids, candidates.texts):
            normalized = _normalize_text(text)
            equivalent_items.setdefault(normalized, set()).add(item_id)
            item_texts[item_id] = normalized
        for qid, targets in result.items():
            expanded = set(targets)
            for item_id in targets:
                expanded.update(equivalent_items[item_texts[item_id]])
            result[qid] = expanded
    return MappingProxyType({key: frozenset(value) for key, value in result.items()})


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _check_deadline(start, max_seconds):
    if time.perf_counter() - start > max_seconds:
        raise TimeoutError(f"Business run exceeded {max_seconds:g}s wall-time budget")


def _evaluate(model, catalog_tokens, query_tokens, target_ids, item_ids, device, start, max_seconds):
    model.eval()
    vectors = []
    ranks = []
    with torch.inference_mode():
        # Recompute every item vector for the current weights; no stale model index.
        for offset in range(0, len(catalog_tokens), 512):
            _check_deadline(start, max_seconds)
            vectors.append(model.item(catalog_tokens[offset:offset + 512].to(device)))
        item_vectors = torch.cat(vectors)
        for offset in range(0, len(query_tokens), 64):
            _check_deadline(start, max_seconds)
            query_vectors = model.query(query_tokens[offset:offset + 64].to(device))
            scores = (query_vectors @ item_vectors.T).cpu().numpy()
            ranks.extend(ranks_from_scores(scores, target_ids[offset:offset + 64], item_ids).tolist())
    return metrics_from_ranks(np.asarray(ranks, dtype=np.int64))


def run_business(prepared_dir, task_id, strategy, output_dir, seed=17, steps=100,
                 batch_size=32, num_negatives=16, device="cuda", config=None,
                 sampler_timeout_seconds=10.0, max_runtime_seconds=600.0,
                 sampler_executor=None, sampler_code_sha256=None, agent_checkpoint=None):
    """Run BM25 or a fixed dual tower using a trusted sampler, recording failures.

    Only reward_validation is evaluated. All strategies share model initialization,
    training-query RNG, candidate pool, optimizer and exact optimization step count.
    This API deliberately does not accept/import arbitrary generated sampler code.
    ``isolated`` invokes a trusted host callback responsible for a real sandbox;
    it receives (batch, candidate_view, model_view, rng, num_negatives). The callback
    must isolate its source code and return only a decoded list of negative IDs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "run.json"
    if report_path.exists() or (output_dir / "checkpoint.pt").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output_dir}")
    config = config or BusinessConfig()
    if isinstance(config, dict):
        config = BusinessConfig(**config)
    start = time.perf_counter()
    training_start = None
    training_end = None
    actual_device = None
    result = {
        "schema_version": 1, "status": "running", "task_id": task_id,
        "strategy": strategy, "seed": seed, "steps": steps,
        "batch_size": batch_size, "num_negatives": num_negatives,
        "config": asdict(config), "device": device,
        "evaluation_split": "reward_validation", "trusted_sampler_only": strategy != "isolated",
        "sampler_execution_mode": "host_isolated_callback" if strategy == "isolated" else "trusted_baseline",
        "agent_checkpoint": agent_checkpoint, "agent_generation_cost": None, "agent_update_cost": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "cost": {"sampler_seconds": 0.0, "business_training_wall_seconds": 0.0,
                 "business_training_cuda_event_seconds": 0.0,
                 "gpu_peak_allocated_bytes": 0, "gpu_peak_reserved_bytes": 0},
        "completed_steps": 0, "training_tokens": 0,
    }
    try:
        if strategy not in {"uniform", "lexical_hard", "bm25", "isolated"}:
            raise ValueError("Unknown strategy; expected a trusted baseline or isolated callback")
        if strategy == "isolated":
            if not callable(sampler_executor):
                raise ValueError("Isolated strategy requires a trusted host sampler_executor")
            if not isinstance(sampler_code_sha256, str) or len(sampler_code_sha256) != 64 or any(c not in "0123456789abcdef" for c in sampler_code_sha256):
                raise ValueError("Isolated strategy requires the sampler source SHA256")
        elif sampler_executor is not None or sampler_code_sha256 is not None:
            raise ValueError("Baseline strategies do not accept sampler replacement")
        if steps < 1 or batch_size < 1 or num_negatives < 1:
            raise ValueError("steps, batch_size and num_negatives must be positive")
        if sampler_timeout_seconds <= 0 or max_runtime_seconds <= 0:
            raise ValueError("Runtime budgets must be positive")
        if Path(task_id).name != task_id or task_id in {".", ".."}:
            raise ValueError("task_id must be one directory name")
        prepared_dir = Path(prepared_dir)
        paths = {
            "catalog": prepared_dir / "public" / "catalog.jsonl",
            "train": prepared_dir / "tasks" / task_id / "train.jsonl",
            "reward_validation": prepared_dir / "tasks" / task_id / "reward_validation.jsonl",
        }
        manifest = prepared_dir / "manifest.json"
        protocol_manifest = None
        if manifest.exists():
            protocol_manifest = json.loads(manifest.read_text(encoding="utf-8"))
            task_entries = [row for row in protocol_manifest["tasks"] if row["task_id"] == task_id]
            if len(task_entries) != 1:
                raise ValueError("Task must occur exactly once in the prepared manifest")
            if task_entries[0]["outer_split"] not in {"meta_train", "meta_dev"}:
                raise ValueError("Business development runner cannot access meta-test tasks")
            result["outer_split"] = task_entries[0]["outer_split"]
            result["data_protocol"] = protocol_manifest.get("protocol")
            result["dataset_revision"] = protocol_manifest.get("dataset_revision")
            result["prepared_manifest_sha256"] = sha256_file(manifest)
        result["data_sha256"] = {name: sha256_file(path) for name, path in paths.items()}
        if protocol_manifest is not None:
            if protocol_manifest["catalog_sha256"] != result["data_sha256"]["catalog"]:
                raise ValueError("Fixed candidate pool SHA256 differs from prepared manifest")
            expected_queries = {row["path"]: row["sha256"] for row in protocol_manifest["query_files"]}
            if len(expected_queries) != len(protocol_manifest["query_files"]):
                raise ValueError("Duplicate query-file paths in prepared manifest")
            for split_name in ("train", "reward_validation"):
                relative = str(paths[split_name].relative_to(prepared_dir))
                if expected_queries.get(relative) != result["data_sha256"][split_name]:
                    raise ValueError(f"Query split SHA256 differs from prepared manifest: {split_name}")
        result["prepared_manifest_verified"] = protocol_manifest is not None
        code_paths = sorted(Path(__file__).parent.glob("*.py")) + sorted(Path(inspect.getfile(UniformSampler)).parent.glob("*.py"))
        result["business_code_sha256"] = hashlib.sha256("".join(sha256_file(p) for p in code_paths).encode()).hexdigest()
        candidates = CandidateView.from_rows(_rows(paths["catalog"]))
        if protocol_manifest is not None and len(candidates.item_ids) != protocol_manifest["candidate_count"]:
            raise ValueError("Candidate count differs from prepared manifest")
        item_index = candidates.index
        train = _queries(paths["train"], item_index)
        validation = _queries(paths["reward_validation"], item_index)
        result["num_candidates"] = len(candidates.item_ids)
        result["num_train_queries"] = len(train)
        result["num_validation_queries"] = len(validation)
        targets = [row["item_id"] for row in validation]
        if strategy == "bm25":
            result["execution_device"] = "cpu"
            lexical = BM25(candidates.texts)
            ranks = []
            for row in validation:
                _check_deadline(start, max_runtime_seconds)
                ranks.extend(ranks_from_scores(lexical.score(row["query"])[None, :], [row["item_id"]], candidates.item_ids))
            result["metrics"] = metrics_from_ranks(np.asarray(ranks, dtype=np.int64))
            result["business_initialization_sha256"] = None
            result["sampler_code_sha256"] = None
            result["steps"] = 0
            result["status"] = "completed"
            return result

        actual_device = torch.device(device)
        result["execution_device"] = str(actual_device)
        if actual_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but unavailable; no implicit CPU fallback")
        if actual_device.type not in {"cuda", "cpu"}:
            raise ValueError("Only CPU and CUDA business devices are supported")
        torch.use_deterministic_algorithms(True)
        if actual_device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats(actual_device)
            result["gpu_name"] = torch.cuda.get_device_name(actual_device)
        model = initialize_model(config, seed)
        result["business_initialization_sha256"] = model_hash(model)
        model.to(actual_device)
        train_tokens = encode_texts([row["query"] for row in train], config.query_max_tokens, config.vocab_size)
        validation_tokens = encode_texts([row["query"] for row in validation], config.query_max_tokens, config.vocab_size)
        catalog_tokens = encode_texts(candidates.texts, config.item_max_tokens, config.vocab_size)
        tokenization_seconds = time.perf_counter() - start
        result["initial_metrics"] = _evaluate(model, catalog_tokens, validation_tokens, targets, candidates.item_ids, actual_device, start, max_runtime_seconds)
        positives = known_training_positives(train, candidates)
        sampler_start = time.perf_counter()
        if strategy == "isolated":
            sample_negatives = sampler_executor
            result["sampler_code_sha256"] = sampler_code_sha256
        else:
            sampler = UniformSampler(positives) if strategy == "uniform" else LexicalHardSampler(candidates, positives)
            sample_negatives = sampler.sample_negatives
            result["sampler_code_sha256"] = hashlib.sha256(inspect.getsource(type(sampler)).encode()).hexdigest()
        sampler_seconds = time.perf_counter() - sampler_start
        result["cost"]["sampler_seconds"] = sampler_seconds
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        batch_rng = np.random.default_rng(seed)
        sampling_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        schedule_hash = hashlib.sha256()
        loss_history = []
        token_count = 0
        model.train()
        _sync(actual_device)
        training_start = time.perf_counter()
        for step in range(steps):
            _check_deadline(start, max_runtime_seconds)
            indices = batch_rng.integers(0, len(train), size=batch_size)
            schedule_hash.update(indices.astype("<i8").tobytes())
            # Immutable rows and views document the callable interface. Only trusted
            # baselines run in this process; immutability does not imply isolation.
            batch = [MappingProxyType({**train[int(index)],
                                       "known_positive_item_ids": tuple(sorted(positives[train[int(index)]["qid"]]))})
                     for index in indices]
            model_view = MappingProxyType({"step": step, "total_steps": steps})
            sample_start = time.perf_counter()
            negatives = sample_negatives(batch, candidates, model_view, sampling_rng, num_negatives)
            elapsed = time.perf_counter() - sample_start
            sampler_seconds += elapsed
            result["cost"]["sampler_seconds"] = sampler_seconds
            if elapsed > sampler_timeout_seconds:
                raise SamplingError(f"Sampler exceeded time budget: {elapsed:.3f}s")
            validate_negatives(negatives, batch, candidates, num_negatives, positives)
            if step == 0:
                with (output_dir / "negative_audit.jsonl").open("w", encoding="utf-8") as stream:
                    for row, selected in list(zip(batch, negatives))[:5]:
                        stream.write(json.dumps({"qid": row["qid"], "query": row["query"], "positive_item_id": row["item_id"], "negative_item_ids": selected}, ensure_ascii=False) + "\n")
            item_indices = torch.tensor([[item_index[row["item_id"]]] + [item_index[value] for value in selected]
                                         for row, selected in zip(batch, negatives)], dtype=torch.long)
            query_input = train_tokens[indices].to(actual_device)
            item_input = catalog_tokens[item_indices].to(actual_device)
            token_count += int(torch.count_nonzero(query_input).item() + torch.count_nonzero(item_input).item())
            result["training_tokens"] = token_count
            if actual_device.type == "cuda":
                gpu_start = torch.cuda.Event(enable_timing=True)
                gpu_end = torch.cuda.Event(enable_timing=True)
                gpu_start.record()
            optimizer.zero_grad(set_to_none=True)
            logits = model(query_input, item_input)
            loss = F.cross_entropy(logits, torch.zeros(batch_size, dtype=torch.long, device=actual_device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            if actual_device.type == "cuda":
                gpu_end.record()
                gpu_end.synchronize()
                result["cost"]["business_training_cuda_event_seconds"] += gpu_start.elapsed_time(gpu_end) / 1000.0
            loss_history.append(float(loss.detach().cpu()))
            result["loss_history"] = loss_history
            result["completed_steps"] = step + 1
        _sync(actual_device)
        training_end = time.perf_counter()
        training_seconds = training_end - training_start
        result["batch_schedule_sha256"] = schedule_hash.hexdigest()
        result["loss_history"] = loss_history
        result["training_tokens"] = token_count
        result["metrics"] = _evaluate(model, catalog_tokens, validation_tokens, targets, candidates.item_ids, actual_device, start, max_runtime_seconds)
        result["business_final_sha256"] = model_hash(model)
        result["parameters_changed"] = result["business_initialization_sha256"] != result["business_final_sha256"]
        if not result["parameters_changed"]:
            raise RuntimeError("Business training completed without parameter changes")
        torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "config": asdict(config), "seed": seed, "steps": steps,
                    "business_initialization_sha256": result["business_initialization_sha256"],
                    "data_sha256": result["data_sha256"]}, output_dir / "checkpoint.pt")
        result["checkpoint_sha256"] = sha256_file(output_dir / "checkpoint.pt")
        result["cost"].update({
            "preparation_seconds": tokenization_seconds,
            "business_training_wall_seconds": training_seconds,
            "sampler_seconds": sampler_seconds,
            "cuda_training_phase_wall_seconds": training_seconds if actual_device.type == "cuda" else None,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(actual_device) if actual_device.type == "cuda" else 0,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(actual_device) if actual_device.type == "cuda" else 0,
        })
        result["status"] = "completed"
        return result
    except Exception as error:
        result["status"] = "failed"
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if training_start is not None:
            result["cost"]["business_training_wall_seconds"] = (training_end or time.perf_counter()) - training_start
        if actual_device is not None and actual_device.type == "cuda" and torch.cuda.is_initialized():
            result["cost"]["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(actual_device)
            result["cost"]["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(actual_device)
        result["cost"]["business_total_wall_seconds"] = time.perf_counter() - start
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
