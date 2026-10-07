"""Use pinned OpenRSI program evolution with a constrained sampler prompt.

The program database, operator selection, parent utilities, and experience cards
are imported from the actual checkout, not reimplemented here. GPU training and
execution rewards belong to Commerce-RSI's separately documented backend.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


UPSTREAM_COMMIT = "71ae803a035d5e3b78c19fa49ed9f67d0550cbaa"
TASK_EXECUTOR = "commerce-rsi-sampler-only-v1"
PROMPT_SCHEMA_VERSION = 2
_SELECTION_LOCK = threading.Lock()

SAMPLER_SYSTEM_PROMPT = """You write negative-sampling functions for a fixed product retrieval experiment.
Return one Python code block defining:
def sample_negatives(batch, candidate_view, model_view, rng, num_negatives):
Return a list of lists of product ID strings, one list per query, with exactly
num_negatives unique IDs per query. IDs must exist in candidate_view['item_ids'].
Exclude each query's item_id and every ID in its known_positive_ids list.
batch is a LIST OF DICTIONARIES, never a list of tuples. Each row has exactly
qid, query, item_id, known_positive_ids. Access row["item_id"], row["query"], etc.
Do not unpack a row as qid, query, item_id, known_positive_ids: that unpacks keys.
Illustrative batch schema:
[{"qid":"q0","query":"example request","item_id":"p0","known_positive_ids":["p0"]}]
candidate_view has aligned item_ids, texts, categories lists of public products.
candidate_view is a dictionary: access candidate_view["item_ids"], not attributes.
Every catalog ID is available; texts are truncated to 256 characters and categories
to 128 characters. model_view contains the available step/total_steps/progress.
rng is Python random.Random: use rng.sample, rng.random, rng.choices, or rng.randrange.
Use the supplied rng for random choices; do not use global random state.
Use pure Python; math/random/json are preloaded. No numpy, pandas or torch imports.
Return JSON-compatible nested lists, not a dictionary, generator, tuple or array.
Illustrative output shape for one query with num_negatives=2: [["p1","p2"]].
The example IDs are placeholders; use actual IDs from this call's candidate_view.
No files,
network, subprocesses, external packages, GPU access, or persistence are available.
The trusted host fixes model, loss, optimizer, initialization, training budget,
candidate catalog, and evaluation. Your function only chooses training negatives.
Feedback is host-computed reward-validation MRR (untruncated reciprocal rank).
MRR@10, Hit@10 and NDCG@10 are additional reporting metrics. You cannot access held-out
answers, change training or scoring, or supply a self-reported score.
Explore a sampling strategy based on the allowed inputs and execution feedback.
"""


def sampler_prompt_builder(**kwargs: Any) -> tuple[str, str]:
    """Injected into upstream AIRAInferenceEvoSearch; accepts its native kwargs."""
    mode = kwargs["mode"]
    parts = [f"Operator: {mode}", str(kwargs.get("task_description", "")),
             str(kwargs.get("data_description", ""))]
    for label, parent in (("Parent", kwargs.get("parent_program")),
                          ("Second parent", kwargs.get("secondary_parent_program"))):
        if parent is not None:
            feedback = {
                "status_code": parent.status_code,
                "reward_validation_mrr": parent.score,
                "reward": parent.reward,
                "execution_feedback": parent.metadata.get("clear_run_log", ""),
            }
            parts.extend([f"{label} code:\n```python\n{parent.code}\n```",
                          f"{label} host feedback:\n{json.dumps(feedback, ensure_ascii=False)}"])
    if kwargs.get("memory"):
        parts.append("Upstream experience memory:\n" + str(kwargs["memory"]))
    parts.append("Provide the complete sample_negatives function, including any helper functions.")
    return SAMPLER_SYSTEM_PROMPT, "\n\n".join(part for part in parts if part)


def default_upstream_root() -> Path:
    return Path(__file__).resolve().parents[3] / "third_party" / "OpenRSI"


@lru_cache(maxsize=4)
def load_upstream(root: str | None = None):
    checkout = Path(root).resolve() if root is not None else default_upstream_root()
    result = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True, timeout=10)
    if result.stdout.strip() != UPSTREAM_COMMIT:
        raise RuntimeError(f"OpenRSI revision mismatch; expected {UPSTREAM_COMMIT}")
    directory = checkout / "OpenMLE-ERL" / "RL"

    def load_module(name: str, path: Path):
        if name in sys.modules:
            module = sys.modules[name]
            if Path(module.__file__).resolve() != path.resolve():
                raise RuntimeError(f"Unexpected existing module {name}; refusing mixed upstream versions")
            return module
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load upstream module {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module

    experience = load_module("airaevo_experience", directory / "airaevo_experience.py")
    database = load_module("commerce_rsi_openrsi_program_database", directory / "program_database.py")
    return database, experience


def upstream_manifest(root: str | None = None) -> dict[str, Any]:
    checkout = Path(root).resolve() if root is not None else default_upstream_root()
    load_upstream(str(checkout))
    files = ("OpenMLE-ERL/RL/program_database.py", "OpenMLE-ERL/RL/airaevo_experience.py")
    return {"source": "https://github.com/FrontisAI/OpenRSI", "commit": UPSTREAM_COMMIT,
            "branch": "commerce-rsi-adapter",
            "sampler_prompt_schema_version": PROMPT_SCHEMA_VERSION,
            "file_sha256": {path: hashlib.sha256((checkout / path).read_bytes()).hexdigest()
                            for path in files}}


def export_task_package(prepared_dir: str | Path, output: str | Path,
                        outer_split: str) -> dict[str, Any]:
    """Export upstream prompt/metadata Parquet for the custom sampler backend.

    This input selection does not make stock Gym's full-program sandbox sampler
    compatible. Final/meta-test export is unavailable before protocol freezing.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if outer_split not in {"meta_train", "meta_dev"}:
        raise ValueError("Pilot task export permits only meta_train or meta_dev")
    prepared = Path(prepared_dir).resolve()
    manifest_bytes = (prepared / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    rows = []
    for task in manifest["tasks"]:
        if task["outer_split"] != outer_split:
            continue
        task_name = task["task_id"]
        data_dir = prepared / "tasks" / task_name
        if not (data_dir / "train.jsonl").is_file() or not (data_dir / "reward_validation.jsonl").is_file():
            raise ValueError(f"Incomplete prepared task: {task_name}")
        task_description = f"Find an effective training negative sampler for {task['category']} product retrieval."
        data_description = (
            "The trusted host supplies training queries and public product views by JSON. "
            "Primary feedback is reward-validation MRR (untruncated reciprocal rank). "
            "All final-evaluation query answers remain private."
        )
        system, user = sampler_prompt_builder(mode="draft", task_description=task_description,
                                              data_description=data_description)
        metadata = {
            "task_name": task_name, "task": "commerce-negative-sampling", "cpu_gpu": "gpu",
            "source": "McAuley-Lab/Amazon-C4/custom-protocol", "modality": "text",
            "higher_is_better": True, "theoretical_min": 0.0, "theoretical_max": 1.0,
            "leaderboard_min": None, "leaderboard_max": None,
            "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, manifest_sha256 + "/" + task_name)),
            "data_dir": str(data_dir), "task_description": task_description,
            "data_description": data_description, "category": task["category"],
            "outer_split": outer_split, "executor": TASK_EXECUTOR,
            "openrsi_commit": UPSTREAM_COMMIT, "protocol": manifest["protocol"],
            "prepared_manifest_sha256": manifest_sha256,
            "catalog_sha256": manifest["catalog_sha256"],
            "primary_metric": "mrr",
            "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        }
        rows.append({"prompt": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}], "metadata": metadata})
    if not rows:
        raise ValueError(f"No tasks for {outer_split}")
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), target)
    return {"path": str(target), "rows": len(rows), "outer_split": outer_split,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "executor": TASK_EXECUTOR}


def read_task_package(path: str | Path) -> list[dict[str, Any]]:
    """Read the actual constrained training/development task input records."""
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError("Task package is empty")
    seen = set()
    splits = set()
    for row in rows:
        if set(row) != {"prompt", "metadata"} or not isinstance(row["metadata"], dict):
            raise ValueError("Expected upstream prompt/metadata task records")
        metadata = row["metadata"]
        if metadata.get("executor") != TASK_EXECUTOR or metadata.get("openrsi_commit") != UPSTREAM_COMMIT:
            raise ValueError("Task executor or upstream version mismatch")
        if metadata.get("primary_metric") != "mrr":
            raise ValueError("Task primary metric must be untruncated MRR")
        if metadata.get("prompt_schema_version") != PROMPT_SCHEMA_VERSION:
            raise ValueError("Task prompt schema version mismatch; export a fresh package")
        if metadata.get("outer_split") not in {"meta_train", "meta_dev"}:
            raise ValueError("Final/meta-test package is locked during the pilot")
        splits.add(metadata["outer_split"])
        if metadata["uuid"] in seen:
            raise ValueError("Duplicate task UUID")
        seen.add(metadata["uuid"])
        if row["prompt"][0] != {"role": "system", "content": SAMPLER_SYSTEM_PROMPT}:
            raise ValueError("Task is missing the constrained sampler system prompt")
        data_dir = Path(metadata["data_dir"])
        if not data_dir.is_absolute() or data_dir.name != metadata["task_name"] or data_dir.parent.name != "tasks":
            raise ValueError("Unexpected task data_dir")
        manifest_path = data_dir.parent.parent / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if manifest_sha256 != metadata["prepared_manifest_sha256"]:
            raise ValueError("Prepared manifest changed since task export")
        manifest = json.loads(manifest_bytes)
        authoritative_tasks = [task for task in manifest["tasks"]
                               if task["task_id"] == metadata["task_name"]]
        if len(authoritative_tasks) != 1:
            raise ValueError("Task name is not uniquely present in the authoritative manifest")
        authoritative_task = authoritative_tasks[0]
        for key in ("outer_split", "category"):
            if metadata.get(key) != authoritative_task[key]:
                raise ValueError(f"Task {key} differs from the authoritative manifest")
        for key in ("catalog_sha256", "protocol"):
            if metadata.get(key) != manifest[key]:
                raise ValueError(f"Task {key} differs from the authoritative manifest")
        expected_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, manifest_sha256 + "/" + authoritative_task["task_id"]))
        if metadata["uuid"] != expected_uuid:
            raise ValueError("Task UUID differs from the authoritative manifest identity")
        expected_dir = manifest_path.parent.resolve() / "tasks" / authoritative_task["task_id"]
        if data_dir.resolve(strict=True) != expected_dir or data_dir != expected_dir:
            raise ValueError("Task data_dir must resolve exactly inside prepared/tasks")
        for filename in ("train.jsonl", "reward_validation.jsonl"):
            if not (expected_dir / filename).is_file():
                raise ValueError("Task data_dir is missing a required training/validation input")
    if len(splits) != 1:
        raise ValueError("Training and development task selections must remain separate")
    return rows


@dataclass(frozen=True)
class SearchRequest:
    task_id: str
    task_name: str
    operator: str
    messages: list[dict[str, str]]
    parent_ids: tuple[int, ...]
    selection_metadata: dict[str, Any]


class SamplerSearch:
    """Sequential host bridge to the actual upstream search/database objects."""

    def __init__(self, database_path: str | Path, *, seed: int = 0,
                 policy: str = "rl_mixed", upstream_root: str | None = None) -> None:
        self.modules, self.experience = load_upstream(upstream_root)
        # Retain failures and full lineage; upstream max_per_task > 0 prunes them.
        self.database = self.modules.ProgramDatabase(str(database_path), max_per_task=0)
        self.search = self.modules.AIRAInferenceEvoSearch(
            policy=policy, execution_timeout=5, prompt_builder=sampler_prompt_builder)
        self.rng = random.Random(seed)

    def next_prompt(self, task_id: str, task_name: str, *, task_description: str = "",
                    data_description: str = "") -> SearchRequest:
        # Upstream uses module-global random. Preserve the caller state while
        # making this search's parent/operator selection independently replayable.
        with _SELECTION_LOCK:
            previous_state = random.getstate()
            random.setstate(self.rng.getstate())
            try:
                prompts, parent, operator, secondary = self.search.select(
                    self.database, task_id, task_description, data_description,
                    SAMPLER_SYSTEM_PROMPT, "Write a sampler.", "/public/sampler-input",
                    task_name=task_name, evaluation=False,
                )
                self.rng.setstate(random.getstate())
            finally:
                random.setstate(previous_state)
        assert prompts is not None
        return SearchRequest(
            task_id=task_id, task_name=task_name, operator=operator,
            messages=[{"role": "system", "content": prompts[0]},
                      {"role": "user", "content": prompts[1]}],
            parent_ids=tuple(node.id for node in (parent, secondary) if node is not None),
            selection_metadata=dict(self.search.last_selection_metadata),
        )

    def record(self, request: SearchRequest, source: str, *, score: float | None,
               success: bool, execution_seconds: float = 0.0, feedback: str = "",
               raw_response: str = "", metadata: dict[str, Any] | None = None) -> int:
        if success and (score is None or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Successful execution needs finite untruncated MRR in [0, 1]")
        if not success and score is not None:
            raise ValueError("Failed execution must have score=None")
        reward = float(score) if success else 0.0
        details = dict(metadata or {})
        details.update(request.selection_metadata)
        details.update({"status": "valid" if success else "failed",
                        "clear_run_log": feedback, "parent_ids": list(request.parent_ids),
                        "metric_static_base_reward": reward,
                        "primary_metric": "mrr", "reward_validation_mrr": score,
                        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
                        "reward_split": "reward-validation",
                        "code_sha256": hashlib.sha256(source.encode()).hexdigest()})
        parent = self.database.get_by_id(request.parent_ids[0]) if request.parent_ids else None
        program = self.modules.Program(
            task_id=request.task_id, task_name=request.task_name,
            status_code=200 if success else 400, code=source, score=score,
            running_time=execution_seconds, reward=reward, base_reward=reward,
            parent_id=request.parent_ids[0] if request.parent_ids else None,
            parent_code=parent.code if parent is not None else "",
            generation_mode=request.operator, raw_text=raw_response,
            payload={"status": "valid" if success else "failed",
                     "result": {"score": score, "run_log": feedback}}, metadata=details,
        )
        program.id = self.database.add(program)
        card = self.experience.build_experience_card(
            program, all_programs=self.database.get_all(request.task_name), lower_is_better=False)
        self.database.merge_program_metadata(program.id, {"airaevo_experience_card": card})
        return program.id

    def close(self) -> None:
        self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
