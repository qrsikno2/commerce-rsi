"""Single-card on-policy RLOO sequence policy gradient with execution rewards.

This intentionally small backend is an adaptation, not a reproduction of OpenRSI's
SLIME/Megatron trainer and not GRPO/GSPO. A group comes from one exact current policy.
One optimizer step uses sum(completion log probabilities), no length normalization,
and a leave-one-out reward baseline. Prompt/padding tokens do not receive loss.
"""

import gc
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn import functional as F


LORA_SETTINGS = {
    "r": 8, "lora_alpha": 16, "lora_dropout": 0.0,
    "target_modules": ["q_proj", "v_proj"], "bias": "none", "task_type": "CAUSAL_LM",
}
SAMPLING_SETTINGS = {
    "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
    "typical_p": 1.0, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _base_model_hash_cached(directory, inventory):
    digest = hashlib.sha256()
    for relative, _, _ in inventory:
        digest.update(relative.encode("utf-8"))
        digest.update(_sha256_file(Path(directory) / relative).encode("ascii"))
    return digest.hexdigest()


def base_model_hash(model_dir):
    """Hash local base weights/config/tokenizer, caching only unchanged files."""
    directory = Path(model_dir).resolve()
    paths = sorted(path for path in directory.iterdir() if path.is_file() and (
        path.suffix in {".safetensors", ".json", ".txt", ".model"}
        or path.name.startswith("pytorch_model")))
    if not any(path.suffix == ".safetensors" or path.name.startswith("pytorch_model") for path in paths):
        raise ValueError(f"No local base model weights found: {directory}")
    inventory = tuple((path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in paths)
    return _base_model_hash_cached(str(directory), inventory)


def tensor_state_hash(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        # NumPy has no standard bfloat16 dtype; uint8 preserves the exact bits.
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def adapter_state(model):
    from peft import get_peft_model_state_dict

    return get_peft_model_state_dict(model)


def adapter_hash(model):
    return tensor_state_hash(adapter_state(model))


def checkpoint_identity(base_hash, lora_hash):
    return hashlib.sha256((base_hash + ":" + lora_hash).encode("ascii")).hexdigest()


def rloo_advantages(rewards):
    """Independent-sample leave-one-out baseline, preserving reward scale."""
    if len(rewards) < 2:
        raise ValueError("RLOO requires at least two sampled completions for one prompt")
    values = [float(value) for value in rewards]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Execution rewards must be finite")
    total = sum(values)
    return [value - (total - value) / (len(values) - 1) for value in values]


def completion_logprob_sum(logits, input_ids, prompt_length, chunk_size=32):
    """Causal completion log probability; includes EOS and excludes the prompt."""
    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise ValueError("Expected logits [1, sequence, vocab] and matching token IDs")
    if input_ids.shape[0] != 1 or not 1 <= prompt_length < input_ids.shape[1]:
        raise ValueError("Expected one nonempty prompt and a nonempty completion")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    total = logits.new_zeros((), dtype=torch.float32)
    # The token at position p is predicted from the logits at p-1.
    for token_start in range(prompt_length, input_ids.shape[1], chunk_size):
        token_end = min(token_start + chunk_size, input_ids.shape[1])
        prediction = logits[:, token_start - 1:token_end - 1].float()
        targets = input_ids[:, token_start:token_end]
        total = total - F.cross_entropy(prediction.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
    return total


def validate_group(group, expected_checkpoint_id, vocab_size=None):
    if group.get("schema_version") != 1:
        raise ValueError("Unknown rollout group schema")
    if group.get("checkpoint_id") != expected_checkpoint_id:
        raise ValueError("Off-policy rollout rejected: checkpoint does not match loaded policy")
    if group.get("sampling") != SAMPLING_SETTINGS:
        raise ValueError("Rollout distribution does not match exact policy sampling")
    prompt = group.get("prompt_token_ids")
    completions = group.get("completions")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("Rollout requires one nonempty shared prompt")
    if not isinstance(completions, list) or len(completions) < 2:
        raise ValueError("Rollout requires a group of at least two completions")
    expected_prompt_hash = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
    if group.get("prompt_tokens_sha256") != expected_prompt_hash:
        raise ValueError("Prompt token hash mismatch")
    for completion in completions:
        tokens = completion.get("token_ids")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("Completion tokens must be a nonempty list")
        for value in prompt + tokens:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or (vocab_size is not None and value >= vocab_size):
                raise ValueError("Invalid token ID in rollout")
        if completion.get("prompt_tokens_sha256") != expected_prompt_hash:
            raise ValueError("Completions from different prompts cannot share a RLOO group")
        if completion.get("finish_reason") not in {"eos", "length"}:
            raise ValueError("Unknown completion termination")
    return completions


def _cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class AgentSession:
    def __init__(self, model, tokenizer, model_dir, seed, base_hash, load_seconds):
        self.model = model
        self.tokenizer = tokenizer
        self.model_dir = str(Path(model_dir).resolve())
        self.model_seed = seed
        self.base_model_sha256 = base_hash
        self.adapter_sha256 = adapter_hash(model)
        self.checkpoint_id = checkpoint_identity(base_hash, self.adapter_sha256)
        self.load_seconds = load_seconds
        self.device = next(model.parameters()).device

    def generate_group(self, prompt, count=2, max_new_tokens=768, seed=17, max_prompt_tokens=2048):
        from transformers import GenerationConfig

        if self.model is None:
            raise RuntimeError("Agent session has been released")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be nonempty text")
        if count < 2 or max_new_tokens < 1:
            raise ValueError("At least two completions and a positive token budget are required")
        if adapter_hash(self.model) != self.adapter_sha256:
            raise RuntimeError("Agent weights changed after session identity was assigned")
        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=True, return_tensors="pt")
        if prompt_ids.shape[1] > max_prompt_tokens:
            raise ValueError(f"Prompt has {prompt_ids.shape[1]} tokens; budget {max_prompt_tokens}; no silent truncation")
        context_limit = getattr(self.model.config, "max_position_embeddings", 0)
        if context_limit and prompt_ids.shape[1] + max_new_tokens > context_limit:
            raise ValueError("Prompt plus completion budget exceeds model context")
        prompt_tokens = prompt_ids[0].tolist()
        prompt_hash = hashlib.sha256(json.dumps(prompt_tokens, separators=(",", ":")).encode()).hexdigest()
        group = {
            "schema_version": 1, "checkpoint_id": self.checkpoint_id,
            "base_model_sha256": self.base_model_sha256, "adapter_sha256": self.adapter_sha256,
            "model_seed": self.model_seed, "model_dir": self.model_dir,
            "prompt": prompt, "prompt_token_ids": prompt_tokens, "prompt_tokens_sha256": prompt_hash,
            "sampling": dict(SAMPLING_SETTINGS), "generation_seed": seed,
            "max_new_tokens": max_new_tokens, "lora": dict(LORA_SETTINGS), "completions": [],
        }
        self.model.eval()
        eos_ids = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        _cuda_sync(self.device)
        start = time.perf_counter()
        prompt_ids = prompt_ids.to(self.device)
        # A fresh config prevents checkpoint-specific generation penalties or
        # truncation settings from silently changing the policy distribution.
        generation_config = GenerationConfig(
            **SAMPLING_SETTINGS, max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_ids, use_cache=True,
        )
        with torch.inference_mode():
            for index in range(count):
                torch.manual_seed(seed + index)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(seed + index)
                output = self.model.generate(
                    input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                    generation_config=generation_config,
                )
                tokens = output[0, len(prompt_tokens):].tolist()
                finish_reason = "length"
                for position, token in enumerate(tokens):
                    if token in eos_ids:
                        tokens = tokens[:position + 1]
                        finish_reason = "eos"
                        break
                group["completions"].append({
                    "index": index, "token_ids": tokens,
                    "text": self.tokenizer.decode(tokens, skip_special_tokens=True),
                    "finish_reason": finish_reason, "prompt_tokens_sha256": prompt_hash,
                })
                del output
        _cuda_sync(self.device)
        group["cost"] = {
            "agent_load_seconds": self.load_seconds,
            "agent_generation_wall_seconds": time.perf_counter() - start,
            "prompt_tokens_per_completion": len(prompt_tokens),
            "completion_tokens": sum(len(row["token_ids"]) for row in group["completions"]),
            "truncated_completions": sum(row["finish_reason"] == "length" for row in group["completions"]),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0,
        }
        validate_group(group, self.checkpoint_id, self.model.config.vocab_size)
        return group

    def release(self):
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def load_agent(model_dir, adapter_dir=None, seed=17, device="cuda"):
    """Load local Qwen-compatible BF16 base plus deterministic LoRA or a saved adapter."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    start = time.perf_counter()
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no implicit CPU fallback")
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA are supported")
    if target.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This backend requires native BF16 support")
    base_hash = base_model_hash(model_dir)
    torch.manual_seed(seed)
    if target.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16 if target.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(target)
    if getattr(base_model.config, "attention_dropout", 0.0) != 0:
        raise ValueError("Nonzero attention dropout would change the rollout/update policy")
    if adapter_dir is None:
        # Keep adapter initialization independent of how the base loader consumes
        # RNG while constructing modules before loading their pretrained weights.
        torch.manual_seed(seed)
        if target.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = get_peft_model(base_model, LoraConfig(**LORA_SETTINGS))
    else:
        model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True, local_files_only=True)
        loaded_config = model.peft_config["default"]
        for key in ("r", "lora_alpha", "lora_dropout", "bias"):
            if getattr(loaded_config, key) != LORA_SETTINGS[key]:
                raise ValueError(f"Saved adapter changed frozen LoRA configuration: {key}")
        if set(loaded_config.target_modules) != set(LORA_SETTINGS["target_modules"]):
            raise ValueError("Saved adapter changed LoRA target modules")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0:
            raise ValueError("Nonzero dropout would change the rollout/update policy")
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Only LoRA parameters may be trainable")
    model.eval()
    return AgentSession(model, tokenizer, model_dir, seed, base_hash, time.perf_counter() - start)


def _cpu_optimizer_state(optimizer):
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(move(item) for item in value)
        return value
    return move(optimizer.state_dict())


def update_from_group(model_dir, input_adapter, group, rewards, output_adapter,
                      learning_rate=1e-5, device="cuda", max_grad_norm=1.0):
    """One exact-policy RLOO step, preserving optimizer state between generations.

    Equal rewards produce an explicit no-update result and no new adapter. Rollouts
    generated from any other checkpoint are rejected before optimizer construction.
    """
    destination = Path(output_adapter)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite adapter output: {destination}")
    start = time.perf_counter()
    report = {
        "schema_version": 1, "method": "on_policy_rloo_sequence_policy_gradient",
        "loss_normalization": "sum_completion_logprobs_then_mean_over_group",
        "optimizer_steps": 0, "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "input_adapter": str(input_adapter) if input_adapter else None,
        "rollout_checkpoint_id": group.get("checkpoint_id"),
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "training_tokens": 0, "optimizer_step_wall_seconds": 0.0,
    }
    session = None
    model = None
    optimizer = None
    try:
        if not math.isfinite(learning_rate) or learning_rate <= 0 or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("Invalid optimizer configuration")
        advantages = rloo_advantages(rewards)
        if len(rewards) != len(group.get("completions", [])):
            raise ValueError("Exactly one execution reward is required per completion")
        report["rewards"] = [float(value) for value in rewards]
        report["advantages"] = advantages
        session = load_agent(model_dir, input_adapter, seed=group.get("model_seed", 17), device=device)
        model = session.model
        completions = validate_group(group, session.checkpoint_id, model.config.vocab_size)
        if group.get("base_model_sha256") != session.base_model_sha256 or group.get("adapter_sha256") != session.adapter_sha256:
            raise ValueError("Rollout base/adapter hash does not match reloaded policy")
        if group.get("lora") != LORA_SETTINGS:
            raise ValueError("Rollout LoRA configuration mismatch")
        report["input_checkpoint_id"] = session.checkpoint_id
        report["base_model_sha256"] = session.base_model_sha256
        report["adapter_before_sha256"] = session.adapter_sha256
        report["trainable_parameters"] = sum(value.numel() for value in model.parameters() if value.requires_grad)
        report["total_parameters"] = sum(value.numel() for value in model.parameters())
        report["agent_load_seconds"] = session.load_seconds
        prior_optimizer = Path(input_adapter) / "optimizer.pt" if input_adapter else None
        if prior_optimizer is not None and not prior_optimizer.is_file():
            raise ValueError("Continued RL update requires the input adapter's optimizer.pt; implicit optimizer reset is forbidden")
        if all(value == 0.0 for value in advantages):
            report.update(status="no_update", reason="all_group_rewards_equal", parameters_changed=False)
            return report
        if session.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(session.device)
        model.train()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=learning_rate, weight_decay=0.0)
        if prior_optimizer is not None:
            prior_update = Path(input_adapter) / "update.json"
            metadata = json.loads(prior_update.read_text(encoding="utf-8"))
            if metadata.get("output_checkpoint_id") != session.checkpoint_id:
                raise ValueError("Optimizer lineage does not match input adapter")
            if metadata.get("optimizer_sha256") != _sha256_file(prior_optimizer):
                raise ValueError("Optimizer state SHA256 does not match checkpoint metadata")
            optimizer.load_state_dict(torch.load(prior_optimizer, map_location="cpu", weights_only=True))
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate
            report["optimizer_resumed"] = True
        else:
            report["optimizer_resumed"] = False
        optimizer.zero_grad(set_to_none=True)
        prompt_tokens = group["prompt_token_ids"]
        logprob_sums = []
        loss_total = 0.0
        training_tokens = 0
        _cuda_sync(session.device)
        update_start = time.perf_counter()
        for completion, advantage in zip(completions, advantages):
            token_ids = torch.tensor([prompt_tokens + completion["token_ids"]], dtype=torch.long, device=session.device)
            training_tokens += token_ids.numel()
            report["training_tokens"] = training_tokens
            output = model(input_ids=token_ids, attention_mask=torch.ones_like(token_ids), use_cache=False)
            logprob = completion_logprob_sum(output.logits, token_ids, len(prompt_tokens))
            loss = -advantage * logprob / len(completions)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite policy gradient loss")
            logprob_sums.append(float(logprob.detach().cpu()))
            loss_total += float(loss.detach().cpu())
            loss.backward()
            report["optimizer_step_wall_seconds"] = time.perf_counter() - update_start
            del output, token_ids, loss, logprob
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [value for value in model.parameters() if value.requires_grad], max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        _cuda_sync(session.device)
        report["optimizer_step_wall_seconds"] = time.perf_counter() - update_start
        report["optimizer_steps"] = 1
        report["training_tokens"] = training_tokens
        report["completion_logprob_sums"] = logprob_sums
        report["loss"] = loss_total
        report["gradient_norm_before_clipping"] = float(gradient_norm.detach().cpu())
        report["adapter_after_sha256"] = adapter_hash(model)
        report["parameters_changed"] = report["adapter_after_sha256"] != report["adapter_before_sha256"]
        if not report["parameters_changed"]:
            raise RuntimeError("Nonzero-advantage optimizer step did not change LoRA parameters")
        report["output_checkpoint_id"] = checkpoint_identity(session.base_model_sha256, report["adapter_after_sha256"])
        model.save_pretrained(destination, safe_serialization=True)
        session.tokenizer.save_pretrained(destination)
        torch.save(_cpu_optimizer_state(optimizer), destination / "optimizer.pt")
        report["optimizer_sha256"] = _sha256_file(destination / "optimizer.pt")
        report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(session.device) if session.device.type == "cuda" else 0
        report["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(session.device) if session.device.type == "cuda" else 0
        # Release every live training reference before loading a second copy.
        optimizer = None
        model = None
        session.release()
        session = load_agent(model_dir, destination, seed=group["model_seed"], device=device)
        report["adapter_reloaded_sha256"] = session.adapter_sha256
        report["reload_verified"] = session.adapter_sha256 == report["adapter_after_sha256"]
        if not report["reload_verified"]:
            raise RuntimeError("Saved adapter hash differs after reload")
        report["status"] = "updated"
        return report
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        optimizer = None
        model = None
        if session is not None:
            current_device = getattr(session, "device", None)
            if current_device is not None and current_device.type == "cuda" and torch.cuda.is_initialized():
                report["gpu_peak_allocated_bytes"] = max(report.get("gpu_peak_allocated_bytes", 0), torch.cuda.max_memory_allocated(current_device))
                report["gpu_peak_reserved_bytes"] = max(report.get("gpu_peak_reserved_bytes", 0), torch.cuda.max_memory_reserved(current_device))
            session.release()
        report["agent_update_total_wall_seconds"] = time.perf_counter() - start
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (destination / "update.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
