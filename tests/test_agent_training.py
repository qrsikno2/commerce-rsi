import hashlib
import json
import math
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from commerce_rsi.agent_training.backend import (
    LORA_SETTINGS, SAMPLING_SETTINGS, checkpoint_identity, completion_logprob_sum,
    load_agent, rloo_advantages, tensor_state_hash, update_from_group, validate_group,
)


def group_fixture():
    prompt_tokens = [1, 2]
    prompt_hash = hashlib.sha256(json.dumps(prompt_tokens, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": 1, "checkpoint_id": "checkpoint-a0", "model_seed": 17,
        "base_model_sha256": "base-hash", "adapter_sha256": "adapter-hash",
        "sampling": dict(SAMPLING_SETTINGS), "lora": dict(LORA_SETTINGS),
        "prompt_token_ids": prompt_tokens, "prompt_tokens_sha256": prompt_hash,
        "completions": [
            {"token_ids": [3, 0], "finish_reason": "eos", "prompt_tokens_sha256": prompt_hash},
            {"token_ids": [2, 3], "finish_reason": "length", "prompt_tokens_sha256": prompt_hash},
        ],
    }


class AgentTrainingTests(unittest.TestCase):
    def test_rloo_matches_analytic_leave_one_out(self):
        self.assertEqual(rloo_advantages([0.0, 1.0]), [-1.0, 1.0])
        self.assertEqual(rloo_advantages([1.0, 2.0, 3.0]), [-1.5, 0.0, 1.5])
        self.assertEqual(rloo_advantages([0.2, 0.2]), [0.0, 0.0])
        for rewards in ([1.0], [1.0, float("nan")]):
            with self.assertRaises(ValueError):
                rloo_advantages(rewards)

    def test_causal_mask_counts_completion_and_eos_only(self):
        logits = torch.zeros((1, 4, 4), requires_grad=True)
        token_ids = torch.tensor([[1, 2, 3, 0]])
        value = completion_logprob_sum(logits, token_ids, prompt_length=2, chunk_size=1)
        self.assertAlmostEqual(float(value.detach()), -2 * math.log(4), places=6)
        value.backward()
        self.assertTrue(torch.equal(logits.grad[0, 0], torch.zeros(4)))
        self.assertTrue(torch.equal(logits.grad[0, 3], torch.zeros(4)))
        self.assertAlmostEqual(float(logits.grad[0, 1, 3]), 0.75)
        self.assertAlmostEqual(float(logits.grad[0, 2, 0]), 0.75)
        self.assertAlmostEqual(float(logits.grad[0, 1, 0]), -0.25)

    def test_positive_advantage_increases_completion_probability(self):
        parameter = torch.nn.Parameter(torch.zeros((1, 2, 3)))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        tokens = torch.tensor([[0, 2]])
        initial = float(completion_logprob_sum(parameter, tokens, 1).detach())
        loss = -completion_logprob_sum(parameter, tokens, 1)
        loss.backward()
        optimizer.step()
        self.assertGreater(float(completion_logprob_sum(parameter, tokens, 1).detach()), initial)

    def test_rejects_old_policy_changed_sampling_and_mixed_prompts(self):
        validate_group(group_fixture(), "checkpoint-a0", vocab_size=4)
        with self.assertRaisesRegex(ValueError, "Off-policy"):
            validate_group(group_fixture(), "checkpoint-a1", vocab_size=4)
        changed = group_fixture()
        changed["sampling"]["temperature"] = 0.8
        with self.assertRaisesRegex(ValueError, "distribution"):
            validate_group(changed, "checkpoint-a0")
        changed = group_fixture()
        changed["completions"][1]["prompt_tokens_sha256"] = "different-prompt"
        with self.assertRaisesRegex(ValueError, "different prompts"):
            validate_group(changed, "checkpoint-a0")
        changed = group_fixture()
        changed["prompt_token_ids"] = [1, 3]
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_group(changed, "checkpoint-a0")

    def test_tensor_hash_preserves_bfloat16_and_checkpoint_lineage(self):
        first = {"a": torch.tensor([1, 2], dtype=torch.bfloat16)}
        second = {"a": torch.tensor([1, 3], dtype=torch.bfloat16)}
        first_hash = tensor_state_hash(first)
        self.assertEqual(first_hash, tensor_state_hash({"a": first["a"].clone()}))
        self.assertNotEqual(first_hash, tensor_state_hash(second))
        self.assertNotEqual(checkpoint_identity("base", first_hash), checkpoint_identity("base2", first_hash))

    def test_equal_rewards_record_no_update_and_save_no_adapter(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2))
                self.config = type("Config", (), {"vocab_size": 4})()

        class FakeSession:
            model = FakeModel()
            checkpoint_id = "checkpoint-a0"
            base_model_sha256 = "base-hash"
            adapter_sha256 = "adapter-hash"
            load_seconds = 0.0

            def release(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "no-update"
            with patch("commerce_rsi.agent_training.backend.load_agent", return_value=FakeSession()):
                report = update_from_group("unused", None, group_fixture(), [0.0, 0.0], output, device="cpu")
            self.assertEqual(report["status"], "no_update")
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertFalse(report["parameters_changed"])
            self.assertEqual(list(output.iterdir()), [output / "update.json"])

    def test_nonfinite_gradient_clip_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            for i, value in enumerate((float("nan"), float("inf"))):
                with patch("commerce_rsi.agent_training.backend.load_agent") as loader:
                    with self.assertRaisesRegex(ValueError, "optimizer configuration"):
                        update_from_group("unused", None, group_fixture(), [0.0, 1.0],
                                          Path(directory) / str(i), device="cpu", max_grad_norm=value)
                    loader.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("transformers") and importlib.util.find_spec("peft"),
                         "Optional Transformers/PEFT integration dependencies unavailable")
    def test_tiny_cpu_lora_generate_update_reload_and_optimizer_lineage(self):
        # Synthetic plumbing only: these random tokens/rewards are not RSI evidence.
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "tiny-base"
            base.mkdir()
            vocabulary = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                          "user": 4, "assistant": 5, "hello": 6}
            vocabulary.update({f"token{i}": i for i in range(7, 32)})
            raw_tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            raw_tokenizer.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer, bos_token="[BOS]",
                                               eos_token="[EOS]", pad_token="[PAD]", unk_token="[UNK]")
            tokenizer.chat_template = "user {{ messages[0]['content'] }} assistant"
            tokenizer.save_pretrained(base)
            torch.manual_seed(73)
            model = Qwen2ForCausalLM(Qwen2Config(
                vocab_size=32, hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=128, attention_dropout=0.0,
                bos_token_id=1, eos_token_id=2, pad_token_id=0,
            ))
            model.save_pretrained(base)
            del model
            a0 = load_agent(base, seed=17, device="cpu")
            group_a0 = a0.generate_group("hello", count=2, max_new_tokens=4, seed=47)
            a0.release()
            destination_a1 = Path(directory) / "a1"
            report_a1 = update_from_group(base, None, group_a0, [0.0, 1.0], destination_a1, device="cpu")
            self.assertEqual(report_a1["status"], "updated")
            self.assertTrue(report_a1["parameters_changed"])
            self.assertTrue(report_a1["reload_verified"])
            self.assertFalse(report_a1["optimizer_resumed"])
            a1 = load_agent(base, destination_a1, seed=17, device="cpu")
            group_a1 = a1.generate_group("hello", count=2, max_new_tokens=4, seed=61)
            self.assertEqual(group_a1["checkpoint_id"], report_a1["output_checkpoint_id"])
            a1.release()
            report_a2 = update_from_group(base, destination_a1, group_a1, [1.0, 0.0],
                                          Path(directory) / "a2", device="cpu")
            self.assertTrue(report_a2["optimizer_resumed"])
            self.assertTrue(report_a2["reload_verified"])
            with self.assertRaisesRegex(ValueError, "Off-policy"):
                update_from_group(base, destination_a1, group_a0, [0.0, 1.0],
                                  Path(directory) / "old-policy", device="cpu")
            without_optimizer = Path(directory) / "missing-optimizer"
            shutil.copytree(destination_a1, without_optimizer, ignore=shutil.ignore_patterns("optimizer.pt"))
            with self.assertRaisesRegex(ValueError, "implicit optimizer reset"):
                update_from_group(base, without_optimizer, group_a1, [0.0, 1.0],
                                  Path(directory) / "reset-rejected", device="cpu")


if __name__ == "__main__":
    unittest.main()
