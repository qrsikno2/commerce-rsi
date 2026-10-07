import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from commerce_rsi.retrieval.lexical import BM25
from commerce_rsi.retrieval.metrics import metrics_from_ranks, ranks_from_scores
from commerce_rsi.retrieval.model import BusinessConfig, initialize_model, model_hash
from commerce_rsi.retrieval.runner import run_business, sha256_file
from commerce_rsi.sampling import SamplingError


class RetrievalTests(unittest.TestCase):
    def test_metrics_match_hand_calculation(self):
        metrics = metrics_from_ranks(np.array([1, 2, 10, 11]))
        self.assertAlmostEqual(metrics["mrr"], (1 + 0.5 + 0.1 + 1 / 11) / 4)
        self.assertAlmostEqual(metrics["mrr@10"], (1 + 0.5 + 0.1) / 4)
        self.assertAlmostEqual(metrics["hit@10"], 0.75)
        self.assertAlmostEqual(metrics["ndcg@10"], (1 + 1 / math.log2(3) + 1 / math.log2(11)) / 4)

    def test_full_mrr_retains_feedback_beyond_rank_ten(self):
        metrics = metrics_from_ranks(np.array([1, 2, 11]))
        self.assertAlmostEqual(metrics["mrr"], (1 + 0.5 + 1 / 11) / 3)
        self.assertAlmostEqual(metrics["mrr@10"], (1 + 0.5) / 3)

    def test_ties_use_item_id_not_input_order(self):
        ranks = ranks_from_scores([[0, 1, 1], [0, 0, 0]], ["c", "a"], ["b", "c", "a"])
        self.assertEqual(ranks.tolist(), [2, 1])
        with self.assertRaises(ValueError):
            ranks_from_scores([[float("nan")]], ["a"], ["a"])
        with self.assertRaises(ValueError):
            ranks_from_scores([[1]], ["missing"], ["a"])

    def test_bm25_recovers_distinct_lexical_target(self):
        index = BM25(["wireless mouse computer", "blue cotton dress", "wooden desk"])
        self.assertEqual(int(np.argmax(index.score("blue dress"))), 1)

    def test_initialization_resets_after_training_mutation(self):
        config = BusinessConfig(vocab_size=64, embedding_dim=8, output_dim=8)
        initial = initialize_model(config, 19)
        expected = model_hash(initial)
        with torch.no_grad():
            next(initial.parameters()).add_(1)
        self.assertNotEqual(model_hash(initial), expected)
        self.assertEqual(model_hash(initialize_model(config, 19)), expected)
        self.assertNotEqual(model_hash(initialize_model(config, 20)), expected)

    def test_shared_towers_have_one_parameter_set_and_both_paths_train(self):
        config = BusinessConfig(vocab_size=64, embedding_dim=8, output_dim=8, shared_towers=True)
        model = initialize_model(config, 17)
        self.assertIs(model.query, model.item)
        self.assertEqual(len(list(model.parameters())), len(list(model.query.parameters())))
        query_tokens = torch.tensor([[1, 2, 0]])
        item_tokens = torch.tensor([[[3, 4, 0], [5, 6, 0]]])
        loss = torch.nn.functional.cross_entropy(model(query_tokens, item_tokens), torch.tensor([0]))
        loss.backward()
        gradient = model.query.embedding.weight.grad
        self.assertGreater(float(gradient[1:3].abs().sum()), 0)
        self.assertGreater(float(gradient[3:7].abs().sum()), 0)
        independent = initialize_model(BusinessConfig(vocab_size=64, embedding_dim=8, output_dim=8), 17)
        self.assertIsNot(independent.query, independent.item)

    def _fixture(self, directory):
        path = Path(directory) / "prepared"
        (path / "public").mkdir(parents=True)
        (path / "tasks" / "task0").mkdir(parents=True)
        (path / "public" / "catalog.jsonl").write_text("\n".join(json.dumps({"item_id": str(i), "text": text, "category": "test"}) for i, text in enumerate(["red shirt", "blue shirt", "yellow lamp", "black chair"])))
        rows = [{"qid": str(i), "query": text, "item_id": str(i)} for i, text in enumerate(["red shirt", "blue shirt"])]
        for split in ("train", "reward_validation"):
            (path / "tasks" / "task0" / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        return path

    def test_paired_strategy_runs_reset_weights_and_keep_candidates(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            prepared = self._fixture(directory)
            config = BusinessConfig(vocab_size=128, embedding_dim=8, output_dim=8, query_max_tokens=4, item_max_tokens=4)
            results = [run_business(prepared, "task0", strategy, Path(directory) / strategy,
                                    config=config, device="cpu", steps=2, batch_size=2, num_negatives=2)
                       for strategy in ("uniform", "lexical_hard")]
            self.assertEqual(results[0]["business_initialization_sha256"], results[1]["business_initialization_sha256"])
            self.assertEqual(results[0]["data_sha256"], results[1]["data_sha256"])
            self.assertEqual(results[0]["batch_schedule_sha256"], results[1]["batch_schedule_sha256"])
            self.assertTrue(all(row["parameters_changed"] and row["completed_steps"] == 2 for row in results))
            checkpoint = torch.load(Path(directory) / "uniform" / "checkpoint.pt", weights_only=True)
            restored = initialize_model(config, 0)
            restored.load_state_dict(checkpoint["state_dict"])
            self.assertEqual(model_hash(restored), results[0]["business_final_sha256"])

    def test_bad_sampler_records_failure_without_random_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = self._fixture(directory)
            destination = Path(directory) / "failed"
            with patch("commerce_rsi.retrieval.runner.UniformSampler.sample_negatives", return_value=[["0", "0"], ["0", "0"]]):
                with self.assertRaises(SamplingError):
                    run_business(prepared, "task0", "uniform", destination, device="cpu", steps=1,
                                 batch_size=2, num_negatives=2,
                                 config=BusinessConfig(vocab_size=32, embedding_dim=4, output_dim=4))
            report = json.loads((destination / "run.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failure"]["type"], "SamplingError")
            self.assertEqual(report["completed_steps"], 0)
            self.assertFalse((destination / "checkpoint.pt").exists())

    def test_isolated_callback_is_validated_and_records_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = self._fixture(directory)
            received = []

            def host_executor(batch, candidate_view, model_view, rng, num_negatives):
                received.append((batch, candidate_view, model_view))
                return [[item_id for item_id in candidate_view.item_ids
                         if item_id not in row["known_positive_item_ids"]][:num_negatives] for row in batch]

            report = run_business(prepared, "task0", "isolated", Path(directory) / "isolated",
                                  device="cpu", steps=1, batch_size=2, num_negatives=2,
                                  config=BusinessConfig(vocab_size=32, embedding_dim=4, output_dim=4),
                                  sampler_executor=host_executor, sampler_code_sha256="a" * 64,
                                  agent_checkpoint="synthetic-interface-test")
            self.assertEqual(report["sampler_code_sha256"], "a" * 64)
            self.assertEqual(report["agent_checkpoint"], "synthetic-interface-test")
            self.assertFalse(report["trusted_sampler_only"])
            self.assertEqual(set(received[0][2]), {"step", "total_steps"})
            self.assertEqual(set(received[0][0][0]), {"qid", "query", "item_id", "known_positive_item_ids"})

    def test_manifest_rejects_mutated_catalog_queries_and_meta_test(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = self._fixture(directory)
            manifest = {
                "candidate_count": 4,
                "catalog_sha256": sha256_file(prepared / "public" / "catalog.jsonl"),
                "tasks": [{"task_id": "task0", "outer_split": "meta_train"}],
                "query_files": [{"path": f"tasks/task0/{split}.jsonl",
                                 "sha256": sha256_file(prepared / "tasks" / "task0" / f"{split}.jsonl")}
                                for split in ("train", "reward_validation")],
            }
            manifest_path = prepared / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            report = run_business(prepared, "task0", "bm25", Path(directory) / "valid")
            self.assertTrue(report["prepared_manifest_verified"])
            catalog = prepared / "public" / "catalog.jsonl"
            original = catalog.read_text()
            catalog.write_text(original + "\n")
            with self.assertRaisesRegex(ValueError, "candidate pool SHA256"):
                run_business(prepared, "task0", "bm25", Path(directory) / "mutated-catalog")
            catalog.write_text(original)
            train_path = prepared / "tasks" / "task0" / "train.jsonl"
            original_train = train_path.read_text()
            train_path.write_text(original_train + "\n")
            with self.assertRaisesRegex(ValueError, "Query split SHA256"):
                run_business(prepared, "task0", "bm25", Path(directory) / "mutated-query")
            train_path.write_text(original_train)
            manifest["tasks"][0]["outer_split"] = "meta_test"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "meta-test"):
                run_business(prepared, "task0", "bm25", Path(directory) / "locked-meta-test")


if __name__ == "__main__":
    unittest.main()
