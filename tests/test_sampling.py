import unittest

import numpy as np

from commerce_rsi.sampling import CandidateView, LexicalHardSampler, SamplingError, UniformSampler, validate_negatives
from commerce_rsi.retrieval.runner import known_training_positives


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.view = CandidateView.from_rows([
            {"item_id": str(i), "text": text, "category": "test", "ori_review": "NEVER EXPOSE"}
            for i, text in enumerate(["red dress", "red skirt", "blue chair", "green chair", "desk lamp"])
        ])
        self.batch = [{"qid": "q1", "query": "red dress", "item_id": "0"}]
        self.known = {"q1": frozenset({"0", "1"})}

    def test_all_baselines_exclude_all_known_positives(self):
        for sampler in (UniformSampler(self.known), LexicalHardSampler(self.view, self.known)):
            first = sampler.sample_negatives(self.batch, self.view, {}, np.random.default_rng(42), 3)
            second = sampler.sample_negatives(self.batch, self.view, {}, np.random.default_rng(42), 3)
            self.assertEqual(first, second)
            self.assertEqual(set(first[0]), {"2", "3", "4"})
            validate_negatives(first, self.batch, self.view, 3, self.known)

    def test_rejects_invalid_outputs_without_fallback(self):
        for output in (None, [], [["2"]], [["2", "2"]], [["2", "missing"]], [["2", 3]], [["0", "2"]], [["1", "2"]], (("2", "3"),)):
            with self.subTest(output=output), self.assertRaises(SamplingError):
                validate_negatives(output, self.batch, self.view, 2, self.known)

    def test_insufficient_pool_is_explicit_failure(self):
        with self.assertRaises(SamplingError):
            UniformSampler(self.known).sample_negatives(self.batch, self.view, {}, np.random.default_rng(1), 4)

    def test_public_view_has_no_input_extras(self):
        self.assertNotIn("ori_review", repr(self.view))
        self.assertNotIn("NEVER EXPOSE", repr(self.view))

    def test_duplicate_query_labels_are_known_positives(self):
        rows = self.batch + [{"qid": "q2", "query": "  RED   dress ", "item_id": "1"}]
        positives = known_training_positives(rows)
        self.assertEqual(positives["q1"], frozenset({"0", "1"}))
        self.assertEqual(positives["q2"], frozenset({"0", "1"}))

    def test_duplicate_product_text_is_excluded_without_hidden_labels(self):
        view = CandidateView.from_rows([
            {"item_id": "0", "text": "Red dress"},
            {"item_id": "alias", "text": "  RED  DRESS  "},
            {"item_id": "other", "text": "Blue chair"},
        ])
        positives = known_training_positives(self.batch, view)
        self.assertEqual(positives["q1"], frozenset({"0", "alias"}))
        with self.assertRaises(SamplingError):
            validate_negatives([["alias"]], self.batch, view, 1, positives)


if __name__ == "__main__":
    unittest.main()
