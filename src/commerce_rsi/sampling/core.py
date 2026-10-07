"""Small immutable public views and strict, fail-closed negative sampling."""

from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Mapping

import numpy as np


class SamplingError(ValueError):
    """An invalid or over-budget sampler result; never silently replaced."""


@dataclass(frozen=True)
class CandidateView:
    item_ids: tuple[str, ...]
    texts: tuple[str, ...]
    categories: tuple[str, ...]

    @classmethod
    def from_rows(cls, rows):
        # Explicitly discard all fields except this public whitelist.
        ordered = sorted(rows, key=lambda row: row["item_id"])
        ids = tuple(row["item_id"] for row in ordered)
        if not ids or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("Catalog IDs must be nonempty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate catalog item IDs")
        texts = tuple(row["text"] for row in ordered)
        if any(not isinstance(x, str) for x in texts):
            raise ValueError("Catalog text must be strings")
        return cls(ids, texts, tuple(str(row.get("category", "")) for row in ordered))

    @cached_property
    def index(self):
        return MappingProxyType({item_id: i for i, item_id in enumerate(self.item_ids)})

    @cached_property
    def item_id_set(self):
        return frozenset(self.item_ids)


def validate_negatives(result, batch, candidate_view, num_negatives, known_positives):
    """Validate exact shape, ID types/existence, all known positives and duplicates."""
    if not isinstance(num_negatives, int) or isinstance(num_negatives, bool) or num_negatives < 1:
        raise SamplingError("num_negatives must be a positive integer")
    if not isinstance(result, list) or len(result) != len(batch):
        raise SamplingError("Sampler must return one list per query")
    available = candidate_view.item_id_set
    for row_index, (query, negative_ids) in enumerate(zip(batch, result)):
        if not isinstance(negative_ids, list) or len(negative_ids) != num_negatives:
            raise SamplingError(f"Row {row_index}: wrong negative count or row type")
        if any(not isinstance(item_id, str) for item_id in negative_ids):
            raise SamplingError(f"Row {row_index}: negative IDs must be strings")
        if len(set(negative_ids)) != len(negative_ids):
            raise SamplingError(f"Row {row_index}: duplicate negatives")
        if not set(negative_ids).issubset(available):
            raise SamplingError(f"Row {row_index}: unknown catalog item")
        positives = set(known_positives.get(query["qid"], ())) | {query["item_id"]}
        if positives.intersection(negative_ids):
            raise SamplingError(f"Row {row_index}: known positive sampled as negative")
    return result


class UniformSampler:
    def __init__(self, known_positives: Mapping[str, frozenset[str]]):
        self.known_positives = MappingProxyType(dict(known_positives))

    def sample_negatives(self, batch, candidate_view, model_view, rng, num_negatives):
        result = []
        ids = candidate_view.item_ids
        for row in batch:
            excluded = self.known_positives.get(row["qid"], frozenset()) | {row["item_id"]}
            eligible_count = len(ids) - len(excluded.intersection(candidate_view.item_id_set))
            if eligible_count < num_negatives:
                raise SamplingError("Too few eligible candidates")
            if num_negatives * 3 > eligible_count or len(excluded) > len(ids) // 2:
                eligible = [item_id for item_id in ids if item_id not in excluded]
                result.append([eligible[int(i)] for i in rng.choice(len(eligible), size=num_negatives, replace=False)])
                continue
            # Rejection sampling is uniform without replacement and avoids an O(N)
            # eligibility array for every training query.
            chosen = []
            selected = set()
            while len(chosen) < num_negatives:
                item_id = ids[int(rng.integers(len(ids)))]
                if item_id not in excluded and item_id not in selected:
                    chosen.append(item_id)
                    selected.add(item_id)
            result.append(chosen)
        return result


class LexicalHardSampler:
    """Fixed BM25 top-pool sampling, with no claim that these are true negatives."""

    def __init__(self, candidate_view, known_positives, pool_size=100):
        from commerce_rsi.retrieval.lexical import BM25

        self.known_positives = MappingProxyType(dict(known_positives))
        self.pool_size = pool_size
        self.index = BM25(candidate_view.texts)
        self._pools = {}

    def sample_negatives(self, batch, candidate_view, model_view, rng, num_negatives):
        result = []
        for row in batch:
            key = (row["qid"], row["query"], row["item_id"], num_negatives)
            if key not in self._pools:
                scores = self.index.score(row["query"])
                # CandidateView ordering gives deterministic ID ordering for score ties.
                order = np.argsort(-scores, kind="stable")
                excluded = self.known_positives.get(row["qid"], frozenset()) | {row["item_id"]}
                eligible = [candidate_view.item_ids[i] for i in order
                            if candidate_view.item_ids[i] not in excluded]
                self._pools[key] = eligible[:max(self.pool_size, num_negatives)]
            pool = self._pools[key]
            if len(pool) < num_negatives:
                raise SamplingError("Too few eligible lexical candidates")
            selected = rng.choice(len(pool), size=num_negatives, replace=False)
            result.append([pool[int(i)] for i in selected])
        return result
