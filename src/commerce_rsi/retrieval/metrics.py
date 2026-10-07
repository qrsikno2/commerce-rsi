"""Single-annotated-target retrieval metrics with deterministic item-ID ties."""

import numpy as np


def ranks_from_scores(scores, target_ids, candidate_ids):
    scores = np.asarray(scores)
    candidate_ids = tuple(candidate_ids)
    if scores.ndim != 2 or scores.shape != (len(target_ids), len(candidate_ids)):
        raise ValueError("Score matrix shape does not match query/candidate IDs")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate IDs must be unique")
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite retrieval score")
    index = {item_id: i for i, item_id in enumerate(candidate_ids)}
    ids = np.asarray(candidate_ids)
    ranks = []
    for row, target in zip(scores, target_ids):
        if target not in index:
            raise ValueError(f"Target missing from fixed candidate pool: {target}")
        value = row[index[target]]
        rank = 1 + int(np.count_nonzero(row > value))
        rank += int(np.count_nonzero((row == value) & (ids < target)))
        ranks.append(rank)
    return np.asarray(ranks, dtype=np.int64)


def metrics_from_ranks(ranks, k=10):
    ranks = np.asarray(ranks)
    if ranks.ndim != 1 or len(ranks) == 0:
        raise ValueError("Metrics require a nonempty 1D rank array")
    if not np.issubdtype(ranks.dtype, np.integer) or np.any(ranks < 1):
        raise ValueError("Ranks must be positive integers")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    hit = ranks <= k
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        f"mrr@{k}": float(np.mean(np.where(hit, 1.0 / ranks, 0.0))),
        f"hit@{k}": float(np.mean(hit)),
        f"ndcg@{k}": float(np.mean(np.where(hit, 1.0 / np.log2(ranks + 1), 0.0))),
        "num_queries": int(len(ranks)),
    }
