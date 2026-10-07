"""Frozen business retrieval model, metrics and trusted baseline runners."""

from .metrics import metrics_from_ranks, ranks_from_scores


def run_business(*args, **kwargs):
    from .runner import run_business as run
    return run(*args, **kwargs)


__all__ = ["metrics_from_ranks", "ranks_from_scores", "run_business"]
