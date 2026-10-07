"""python -m commerce_rsi.retrieval --help"""

import argparse
import json

from .runner import run_business
from .model import BusinessConfig


def main():
    parser = argparse.ArgumentParser(description="Bounded, trusted Amazon-C4 business retrieval experiment")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--strategy", choices=("bm25", "uniform", "lexical_hard"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-negatives", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shared-towers", action="store_true", help="Pilot Siamese encoder; freeze this choice before agent experiments")
    parser.add_argument("--sampler-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-runtime-seconds", type=float, default=600.0)
    args = parser.parse_args()
    result = run_business(args.prepared, args.task_id, args.strategy, args.output,
                          seed=args.seed, steps=args.steps, batch_size=args.batch_size,
                          num_negatives=args.num_negatives, device=args.device,
                          config=BusinessConfig(shared_towers=args.shared_towers),
                          sampler_timeout_seconds=args.sampler_timeout_seconds,
                          max_runtime_seconds=args.max_runtime_seconds)
    print(json.dumps({key: result[key] for key in ("status", "task_id", "strategy", "metrics", "cost")}, indent=2))


if __name__ == "__main__":
    main()
