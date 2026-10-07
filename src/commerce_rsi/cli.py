"""Small, explicit command line entry points; no implicit training or test access."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="commerce-rsi")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download and verify pinned Amazon-C4 files")
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--endpoint", default="https://huggingface.co")
    download.add_argument("--revision")
    download.add_argument("--transfer-endpoint", help="Optional mirror for bytes; checksums still from --endpoint")
    audit = commands.add_parser("audit", help="Audit actual raw files, including split feasibility")
    audit.add_argument("--raw", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="Build custom grouped protocol and fixed catalog")
    prepare.add_argument("--raw", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--candidate-count", type=int, default=30000)
    prepare.add_argument("--seed", type=int, default=20261007)
    prepare.add_argument("--minimum-task-queries", type=int, default=150)
    args = parser.parse_args()
    if args.command == "download":
        from commerce_rsi.data.download import download_dataset
        result = download_dataset(args.output, args.endpoint, args.revision, args.transfer_endpoint)
    elif args.command == "audit":
        from commerce_rsi.data.audit import audit_dataset
        result = audit_dataset(args.raw, args.output)
    else:
        from commerce_rsi.data.prepare import prepare_dataset
        result = prepare_dataset(args.raw, args.output, candidate_count=args.candidate_count,
                                 seed=args.seed, minimum_task_queries=args.minimum_task_queries)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
