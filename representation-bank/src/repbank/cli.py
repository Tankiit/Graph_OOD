from __future__ import annotations

import argparse
import json

from .extract import extract
from .quantiles import register_direction
from .tinker_ops import (
    export_adapter,
    load_training_records,
    run,
    select_role_records,
    smoke_prompt_logprobs,
    train_adapter,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="repbank")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke-logprobs")
    smoke.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    smoke.add_argument("--text", default="Question: What is the capital of France? Answer: Paris")
    train = sub.add_parser("train-adapter")
    train.add_argument("--model", required=True); train.add_argument("--rank", type=int, required=True)
    train.add_argument("--role", choices=["true", "hal"], required=True); train.add_argument("--data", required=True)
    train.add_argument("--epochs", type=int, default=3); train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--ttl-seconds", type=int, default=604800)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--renderer", default=None)
    train.add_argument("--split", default="train", help="Hugging Face dataset split")
    train.add_argument("--dataset-config", default=None, help="Hugging Face dataset config/subset")
    train.add_argument("--revision", default=None, help="Hugging Face dataset revision")
    train.add_argument("--messages-column", default="messages")
    train.add_argument("--role-column", default="role")
    train.add_argument("--max-samples", type=int, default=None)
    train.add_argument("--validate-only", action="store_true",
                       help="validate/load data without creating a Tinker training job")
    export = sub.add_parser("export-adapter")
    export.add_argument("--tinker-path", required=True); export.add_argument("--base-model", required=True)
    export.add_argument("--output", required=True); export.add_argument("--merge", action="store_true")
    ext = sub.add_parser("extract")
    ext.add_argument("--config", required=True); ext.add_argument("--manifest", required=True)
    quant = sub.add_parser("register-direction")
    quant.add_argument("--cache", required=True); quant.add_argument("--name", required=True)
    quant.add_argument("--direction", required=True)
    args = parser.parse_args()
    if args.command == "smoke-logprobs":
        print(json.dumps(run(smoke_prompt_logprobs(args.model, args.text))))
    elif args.command == "train-adapter":
        if args.validate_only:
            records = load_training_records(
                args.data, split=args.split, dataset_config=args.dataset_config,
                revision=args.revision, messages_column=args.messages_column,
                role_column=args.role_column, max_samples=args.max_samples,
            )
            records = select_role_records(records, args.role)
            if not records:
                raise ValueError(f"no role={args.role!r} rows in {args.data}")
            print(json.dumps({"status": "valid", "records": len(records),
                              "role": args.role, "data": args.data}))
            return
        print(run(train_adapter(args.model, args.rank, args.role, args.data, args.epochs, args.lr,
                                ttl_seconds=args.ttl_seconds, batch_size=args.batch_size,
                                renderer_name=args.renderer, split=args.split,
                                dataset_config=args.dataset_config, revision=args.revision,
                                messages_column=args.messages_column, role_column=args.role_column,
                                max_samples=args.max_samples)))
    elif args.command == "export-adapter":
        print(export_adapter(args.tinker_path, args.base_model, args.output, args.merge))
    elif args.command == "extract":
        extract(args.config, args.manifest)
    elif args.command == "register-direction":
        print(json.dumps(register_direction(args.cache, args.name, args.direction)))
