#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from repbank.generation_set import (
    freeze_tinker_generations,
    g1_report,
    write_adapter_view,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-view", type=Path)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--source-model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    decode_config = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "samples": args.samples,
        "stop": "\\n\\n",
    }
    frozen = freeze_tinker_generations(
        rows, tokenizer, tokenizer_id=args.tokenizer, source_model=args.source_model,
        decode_config=decode_config, base_seed=args.seed,
    )
    frozen.write(args.output, force=args.force)
    if args.adapter_view:
        write_adapter_view(frozen, args.adapter_view)
    print(json.dumps(g1_report(frozen), indent=2))


if __name__ == "__main__":
    main()
