#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch

from repbank.coupled_extract import extract_frozen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--adapter-id", default="base")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--base-model", action="store_true")
    parser.add_argument("--generation-set", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--span-cap", type=int, default=32)
    parser.add_argument("--h-last-only", action="store_true")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device-map", default="cuda")
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    result = extract_frozen(
        model_path=args.model, adapter_path=args.adapter, adapter_id=args.adapter_id,
        rank=args.rank, base_model=args.base_model, generation_set_path=args.generation_set,
        output_path=args.output, batch_size=args.batch_size, span_cap=args.span_cap,
        include_span=not args.h_last_only, dtype=dtype, device_map=args.device_map,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
