"""Run the leakage-safe layer x contamination covariance experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from repbank.contamination import (
    ContaminationConfig,
    contamination_grid,
    load_feature_bank,
    merge_contamination_outputs,
)


def _numbers(value: str, convert):
    return tuple(convert(item) for item in value.split(",") if item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--merge", type=Path, nargs="+", help="merge layer-chunk JSON reports")
    parser.add_argument("--position", choices=("answer_last", "prompt_last", "answer_mean"),
                        default="answer_last")
    parser.add_argument("--pis", default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--dimensions", default="32,64,128")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--ridge-fraction", type=float, default=0.1)
    parser.add_argument("--shrinkage", choices=("ridge", "oas"), default="ridge")
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--blocks", help="comma-separated transformer block indices; default is all blocks")
    parser.add_argument("--seed", type=int, default=20_260_901)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.bank) == bool(args.merge):
        parser.error("pass exactly one of --bank or --merge")
    if args.merge:
        if not args.output:
            parser.error("--merge requires --output")
        result = merge_contamination_outputs(args.merge)
        rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
        print(rendered)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        return
    states, labels, groups, metadata = load_feature_bank(args.bank, args.position)
    config = ContaminationConfig(
        contamination=_numbers(args.pis, float), dimensions=_numbers(args.dimensions, int),
        folds=args.folds, repeats=args.repeats, ridge_fraction=args.ridge_fraction,
        shrinkage=args.shrinkage,
        pool_size=args.pool_size, seed=args.seed,
        blocks=_numbers(args.blocks, int) if args.blocks else None,
    )
    result = contamination_grid(states, labels, groups, config)
    result["bank"] = str(args.bank)
    result["position"] = args.position
    result["bank_metadata"] = {
        key: metadata.get(key) for key in
        ("model_path", "model_hash", "generation_set_checksum", "n_layers", "d_model")
    }
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
