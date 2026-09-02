#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=501693)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    groups = sorted({row["pair_id"] for row in rows})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(groups)
    heldout_count = round(len(groups) * args.heldout_fraction)
    heldout = set(groups[:heldout_count])
    training = [row for row in rows if row["pair_id"] not in heldout]

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.train_output.write_text("".join(json.dumps(row) + "\n" for row in training))
    split = {
        "seed": args.seed,
        "heldout_fraction": args.heldout_fraction,
        "training_groups": sorted(set(groups) - heldout),
        "heldout_groups": sorted(heldout),
        "training_rows": len(training),
        "heldout_rows": len(rows) - len(training),
    }
    args.split_output.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
