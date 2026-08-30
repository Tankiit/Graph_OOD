#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from repbank.generation_set import (
    FrozenGenerationSet,
    g1_report,
    merge_frozen_sets,
    write_adapter_view,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-view", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    merged = merge_frozen_sets([FrozenGenerationSet.read(path) for path in args.input])
    merged.write(args.output, force=args.force)
    if args.adapter_view:
        write_adapter_view(merged, args.adapter_view)
    print(json.dumps(g1_report(merged), indent=2))


if __name__ == "__main__":
    main()
