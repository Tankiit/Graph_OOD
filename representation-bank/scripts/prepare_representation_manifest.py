#!/usr/bin/env python3
"""Filter generated records into an extraction-ready manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("role") in {"true", "hal"}]
    if args.paired_only:
        roles_by_pair: dict[str, set[str]] = {}
        for row in rows:
            roles_by_pair.setdefault(row["pair_id"], set()).add(row["role"])
        paired = {pair_id for pair_id, roles in roles_by_pair.items() if roles == {"true", "hal"}}
        rows = [row for row in rows if row["pair_id"] in paired]
    if args.limit is not None:
        rows = rows[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".partial")
    with temp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(args.output)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
