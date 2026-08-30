#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from repbank.gates import compare_model_axis, gate_g2_g3, gate_g4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--m-true", type=Path)
    parser.add_argument("--m-hal", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.base and args.adapter:
        result = compare_model_axis(args.base, args.adapter)
    elif args.bank:
        result = gate_g2_g3(args.bank)
    elif args.m_true and args.m_hal:
        result = gate_g4(args.m_true, args.m_hal)
    else:
        raise SystemExit("pass --bank, --base/--adapter, or --m-true/--m-hal")
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
