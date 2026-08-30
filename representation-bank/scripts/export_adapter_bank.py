#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from repbank.tinker_ops import download_raw_adapter, export_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/adapter_bank.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/adapters"))
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--raw-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for adapter_id, entry in manifest["adapters"].items():
        if args.only and adapter_id not in args.only:
            continue
        if entry.get("status") != "complete":
            continue
        destination = args.output / adapter_id
        if args.raw_only:
            raw = destination / "tinker_adapter"
            result = download_raw_adapter(entry["tinker_path"], str(raw))
        else:
            result = export_adapter(entry["tinker_path"], entry["base"], str(destination), args.merge)
        entry["export_path"] = str(result)
        entry["export_type"] = "raw" if args.raw_only else ("merged" if args.merge else "peft")
        print(f"{adapter_id}: {result}")
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
