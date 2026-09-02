#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from repbank.generation_set import FrozenGenerationSet
from repbank.tinker_ops import run, train_adapter


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/adapter_bank.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/adapter_bank.json"))
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    frozen = FrozenGenerationSet.read(config["generation_set"])
    existing = json.loads(args.manifest.read_text()) if args.manifest.exists() else {"adapters": {}}
    manifest = {
        "schema_version": 1,
        "generation_set": config["generation_set"],
        "generation_set_checksum": frozen.checksum_sha256,
        "training_data": config["training_data"],
        "adapters": existing.get("adapters", {}),
    }
    defaults = config["defaults"]
    selected = [item for item in config["adapters"] if item.get("enabled", True)]
    if args.only:
        selected = [item for item in selected if item["id"] in args.only]
    unknown = set(args.only or ()) - {item["id"] for item in config["adapters"]}
    if unknown:
        raise SystemExit(f"unknown adapters: {', '.join(sorted(unknown))}")
    for item in selected:
        prior = manifest["adapters"].get(item["id"], {})
        if prior.get("status") == "complete":
            print(f"{item['id']}: already complete")
            continue
        entry = {**item, "status": "planned"}
        manifest["adapters"][item["id"]] = entry
        write_manifest(args.manifest, manifest)
        if not args.execute:
            print(json.dumps(entry, sort_keys=True))
            continue
        entry.update(status="running", started_at=time.time())
        write_manifest(args.manifest, manifest)
        try:
            path = run(train_adapter(
                item["base"], item["rank"], item["role"], config["training_data"],
                defaults["epochs"], defaults["lr"], ttl_seconds=defaults["ttl_seconds"],
                seed=defaults["seed"], batch_size=defaults["batch_size"],
                save_name=item["id"],
                lr_schedule=defaults.get("lr_schedule", "constant"),
                warmup_ratio=defaults.get("warmup_ratio", 0.0),
                min_lr_ratio=defaults.get("min_lr_ratio", 0.1),
            ))
        except Exception as exc:
            entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
            write_manifest(args.manifest, manifest)
            raise
        entry.update(status="complete", tinker_path=path, finished_at=time.time())
        write_manifest(args.manifest, manifest)
        print(f"{item['id']}: {path}")


if __name__ == "__main__":
    main()
