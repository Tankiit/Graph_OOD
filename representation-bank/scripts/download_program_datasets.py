#!/usr/bin/env python3
"""Download the programme's public benchmark splits as portable JSONL files."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Xet-backed transfers can hang indefinitely on restricted/research networks;
# regular HTTP is slower but predictable and resumable for these small splits.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "15")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from datasets import load_dataset


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo: str
    config: str | None
    splits: tuple[str, ...]
    group: str


CATALOG = (
    DatasetSpec("truthfulqa", "truthfulqa/truthful_qa", "generation", ("validation",), "protocol_qa"),
    DatasetSpec("triviaqa", "mandarjoshi/trivia_qa", "rc.nocontext", ("validation",), "protocol_qa"),
    DatasetSpec("coqa", "stanfordnlp/coqa", None, ("validation",), "protocol_qa"),
    DatasetSpec("tydiqa_gp", "google-research-datasets/tydiqa", "secondary_task", ("validation",), "protocol_qa"),
    DatasetSpec("nq_open", "google-research-datasets/nq_open", None, ("validation",), "protocol_qa"),
    DatasetSpec("sciq", "allenai/sciq", None, ("validation", "test"), "protocol_qa"),
    DatasetSpec("halueval_qa", "pminervini/HaluEval", "qa", ("data",), "typed_modes"),
    DatasetSpec("halueval_dialogue", "pminervini/HaluEval", "dialogue", ("data",), "typed_modes"),
    DatasetSpec("halueval_summarization", "pminervini/HaluEval", "summarization", ("data",), "typed_modes"),
    DatasetSpec("fava", "fava-uw/fava-data", None, ("train",), "typed_modes"),
    DatasetSpec("mu_shroom", "Helsinki-NLP/mu-shroom", "all", ("validation", "test"), "typed_modes"),
    DatasetSpec("gsm8k", "openai/gsm8k", "main", ("train", "test"), "clean_label_control"),
    DatasetSpec("mmlu", "cais/mmlu", "all", ("validation", "test"), "capability_control"),
)


def json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def download(spec: DatasetSpec, root: Path, force: bool) -> dict[str, Any]:
    target_dir = root / spec.group / spec.name
    target_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {**asdict(spec), "splits": {}, "status": "ok"}
    for split in spec.splits:
        target = target_dir / f"{split}.jsonl"
        if target.exists() and not force:
            count = sum(1 for _ in target.open())
            result["splits"][split] = {"rows": count, "path": str(target), "cached": True}
            continue
        dataset = load_dataset(spec.repo, spec.config, split=split)
        tmp = target.with_suffix(".jsonl.partial")
        with tmp.open("w") as handle:
            for row in dataset:
                handle.write(json.dumps(dict(row), ensure_ascii=False, default=json_default) + "\n")
        tmp.replace(target)
        result["splits"][split] = {"rows": len(dataset), "path": str(target), "cached": False}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/program_datasets"))
    parser.add_argument("--only", nargs="*", help="catalog names to download")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    selected = [spec for spec in CATALOG if not args.only or spec.name in args.only]
    unknown = set(args.only or ()) - {spec.name for spec in CATALOG}
    if unknown:
        raise SystemExit(f"unknown dataset names: {', '.join(sorted(unknown))}")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    existing = json.loads(manifest.read_text()) if manifest.exists() else []
    results_by_name = {result["name"]: result for result in existing}
    for spec in selected:
        print(f"Downloading {spec.name} ({spec.repo})", flush=True)
        try:
            results_by_name[spec.name] = download(spec, args.output, args.force)
        except Exception as exc:  # noqa: BLE001 - preserve progress across independent datasets
            results_by_name[spec.name] = {
                **asdict(spec), "status": "error", "error": f"{type(exc).__name__}: {exc}"
            }
            print(f"  ERROR: {exc}", flush=True)
        ordered_results = [results_by_name[item.name] for item in CATALOG if item.name in results_by_name]
        manifest.write_text(json.dumps(ordered_results, indent=2, ensure_ascii=False) + "\n")
    results = [results_by_name[item.name] for item in selected]
    failures = sum(result["status"] != "ok" for result in results)
    print(f"Finished: {len(results) - failures} succeeded, {failures} failed")


if __name__ == "__main__":
    main()
