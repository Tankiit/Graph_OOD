"""Cheap L4 smoke/full extraction for the Tinker TruthfulQA pilot.

Usage:
  modal run modal_repbank_extract.py                 # one-row smoke
  modal run modal_repbank_extract.py --limit 10     # bounded pilot
  modal run modal_repbank_extract.py --full         # all 154 labeled rows
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("repbank-qwen35-l4")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "accelerate>=1.0",
        "numpy>=1.26",
        "pydantic>=2.7",
        "pyyaml>=6.0",
        "safetensors>=0.4",
        "torch>=2.6",
        "transformers>=5.5,<5.6",
        "zarr>=2.18,<3",
    )
    .add_local_dir("src/repbank", remote_path="/root/repbank")
    .add_local_file(
        "artifacts/manifests/truthfulqa_base_labeled.jsonl",
        remote_path="/root/truthfulqa_base_labeled.jsonl",
    )
)

hf_cache = modal.Volume.from_name("repbank-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("repbank-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=60 * 60,
    volumes={"/cache": hf_cache, "/outputs": outputs},
    env={"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"},
)
def extract_remote(limit: int = 1, full: bool = False) -> dict:
    import subprocess
    import time

    from repbank.extract import extract

    source = Path("/root/truthfulqa_base_labeled.jsonl")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    selected = rows if full else rows[:limit]
    if not selected:
        raise ValueError("limit selected no rows")

    tag = "full" if full else f"smoke-{len(selected)}"
    manifest = Path(f"/tmp/{tag}.jsonl")
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    output_path = f"/outputs/qwen35-9b-base_truthfulqa_{tag}.zarr"
    config = {
        "model": {
            "hf_id": "Qwen/Qwen3.5-9B-Base",
            "tinker_id": "Qwen/Qwen3.5-9B-Base",
            "base_model": True,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "device_map": "cuda",
        },
        "capture": {
            "depth_fractions": [0.2, 0.5, 0.8],
            "span_cap": 32,
            "last_token_all_layers": True,
            "chunk_rows": 1 if not full else 16,
        },
        "generation": {"max_new_tokens": 64, "temperature": 0.0, "do_sample": False},
        "cache": {"path": output_path, "overwrite": True},
    }
    config_path = Path(f"/tmp/{tag}.json")
    config_path.write_text(json.dumps(config))
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
    ).strip()
    started = time.time()
    extract(str(config_path), str(manifest))
    outputs.commit()
    return {
        "rows": len(selected),
        "output": output_path,
        "gpu": gpu,
        "elapsed_seconds": round(time.time() - started, 2),
    }


@app.local_entrypoint()
def main(limit: int = 1, full: bool = False):
    print(json.dumps(extract_remote.remote(limit=limit, full=full), indent=2))
