"""Extract frozen sequences through the controlled Qwen adapter family on Modal.

Examples:
  modal run modal_coupled_extract.py --target primary
  modal run modal_coupled_extract.py --target ladder-1
"""

from __future__ import annotations

import json

import modal

app = modal.App("repbank-coupled-extract")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "accelerate>=1.0", "numpy>=1.26", "peft>=0.17", "pydantic>=2.7",
        "torch>=2.6", "transformers>=5.5,<5.6",
    )
    .add_local_dir("src/repbank", remote_path="/root/repbank")
    .add_local_file(
        "artifacts/frozen/truthfulqa_qwen35_family.json",
        remote_path="/root/generation_set.json",
    )
    .add_local_file(
        "artifacts/frozen/truthfulqa_qwen35_family_v2.json",
        remote_path="/root/generation_set_v2.json",
    )
    .add_local_file(
        "artifacts/frozen/truthfulqa_qwen35_holdout_0200_0816.json",
        remote_path="/root/generation_set_holdout.json",
    )
    .add_local_dir(
        "artifacts/adapters/ladder-1/tinker_adapter",
        remote_path="/root/adapters/ladder-1",
    )
    .add_local_dir("artifacts/adapters/M_true/tinker_adapter", remote_path="/root/adapters/M_true")
    .add_local_dir("artifacts/adapters/M_hal/tinker_adapter", remote_path="/root/adapters/M_hal")
)
hf_cache = modal.Volume.from_name("repbank-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("repbank-results", create_if_missing=True)
adapter_volume = modal.Volume.from_name("repbank-adapters", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=60 * 60,
    volumes={"/cache": hf_cache, "/outputs": outputs, "/adapter-volume": adapter_volume},
    env={"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"},
)
def extract_target(target: str) -> dict:
    import time

    from repbank.coupled_extract import extract_frozen

    configurations = {
        "primary": {
            "model_path": "Qwen/Qwen3.5-9B",
            "adapter_path": None,
            "adapter_id": "primary",
            "rank": 0,
            "include_span": True,
            "generation_set_path": "/root/generation_set.json",
        },
        "primary-v2": {
            "model_path": "Qwen/Qwen3.5-9B", "adapter_path": None,
            "adapter_id": "primary-v2", "rank": 0, "include_span": True,
            "generation_set_path": "/root/generation_set_v2.json",
        },
        **{
            f"v2-ladder-{rank}": {
                "model_path": "Qwen/Qwen3.5-9B",
                "adapter_path": f"/adapter-volume/v2/v2-ladder-{rank}/tinker_adapter",
                "adapter_id": f"v2-ladder-{rank}", "rank": rank,
                "include_span": False,
                "generation_set_path": "/root/generation_set_v2.json",
            }
            for rank in (1, 8, 16, 32)
        },
        "v2-ladder-32-5ep": {
            "model_path": "Qwen/Qwen3.5-9B",
            "adapter_path": "/adapter-volume/v2/v2-ladder-32-5ep/tinker_adapter",
            "adapter_id": "v2-ladder-32-5ep", "rank": 32,
            "include_span": False,
            "generation_set_path": "/root/generation_set_v2.json",
        },
        "v2-ladder-32-50ep-cosine-train80": {
            "model_path": "Qwen/Qwen3.5-9B",
            "adapter_path": (
                "/adapter-volume/v2/v2-ladder-32-50ep-cosine-train80/tinker_adapter"
            ),
            "adapter_id": "v2-ladder-32-50ep-cosine-train80", "rank": 32,
            "include_span": False,
            "generation_set_path": "/root/generation_set_v2.json",
        },
        "holdout-primary": {
            "model_path": "Qwen/Qwen3.5-9B", "adapter_path": None,
            "adapter_id": "holdout-primary", "rank": 0, "include_span": False,
            "generation_set_path": "/root/generation_set_holdout.json",
        },
        "holdout-rank32-50ep-cosine": {
            "model_path": "Qwen/Qwen3.5-9B",
            "adapter_path": (
                "/adapter-volume/v2/v2-ladder-32-50ep-cosine-train80/tinker_adapter"
            ),
            "adapter_id": "holdout-rank32-50ep-cosine", "rank": 32,
            "include_span": False,
            "generation_set_path": "/root/generation_set_holdout.json",
        },
        "ladder-1": {
            "model_path": "Qwen/Qwen3.5-9B",
            "adapter_path": "/root/adapters/ladder-1",
            "adapter_id": "ladder-1",
            "rank": 1,
            "include_span": False,
            "generation_set_path": "/root/generation_set.json",
        },
        "base-primary": {
            "model_path": "Qwen/Qwen3.5-9B-Base", "adapter_path": None,
            "adapter_id": "base-primary", "rank": 0, "include_span": True,
            "generation_set_path": "/root/generation_set.json",
        },
        "M_true": {
            "model_path": "Qwen/Qwen3.5-9B-Base", "adapter_path": "/root/adapters/M_true",
            "adapter_id": "M_true", "rank": 8, "include_span": False,
            "generation_set_path": "/root/generation_set.json",
        },
        "M_hal": {
            "model_path": "Qwen/Qwen3.5-9B-Base", "adapter_path": "/root/adapters/M_hal",
            "adapter_id": "M_hal", "rank": 8, "include_span": False,
            "generation_set_path": "/root/generation_set.json",
        },
    }
    if target not in configurations:
        raise ValueError(f"unknown target {target!r}; choose {sorted(configurations)}")
    config = configurations[target]
    generation_set_path = config.pop("generation_set_path")
    started = time.time()
    result = extract_frozen(
        **config,
        generation_set_path=generation_set_path,
        output_path=f"/outputs/coupled/{target}.npz",
        batch_size=1,
        span_cap=32,
        base_model=config["model_path"].endswith("-Base"),
    )
    outputs.commit()
    result["elapsed_seconds"] = round(time.time() - started, 2)
    return result


@app.local_entrypoint()
def main(target: str = "primary"):
    print(json.dumps(extract_target.remote(target), indent=2))
