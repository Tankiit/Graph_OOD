"""Run the checksum-coupled activation VJP/JVP probe on Modal.

Example:
  modal run modal_jacobian_probe.py --max-pairs 8
"""

from __future__ import annotations

import json

import modal

app = modal.App("repbank-jacobian-probe")
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
)
hf_cache = modal.Volume.from_name("repbank-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("repbank-results", create_if_missing=True)


@app.function(
    image=image, gpu="L4", cpu=4, memory=32768, timeout=60 * 60,
    volumes={"/cache": hf_cache, "/outputs": outputs},
    env={"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"},
)
def run_probe(max_pairs: int = 8, jvp_examples: int = 3) -> dict:
    from repbank.jacobian import run_jacobian_probe

    result = run_jacobian_probe(
        model_path="Qwen/Qwen3.5-9B",
        bank_path="/outputs/coupled/primary.npz",
        generation_set_path="/root/generation_set.json",
        output_path="/outputs/jacobian/primary.npz",
        layer_fraction=0.5,
        readout_fraction=0.8,
        max_pairs=max_pairs,
        jvp_examples=jvp_examples,
    )
    outputs.commit()
    return result


@app.local_entrypoint()
def main(max_pairs: int = 8, jvp_examples: int = 3):
    print(json.dumps(run_probe.remote(max_pairs, jvp_examples), indent=2))
