from __future__ import annotations

import json

import modal

app = modal.App("repbank-v2-ladder-analysis")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy>=1.26", "pydantic>=2.7", "torch>=2.6", "transformers>=5.5,<5.6")
    .add_local_dir("src/repbank", remote_path="/root/repbank")
)
outputs = modal.Volume.from_name("repbank-results")


@app.function(image=image, cpu=8, memory=32768, timeout=30 * 60, volumes={"/outputs": outputs})
def analyze() -> dict:
    from pathlib import Path

    from repbank.gates import gate_g2_g3, rank_geometry_curve

    paths = [f"/outputs/coupled/v2-ladder-{rank}.npz" for rank in (1, 8, 16, 32)]
    result = rank_geometry_curve(
        "/outputs/coupled/primary-v2.npz",
        paths,
    )
    result["heldout_gates"] = {
        str(rank): gate_g2_g3(path) for rank, path in zip((1, 8, 16, 32), paths, strict=True)
    }
    target = Path("/outputs/coupled/v2-rank-geometry.json")
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    outputs.commit()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(analyze.remote(), indent=2))
