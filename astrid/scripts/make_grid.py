"""Write a steering grid job list: E16 (ResNet, configs/steer/grid_jobs.txt) or E17 (ViT, configs/steer/grid_vit_jobs.txt).

    uv run python scripts/make_grid.py              # ResNet grid (E16) -> configs/steer/grid_jobs.txt
    uv run python scripts/make_grid.py --arch vit   # ViT grid (E17)    -> configs/steer/grid_vit_jobs.txt

One line per run, `<name>\\t<--set arguments>`, applied on top of configs/steer/grid_base.toml.
Grid: 2 models x 8 (layer, sim_layer) pairs x 3 radii = 48 runs. For each steered layer the
similarity is measured at the same layer, at the first block of the next stage, and at the
penultimate feature; at layer4.0 "next" is the penultimate feature, so that pair runs once.
Lines are sorted by expected cost (earlier steering layer = more network after it = slower),
so a round-robin split over shards is balanced.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RADII = [0.25, 0.5, 1.0]
ARCHS = {
    # runs, steered layer -> its "next" layer, extra --set values, job-list file, run-name prefix
    "resnet": dict(runs=["resnet18_scratch_mnist", "resnet18_scratch_fmnist"],
                   layers={"layer2.0": "layer3.0", "layer3.0": "layer4.0", "layer4.0": "penultimate"},
                   extra="", jobs="grid_jobs.txt", prefix="grid"),
    # ViT-S: same relative depths (1/4, 1/2, 3/4); vec_chunk 16 keeps blocks.3 at ~30 GB on a 46 GB RTX 8000
    "vit": dict(runs=["vit_small_ft_mnist", "vit_small_ft_fmnist"],
                layers={"blocks.3": "blocks.6", "blocks.6": "blocks.9", "blocks.9": "penultimate"},
                extra=" vec_chunk=16", jobs="grid_vit_jobs.txt", prefix="gridvit"),
}


def pairs(layers: dict):
    for layer, nxt in layers.items():
        for sim in dict.fromkeys([layer, nxt, "penultimate"]):  # dedup keeps order
            yield layer, sim


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCHS), default="resnet")
    a = ARCHS[ap.parse_args().arch]
    cost = {layer: i for i, layer in enumerate(a["layers"])}  # earlier steering layer = slower: first
    jobs = []
    for layer, sim in pairs(a["layers"]):
        for run in a["runs"]:
            for r in RADII:
                name = f"{a['prefix']}__{run}__{layer}__sim-{sim}__r{r:g}"
                jobs.append((cost[layer], name,
                             f"run={run} layer={layer} sim_layer={sim} radius={r}{a['extra']} name={name}"))
    jobs.sort(key=lambda j: j[0])
    names = [j[1] for j in jobs]
    assert len(names) == len(set(names)) == 48, len(names)

    for _, name, args in jobs:  # every line must resolve to a valid config
        out = subprocess.run([sys.executable, "-m", "actdist.train_steer", "--config", "configs/steer/grid_base.toml",
                              "--set", *args.split(), "--print"], cwd=ROOT, capture_output=True, text=True)
        if out.returncode != 0 or not out.stdout.startswith(name):
            raise SystemExit(f"invalid job {name}:\n{out.stderr[-2000:]}")

    path = ROOT / "configs" / "steer" / a["jobs"]
    path.write_text("".join(f"{name}\t{args}\n" for _, name, args in jobs))
    print(f"wrote {len(jobs)} jobs to {path}")
    for layer, sim in pairs(a["layers"]):
        print(f"  steer {layer:9s} sim {sim}")


if __name__ == "__main__":
    main()
