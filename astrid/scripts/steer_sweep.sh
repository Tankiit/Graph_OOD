#!/usr/bin/env bash
# Layer x radius sweep of steering swarms, then the summary figure.
#
#   scripts/steer_sweep.sh                                   # defaults below
#   LAYERS=layer4.0,layer4.1 RADII=0.25,0.5,1,2 scripts/steer_sweep.sh --set loss=infonce+energy
#
# Extra arguments are passed to train_steer (e.g. --set swarm_size=128 steps=500).
# Cost per step at K=1024 on an RTX 4060 (resnet18_scratch, bs 64+64): penultimate 0.1 s,
# layer4.0 1.8 s, layer3.1 3.6 s, layer2.1 8 s, layer1.0 19 s -> run early layers / ViT on the cluster.
set -euo pipefail
cd "$(dirname "$0")/.."
unset ALL_PROXY all_proxy
export TQDM_DISABLE=1
mkdir -p outputs/logs

CONFIG=${CONFIG:-configs/steer/base.toml}
RUN=${RUN:-resnet18_scratch_mnist}
LAYERS=${LAYERS:-penultimate}
RADII=${RADII:-0.25,0.5,1,2}

uv run python -m actdist.train_steer --config "$CONFIG" --set run="$RUN" "$@" \
    --sweep layer="$LAYERS" radius="$RADII" 2>&1 | tee -a "outputs/logs/steer_sweep_${RUN}.log"

for det in energy msp maxlogit; do
  uv run python -m actdist.plot steer-sweep --exps "outputs/steering/${RUN}__*" --detector "$det" \
      --out "outputs/figures/steer_sweep_${RUN}_${det}.png"
done
