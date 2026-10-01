#!/usr/bin/env bash
# Train 3 models x 2 datasets, extract activations on the test set, draw the distribution figures.
set -euo pipefail
cd "$(dirname "$0")/.."
unset ALL_PROXY all_proxy  # socks:// proxy scheme breaks the HF hub download (httpx)
export TQDM_DISABLE=1
mkdir -p outputs/logs

for ds in mnist fmnist; do
  for m in resnet18_scratch resnet18_ft vit_small_ft; do
    run=${m}_${ds}
    uv run python -m actdist.train --model "$m" --dataset "$ds" 2>&1 | tee "outputs/logs/${run}.log"
    uv run python -m actdist.extract --run "$run"
    uv run python -m actdist.plot heatmap --run "$run"
    for sel in selective variance random; do
      uv run python -m actdist.plot grid --run "$run" --select "$sel"
    done
  done
done
