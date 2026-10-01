#!/usr/bin/env bash
# Grid benchmark: the most expensive grid config (by default ResNet layer2.0; pass a job list, e.g.
# configs/steer/bench_vit_jobs.txt, for another) with 20 training steps
# and the full evaluation, to measure time per step, evaluation time and peak memory on the cluster GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# optional argument: the benchmark job list (default: the ResNet one)
GRID_JOBS=${1:-configs/steer/bench_jobs.txt} bash scripts/run_grid_shard.sh 0 1
