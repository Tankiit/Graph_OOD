#!/usr/bin/env bash
set -euo pipefail
run_dir="${1:-runs/smoke}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
if [ -e "$run_dir" ]; then
  echo "Use a new output path: $run_dir already exists" >&2
  exit 1
fi
mkdir -p "$run_dir"
python -m steering_nlp e0 --include-pytorch --output "$run_dir/e0.json"
python -m steering_nlp fixture --output "$run_dir/fixture.npz"
python -m steering_nlp train-head --cache "$run_dir/fixture.npz" --output "$run_dir/head" --epochs 5
python -m steering_nlp evaluate --cache "$run_dir/fixture.npz" --head "$run_dir/head" --output "$run_dir/static"
python -m steering_nlp crossed --cache "$run_dir/fixture.npz" --head "$run_dir/head" --output "$run_dir/crossed" --references 2 --directions 2 --probes 4 --steps 31 --refine 8 --calibration-draws 2
python -m steering_nlp synthetic --output "$run_dir/gaussian" --references 3 --directions 3 --probes 8 --steps 61 --refine 10 --calibration-draws 2
