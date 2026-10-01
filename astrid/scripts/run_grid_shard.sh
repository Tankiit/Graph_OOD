#!/usr/bin/env bash
# Run shard <shard> of <n_shards> of the E16 grid (configs/steer/grid_jobs.txt), one run after another.
#   bash scripts/run_grid_shard.sh <shard> <n_shards>
# Job k of the list belongs to shard k % n_shards. A run whose summary.csv exists is skipped, so a
# resubmitted shard resumes. A failed run is logged and the shard moves on; the exit code is 1 if any run failed.
# Overrides: GRID_JOBS (job list file), GRID_EXTRA (extra key=value appended to every --set, e.g. for a benchmark).
set -euo pipefail
cd "$(dirname "$0")/.."
SHARD=$1
N=$2
export TQDM_DISABLE=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # cluster deploy installs dependencies, not the project
JOBS=${GRID_JOBS:-configs/steer/grid_jobs.txt}
# the deployed venv's python when present (does not depend on the job activating it), else whatever is on PATH
PY=python
[[ -x .venv/bin/python ]] && PY=.venv/bin/python
echo "python: $($PY -c 'import sys, torch; print(sys.executable, torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")')"
EXTRA=${GRID_EXTRA:-}
failed=0
k=0
while IFS=$'\t' read -r name args; do
  if (( k % N == SHARD )); then
    dir="outputs/steering/$name"
    if [[ -f "$dir/summary.csv" ]]; then
      echo "[$(date +%T)] SKIP $name (done)"
    else
      echo "[$(date +%T)] START $name"
      # shellcheck disable=SC2086
      if [[ -f "$dir/metrics.json" ]] || $PY -m actdist.train_steer --config configs/steer/grid_base.toml --set $args $EXTRA < /dev/null; then
        if $PY -m actdist.compare_steer --exps "$dir" --csv "$dir/summary.csv" < /dev/null > /dev/null; then
          echo "[$(date +%T)] DONE $name"
        else
          echo "[$(date +%T)] FAILED $name (compare_steer)"; failed=1
        fi
      else
        echo "[$(date +%T)] FAILED $name (train_steer)"; failed=1
      fi
    fi
  fi
  k=$((k + 1))
done < "$JOBS"
echo "[$(date +%T)] SHARD $SHARD/$N FINISHED (failed=$failed)"
exit $failed
