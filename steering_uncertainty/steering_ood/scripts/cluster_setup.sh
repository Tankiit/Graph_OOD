#!/usr/bin/env bash
# Deploy steering_ood to the CRIL cluster and stage what scripts/run_learned_directions.py reads.
#
#   scripts/cluster_setup.sh                       # all four vision models, seeds 0 1 2
#   MODELS="resnet18" SEEDS="0" scripts/cluster_setup.sh
#   MODELS="" TEXT_MODELS="minilm" scripts/cluster_setup.sh    # text only
#   SKIP_DEPLOY=1 scripts/cluster_setup.sh         # only re-sync code and data, keep the venv
#
# Compute nodes have no internet, so the venv, the CIFAR images, the run_pipeline.py outputs
# (splits, feature caches, heads) and the pretrained weights are all copied from this machine.
# Never run the deploy step while a job of this project is running: it rebuilds the venv.
set -euo pipefail

HOST=slurm
PKG=$(cd "$(dirname "$0")/.." && pwd)
REMOTE=steering-ood                       # cluster deploy's default: ~/<project name>
MODELS=${MODELS-"resnet18 resnet50 vit_b16 dinov2_s"}
TEXT_MODELS=${TEXT_MODELS-"mpnet minilm bge_base bge_large"}
SEEDS=${SEEDS:-"0 1 2"}
STAGE=${STAGE:-${TMPDIR:-/tmp}/steering-ood-cluster}

# 1. A code-only copy whose dependencies are the pinned versions. `cluster deploy` installs only
#    [project].dependencies (the torch stack is an optional extra here) and uploads the whole
#    project directory, which would include the multi-GB runs/ tree.
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp -r "$PKG/steering_ood" "$PKG/steering_nlp" "$PKG/scripts" "$STAGE/"
deps=$(grep -vE '^[[:space:]]*(#|$)' "$PKG/requirements-pinned.txt" | sed 's/.*/  "&",/')
cat > "$STAGE/pyproject.toml" <<EOF
[project]
name = "steering-ood"
version = "0.2.0"
requires-python = ">=3.12"
dependencies = [
$deps
]
EOF
if [[ -z ${SKIP_DEPLOY:-} ]]; then
    cluster deploy "$STAGE" "$HOST"
else
    rsync -a --delete --exclude __pycache__ "$STAGE/steering_ood" "$STAGE/steering_nlp" "$STAGE/scripts" "$HOST:$REMOTE/"
fi

# 2. Data: CIFAR10/100 images (no SVHN, it is never used for training), and per seed the splits,
#    plus per model the feature cache and trained head. Paths keep the local layout under runs/.
files=()
for s in $SEEDS; do
    sub=$([[ $s == 0 ]] && echo "" || echo "seed$s/")
    files+=("runs/modal/data/${sub}cifar10_splits.json")
    [[ -f $PKG/runs/learned_directions/seed$s/steer_splits.json ]] && files+=("runs/learned_directions/seed$s/steer_splits.json")
    for m in $MODELS; do
        files+=("runs/modal/caches/${sub}$m.npz" "runs/modal/seeds/seed$s/$m/full/head")
    done
    if [[ -n $TEXT_MODELS ]]; then
        files+=("runs/modal/data/${sub}clinc_splits.json")
        [[ -f $PKG/runs/learned_directions/seed$s/steer_splits_clinc.json ]] && files+=("runs/learned_directions/seed$s/steer_splits_clinc.json")
    fi
    for m in $TEXT_MODELS; do
        files+=("runs/modal/caches/${sub}$m.npz" "runs/modal/seeds/seed$s/$m/full/head")
    done
done
# CLINC data_full.json holds the out-of-scope train/val queries used as the text OOD pool.
[[ -n $TEXT_MODELS ]] && files+=("runs/modal/data/clinc_data_full.json")
(cd "$PKG" && rsync -aR --info=progress2 runs/modal/data/images/cifar-10-batches-py \
    runs/modal/data/images/cifar-100-python "${files[@]}" "$HOST:$REMOTE/")

# 3. Pretrained weights: TorchVision checkpoints and the timm DINOv2 snapshot (-L resolves the
#    Hugging Face cache symlinks into real files). Jobs must set HF_HUB_OFFLINE=1.
ssh "$HOST" "mkdir -p .cache/torch/hub/checkpoints .cache/huggingface/hub"
rsync -a --info=progress2 "$HOME/.cache/torch/hub/checkpoints/" "$HOST:.cache/torch/hub/checkpoints/"
rsync -aL --info=progress2 "$HOME/.cache/huggingface/hub/models--timm--vit_small_patch14_dinov2.lvd142m/" \
    "$HOST:.cache/huggingface/hub/models--timm--vit_small_patch14_dinov2.lvd142m/"
declare -A HF=([mpnet]=sentence-transformers--all-mpnet-base-v2 [minilm]=sentence-transformers--all-MiniLM-L6-v2
               [bge_base]=BAAI--bge-base-en-v1.5 [bge_large]=BAAI--bge-large-en-v1.5)
for m in $TEXT_MODELS; do
    rsync -aL --info=progress2 "$HOME/.cache/huggingface/hub/models--${HF[$m]}/" "$HOST:.cache/huggingface/hub/models--${HF[$m]}/"
done

echo "Staged on $HOST:~/$REMOTE. Launch from $PKG with --image-root runs/modal/data/images and HF_HUB_OFFLINE=1."
