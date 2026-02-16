#!/bin/bash

# setup.sh
# Setup script for UAI OOD Detection Experiments

set -e  # Exit on error

echo "=========================================="
echo "UAI OOD Detection Experiment Setup"
echo "=========================================="

# 1. Create directories
echo "Creating directories..."
mkdir -p data
mkdir -p features
mkdir -p results
mkdir -p plots
mkdir -p checkpoints

# 2. Install Python dependencies
echo ""
echo "Installing Python dependencies..."

# Core dependencies
pip install torch torchvision numpy scipy scikit-learn pandas matplotlib seaborn tqdm

# CLIP
pip install open_clip_torch

# DINOv2 (via torch.hub, no additional install needed)

# 3. Download pretrained WRN checkpoints from SAL
echo ""
echo "Downloading pretrained WRN checkpoints from SAL..."
python3 download_sal_checkpoints.py

# Verify checkpoints were downloaded
if [ -f "checkpoints/wrn40_2_cifar10.pt" ] && [ -f "checkpoints/wrn40_2_cifar100.pt" ]; then
    echo "✓ Checkpoints downloaded successfully!"
else
    echo "✗ Checkpoint download failed. Please download manually from:"
    echo "  https://github.com/deeplearning-wisc/sal/tree/main/pretrained"
fi

# 5. Download OOD datasets
echo ""
echo "Downloading OOD datasets..."
echo "Note: CIFAR-10, CIFAR-100, and SVHN will be downloaded automatically"
echo ""
echo "For Textures (DTD):"
echo "  - Download from: https://www.robots.ox.ac.uk/~vgg/data/dtd/"
echo "  - Extract to: data/dtd/"
echo ""
echo "For LSUN-C:"
echo "  - Download from: https://github.com/yuval-alaluf/supervised-modeling-ood"
echo "  - Extract to: data/lsun_c/"
echo ""

# 6. Create environment variables file
cat > .env << EOF
# Configuration for UAI Experiments

# Data directories
DATA_ROOT=/Users/tanmoy/research/data
FEATURE_DIR=./features
RESULTS_DIR=./results
PLOT_DIR=./plots

# Checkpoint paths
WRN_CIFAR10_CKPT=./checkpoints/wrn40_2_cifar10.pt
WRN_CIFAR100_CKPT=./checkpoints/wrn40_2_cifar100.pt

# Device
DEVICE=cuda  # or cpu
EOF

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download pretrained WRN checkpoints"
echo "2. Download Textures and LSUN-C datasets"
echo "3. Run: python extract_features.py"
echo "4. Run: python compute_metrics_and_eval.py"
echo ""
