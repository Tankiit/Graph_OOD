#!/bin/bash

################################################################################
# Runner for training with Energy-based OOD detection
################################################################################

# Configuration - modify these as needed
DATASET=${1:-cifar10}              # ID dataset
WILD_DATA=${2:-svhn}               # Wild/auxiliary dataset
TEST_DATA=${3:-lsun_c}             # Test OOD dataset
PI_1=${4:-0.3}                     # Covariate shift proportion
PI_2=${5:-0.1}                     # Semantic shift proportion
EPOCHS=${6:-100}                   # Number of epochs

echo "======================================"
echo "Training with Energy-based OOD Detection"
echo "======================================"
echo "Dataset: $DATASET"
echo "Wild data: $WILD_DATA"
echo "Test data: $TEST_DATA"
echo "Method: energy"
echo "π₁=$PI_1, π₂=$PI_2"
echo "Epochs: $EPOCHS"
echo "======================================"

# Run training with energy-based OOD detection
python train.py $DATASET \
    --score energy \
    --aux_out_dataset $WILD_DATA \
    --test_out_dataset $TEST_DATA \
    --pi_1 $PI_1 \
    --pi_2 $PI_2 \
    --epochs $EPOCHS \
    --batch_size 128 \
    --learning_rate 0.01 \
    --model wrn \
    --layers 28 \
    --widen-factor 10 \
    --spectral_monitor \
    --spectral_freq 10 \
    --spectral_adaptive \
    --spectral_reg \
    --spectral_reg_alpha 0.01 \
    --spectral_k_neighbors 20 \
    --spectral_analysis \
    --print_freq 100 \
    --ngpu 1

echo "======================================"
echo "Training complete!"
echo "======================================"
