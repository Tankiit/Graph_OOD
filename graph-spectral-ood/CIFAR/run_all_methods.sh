#!/bin/bash

################################################################################
# Runner to test all OOD detection methods
################################################################################

# Configuration
DATASET=${1:-cifar10}
WILD_DATA=${2:-svhn}
TEST_DATA=${3:-lsun_c}
PI_1=${4:-0.3}
PI_2=${5:-0.1}
EPOCHS=${6:-100}

echo "======================================"
echo "Testing All OOD Detection Methods"
echo "======================================"
echo "Dataset: $DATASET"
echo "Wild data: $WILD_DATA"
echo "Test data: $TEST_DATA"
echo "π₁=$PI_1, π₂=$PI_2"
echo "Epochs: $EPOCHS"
echo "======================================"

# Test SCONE method
echo "Testing SCONE method..."
./run_with_spectral.sh $DATASET $WILD_DATA $TEST_DATA scone $PI_1 $PI_2 $EPOCHS

echo "======================================"
echo "SCONE completed!"
echo "======================================"

# Test Energy method
echo "Testing Energy method..."
./run_energy.sh $DATASET $WILD_DATA $TEST_DATA $PI_1 $PI_2 $EPOCHS

echo "======================================"
echo "Energy completed!"
echo "======================================"

# Test WOODS method
echo "Testing WOODS method..."
./run_woods.sh $DATASET $WILD_DATA $TEST_DATA $PI_1 $PI_2 $EPOCHS

echo "======================================"
echo "All methods completed!"
echo "======================================"
