#!/bin/bash

################################################################################
# Spectral Monitoring Experiments Runner
# Runs experiments across multiple ID and wild data combinations
# Saves results and generates visualizations
################################################################################

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RESULTS_BASE_DIR="spectral_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="${RESULTS_BASE_DIR}/experiment_${TIMESTAMP}"
LOG_DIR="${EXPERIMENT_DIR}/logs"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoints"
VIS_DIR="${EXPERIMENT_DIR}/visualizations"

# Create directories
mkdir -p ${LOG_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${VIS_DIR}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Spectral Monitoring Experiment Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Experiment directory: ${GREEN}${EXPERIMENT_DIR}${NC}"

################################################################################
# Experiment Configuration
################################################################################

# ID datasets
ID_DATASETS=("cifar10" "cifar100" "MNIST")

# Wild datasets (auxiliary OOD)
WILD_DATASETS=("svhn" "lsun_c" "FashionMNIST" "iNaturalist")

# Test OOD datasets
TEST_DATASETS=("svhn" "lsun_c" "FashionMNIST")

# Methods to test
METHODS=("scone" "woods" "energy" "OE")

# Pi configurations to test
PI_CONFIGS=(
    "0.0,0.0"   # Pure ID
    "0.3,0.0"   # Only covariate shift
    "0.0,0.3"   # Only semantic shift
    "0.3,0.1"   # Mixed optimal
    "0.2,0.2"   # Balanced mixture
    "0.1,0.3"   # Semantic dominant
)

# Training parameters
EPOCHS=100
BATCH_SIZE=128
LEARNING_RATE=0.001
SPECTRAL_FREQ=10
SPECTRAL_REG_ALPHA=0.01

################################################################################
# Helper Functions
################################################################################

run_experiment() {
    local id_data=$1
    local wild_data=$2
    local test_data=$3
    local method=$4
    local pi_1=$5
    local pi_2=$6
    local exp_name="${id_data}_${wild_data}_${test_data}_${method}_pi${pi_1}_${pi_2}"

    echo -e "\n${YELLOW}Running: ${exp_name}${NC}"
    echo -e "ID: ${id_data}, Wild: ${wild_data}, Test: ${test_data}"
    echo -e "Method: ${method}, π₁=${pi_1}, π₂=${pi_2}"

    # Build command
    CMD="python train.py ${id_data} \
        --score ${method} \
        --aux_out_dataset ${wild_data} \
        --test_out_dataset ${test_data} \
        --pi_1 ${pi_1} \
        --pi_2 ${pi_2} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --spectral_monitor \
        --spectral_freq ${SPECTRAL_FREQ} \
        --spectral_adaptive \
        --spectral_reg \
        --spectral_reg_alpha ${SPECTRAL_REG_ALPHA} \
        --spectral_analysis \
        --spectral_k_neighbors 50 \
        --checkpoints_dir ${CHECKPOINT_DIR} \
        --results_dir ${EXPERIMENT_DIR} \
        --name ${exp_name}"

    # Add method-specific parameters
    case ${method} in
        "scone"|"woods")
            CMD="${CMD} --in_constraint_weight 1.0 --out_constraint_weight 1.0"
            ;;
        "energy")
            CMD="${CMD} --m_in -25 --m_out -5"
            ;;
        "OE")
            CMD="${CMD} --oe_lambda 0.5"
            ;;
    esac

    # Log file
    LOG_FILE="${LOG_DIR}/${exp_name}.log"

    # Run experiment
    echo -e "Logging to: ${LOG_FILE}"
    ${CMD} > ${LOG_FILE} 2>&1

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Completed: ${exp_name}${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed: ${exp_name}${NC}"
        return 1
    fi
}

################################################################################
# Main Experiment Loop
################################################################################

# Track successful experiments
declare -a SUCCESSFUL_EXPERIMENTS
declare -a FAILED_EXPERIMENTS

# Quick test mode (set to false for full experiments)
QUICK_TEST=${1:-false}

if [ "$QUICK_TEST" = "true" ]; then
    echo -e "${YELLOW}Running in QUICK TEST mode (reduced configurations)${NC}"
    ID_DATASETS=("cifar10")
    WILD_DATASETS=("FashionMNIST")
    TEST_DATASETS=("svhn")
    METHODS=("scone")
    PI_CONFIGS=("0.3,0.1")
    EPOCHS=10
fi

# Save experiment configuration
cat > ${EXPERIMENT_DIR}/config.txt << EOF
Experiment Configuration
========================
Timestamp: ${TIMESTAMP}
ID Datasets: ${ID_DATASETS[@]}
Wild Datasets: ${WILD_DATASETS[@]}
Test Datasets: ${TEST_DATASETS[@]}
Methods: ${METHODS[@]}
Pi Configurations: ${PI_CONFIGS[@]}
Epochs: ${EPOCHS}
Batch Size: ${BATCH_SIZE}
Learning Rate: ${LEARNING_RATE}
Spectral Frequency: ${SPECTRAL_FREQ}
Spectral Reg Alpha: ${SPECTRAL_REG_ALPHA}
EOF

echo -e "\n${BLUE}Starting experiments...${NC}"
TOTAL_EXPERIMENTS=0
COMPLETED_EXPERIMENTS=0

# Main experiment loop
for id_data in "${ID_DATASETS[@]}"; do
    for wild_data in "${WILD_DATASETS[@]}"; do
        for test_data in "${TEST_DATASETS[@]}"; do
            # Skip if wild and test are the same
            if [ "$wild_data" = "$test_data" ]; then
                continue
            fi

            for method in "${METHODS[@]}"; do
                for pi_config in "${PI_CONFIGS[@]}"; do
                    IFS=',' read -r pi_1 pi_2 <<< "$pi_config"

                    TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))

                    # Run experiment
                    if run_experiment "$id_data" "$wild_data" "$test_data" "$method" "$pi_1" "$pi_2"; then
                        SUCCESSFUL_EXPERIMENTS+=("${id_data}_${wild_data}_${test_data}_${method}_pi${pi_1}_${pi_2}")
                        COMPLETED_EXPERIMENTS=$((COMPLETED_EXPERIMENTS + 1))
                    else
                        FAILED_EXPERIMENTS+=("${id_data}_${wild_data}_${test_data}_${method}_pi${pi_1}_${pi_2}")
                    fi

                    echo -e "Progress: ${COMPLETED_EXPERIMENTS}/${TOTAL_EXPERIMENTS}"
                done
            done
        done
    done
done

################################################################################
# Results Summary
################################################################################

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Experiment Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total experiments: ${TOTAL_EXPERIMENTS}"
echo -e "${GREEN}Successful: ${#SUCCESSFUL_EXPERIMENTS[@]}${NC}"
echo -e "${RED}Failed: ${#FAILED_EXPERIMENTS[@]}${NC}"

# Save summary
cat > ${EXPERIMENT_DIR}/summary.txt << EOF
Experiment Summary
==================
Total: ${TOTAL_EXPERIMENTS}
Successful: ${#SUCCESSFUL_EXPERIMENTS[@]}
Failed: ${#FAILED_EXPERIMENTS[@]}

Successful Experiments:
$(printf '%s\n' "${SUCCESSFUL_EXPERIMENTS[@]}")

Failed Experiments:
$(printf '%s\n' "${FAILED_EXPERIMENTS[@]}")
EOF

################################################################################
# Generate Visualizations
################################################################################

echo -e "\n${BLUE}Generating visualizations...${NC}"

# Run visualization script
python visualize_spectral_results.py \
    --experiment_dir ${EXPERIMENT_DIR} \
    --output_dir ${VIS_DIR}

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Visualizations generated in ${VIS_DIR}${NC}"
else
    echo -e "${RED}✗ Visualization generation failed${NC}"
fi

################################################################################
# Aggregate Results
################################################################################

echo -e "\n${BLUE}Aggregating results...${NC}"

# Run aggregation script
python aggregate_results.py \
    --experiment_dir ${EXPERIMENT_DIR} \
    --output_file ${EXPERIMENT_DIR}/aggregated_results.csv

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Results aggregated to ${EXPERIMENT_DIR}/aggregated_results.csv${NC}"
else
    echo -e "${RED}✗ Result aggregation failed${NC}"
fi

echo -e "\n${GREEN}Experiment run complete!${NC}"
echo -e "Results saved in: ${EXPERIMENT_DIR}"

# Optional: Open results directory
if command -v xdg-open &> /dev/null; then
    read -p "Open results directory? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        xdg-open ${EXPERIMENT_DIR}
    fi
fi