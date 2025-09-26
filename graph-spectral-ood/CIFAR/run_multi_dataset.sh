#!/bin/bash

################################################################################
# Runner to test multiple ID and OOD dataset combinations
################################################################################

# Configuration
METHOD=${1:-scone}                 # OOD detection method
PI_1=${2:-0.3}                    # Covariate shift proportion
PI_2=${3:-0.1}                    # Semantic shift proportion
EPOCHS=${4:-100}                  # Number of epochs

echo "======================================"
echo "Testing Multiple ID and OOD Datasets"
echo "======================================"
echo "Method: $METHOD"
echo "π₁=$PI_1, π₂=$PI_2"
echo "Epochs: $EPOCHS"
echo "======================================"

# Define datasets
ID_DATASETS=("cifar10" "cifar100")
OOD_DATASETS=("svhn" "lsun_c" "lsun_r" "isun" "dtd" "places")

# Function to run training for a specific combination
run_experiment() {
    local id_dataset=$1
    local ood_dataset=$2
    local method=$3
    local pi1=$4
    local pi2=$5
    local epochs=$6
    
    echo "======================================"
    echo "Running: $id_dataset -> $ood_dataset ($method)"
    echo "======================================"
    
    # Create output directory for this experiment
    local output_dir="results/${id_dataset}_${ood_dataset}_${method}"
    mkdir -p "$output_dir"
    
    # Run the experiment and save output
    if [ "$method" = "scone" ]; then
        ./run_with_spectral.sh "$id_dataset" "$ood_dataset" "$ood_dataset" "$method" "$pi1" "$pi2" "$epochs" 2>&1 | tee "$output_dir/training.log"
    elif [ "$method" = "energy" ]; then
        ./run_energy.sh "$id_dataset" "$ood_dataset" "$ood_dataset" "$pi1" "$pi2" "$epochs" 2>&1 | tee "$output_dir/training.log"
    elif [ "$method" = "woods" ]; then
        ./run_woods.sh "$id_dataset" "$ood_dataset" "$ood_dataset" "$pi1" "$pi2" "$epochs" 2>&1 | tee "$output_dir/training.log"
    else
        echo "Unknown method: $method"
        return 1
    fi
    
    echo "======================================"
    echo "Completed: $id_dataset -> $ood_dataset ($method)"
    echo "======================================"
}

# Create results directory
mkdir -p results

# Run experiments for each ID dataset
for id_dataset in "${ID_DATASETS[@]}"; do
    echo "Starting experiments for ID dataset: $id_dataset"
    
    # Run experiments for each OOD dataset
    for ood_dataset in "${OOD_DATASETS[@]}"; do
        echo "Testing OOD dataset: $ood_dataset"
        
        # Run the experiment
        run_experiment "$id_dataset" "$ood_dataset" "$METHOD" "$PI_1" "$PI_2" "$EPOCHS"
        
        # Add a small delay between experiments
        sleep 5
    done
    
    echo "Completed all OOD datasets for ID dataset: $id_dataset"
    echo "======================================"
done

echo "======================================"
echo "All experiments completed!"
echo "Results saved in: results/"
echo "======================================"

# Generate summary report
echo "Generating summary report..."
python3 -c "
import os
import re
import glob

def extract_metrics(log_file):
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract final accuracy
        acc_match = re.search(r'accuracy ([\d.]+)', content)
        final_acc = float(acc_match.group(1)) if acc_match else 0.0
        
        # Extract final AUROC
        auroc_matches = re.findall(r'AUROC=([\d.]+)%', content)
        final_auroc = float(auroc_matches[-1]) if auroc_matches else 0.0
        
        # Extract final FPR
        fpr_matches = re.findall(r'FPR=([\d.]+)%', content)
        final_fpr = float(fpr_matches[-1]) if fpr_matches else 0.0
        
        return final_acc, final_auroc, final_fpr
    except:
        return 0.0, 0.0, 0.0

# Generate summary
print('\\n' + '='*80)
print('EXPERIMENT SUMMARY')
print('='*80)
print(f'{'ID Dataset':<12} {'OOD Dataset':<12} {'Method':<8} {'Accuracy':<10} {'AUROC':<8} {'FPR':<8}')
print('-'*80)

for log_file in glob.glob('results/*/training.log'):
    parts = log_file.split('/')
    if len(parts) >= 3:
        experiment = parts[1]
        id_dataset, ood_dataset, method = experiment.split('_')
        
        acc, auroc, fpr = extract_metrics(log_file)
        print(f'{id_dataset:<12} {ood_dataset:<12} {method:<8} {acc:<10.2f} {auroc:<8.1f} {fpr:<8.1f}')

print('='*80)
"
