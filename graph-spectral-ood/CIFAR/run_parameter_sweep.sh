#!/bin/bash

################################################################################
# Runner to test different parameter combinations
################################################################################

# Configuration
ID_DATASET=${1:-cifar10}          # ID dataset
OOD_DATASET=${2:-svhn}            # OOD dataset
METHOD=${3:-scone}                # OOD detection method
EPOCHS=${4:-100}                  # Number of epochs

echo "======================================"
echo "Parameter Sweep for OOD Detection"
echo "======================================"
echo "ID Dataset: $ID_DATASET"
echo "OOD Dataset: $OOD_DATASET"
echo "Method: $METHOD"
echo "Epochs: $EPOCHS"
echo "======================================"

# Define parameter ranges
PI_1_VALUES=(0.1 0.2 0.3 0.4 0.5)
PI_2_VALUES=(0.05 0.1 0.15 0.2)
LEARNING_RATES=(0.001 0.005 0.01 0.05)

# Function to run training with specific parameters
run_experiment() {
    local id_dataset=$1
    local ood_dataset=$2
    local method=$3
    local pi1=$4
    local pi2=$5
    local lr=$6
    local epochs=$7
    
    echo "======================================"
    echo "Running: $id_dataset -> $ood_dataset"
    echo "Method: $method, π₁=$pi1, π₂=$pi2, LR=$lr"
    echo "======================================"
    
    # Create output directory for this experiment
    local output_dir="results/param_sweep/${id_dataset}_${ood_dataset}_${method}_pi1_${pi1}_pi2_${pi2}_lr_${lr}"
    mkdir -p "$output_dir"
    
    # Run the experiment and save output
    if [ "$method" = "scone" ]; then
        python train.py "$id_dataset" \
            --score "$method" \
            --aux_out_dataset "$ood_dataset" \
            --test_out_dataset "$ood_dataset" \
            --pi_1 "$pi1" \
            --pi_2 "$pi2" \
            --epochs "$epochs" \
            --batch_size 128 \
            --learning_rate "$lr" \
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
            --ngpu 1 2>&1 | tee "$output_dir/training.log"
    elif [ "$method" = "energy" ]; then
        python train.py "$id_dataset" \
            --score "$method" \
            --aux_out_dataset "$ood_dataset" \
            --test_out_dataset "$ood_dataset" \
            --pi_1 "$pi1" \
            --pi_2 "$pi2" \
            --epochs "$epochs" \
            --batch_size 128 \
            --learning_rate "$lr" \
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
            --ngpu 1 2>&1 | tee "$output_dir/training.log"
    elif [ "$method" = "woods" ]; then
        python train.py "$id_dataset" \
            --score "$method" \
            --aux_out_dataset "$ood_dataset" \
            --test_out_dataset "$ood_dataset" \
            --pi_1 "$pi1" \
            --pi_2 "$pi2" \
            --epochs "$epochs" \
            --batch_size 128 \
            --learning_rate "$lr" \
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
            --ngpu 1 2>&1 | tee "$output_dir/training.log"
    else
        echo "Unknown method: $method"
        return 1
    fi
    
    echo "======================================"
    echo "Completed: π₁=$pi1, π₂=$pi2, LR=$lr"
    echo "======================================"
}

# Create results directory
mkdir -p results/param_sweep

# Run parameter sweep
for pi1 in "${PI_1_VALUES[@]}"; do
    for pi2 in "${PI_2_VALUES[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            echo "Testing parameters: π₁=$pi1, π₂=$pi2, LR=$lr"
            
            # Run the experiment
            run_experiment "$ID_DATASET" "$OOD_DATASET" "$METHOD" "$pi1" "$pi2" "$lr" "$EPOCHS"
            
            # Add a small delay between experiments
            sleep 5
        done
    done
done

echo "======================================"
echo "Parameter sweep completed!"
echo "Results saved in: results/param_sweep/"
echo "======================================"

# Generate parameter sweep summary
echo "Generating parameter sweep summary..."
python3 -c "
import os
import re
import glob
import pandas as pd

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

# Collect results
results = []
for log_file in glob.glob('results/param_sweep/*/training.log'):
    parts = log_file.split('/')
    if len(parts) >= 3:
        experiment = parts[2]
        parts = experiment.split('_')
        if len(parts) >= 8:
            id_dataset = parts[0]
            ood_dataset = parts[1]
            method = parts[2]
            pi1 = float(parts[4])
            pi2 = float(parts[6])
            lr = float(parts[8])
            
            acc, auroc, fpr = extract_metrics(log_file)
            results.append({
                'ID_Dataset': id_dataset,
                'OOD_Dataset': ood_dataset,
                'Method': method,
                'PI_1': pi1,
                'PI_2': pi2,
                'Learning_Rate': lr,
                'Accuracy': acc,
                'AUROC': auroc,
                'FPR': fpr
            })

# Create DataFrame and save
if results:
    df = pd.DataFrame(results)
    df.to_csv('results/param_sweep/summary.csv', index=False)
    
    print('\\n' + '='*100)
    print('PARAMETER SWEEP SUMMARY')
    print('='*100)
    print(df.to_string(index=False))
    
    # Find best parameters
    best_auroc = df.loc[df['AUROC'].idxmax()]
    best_acc = df.loc[df['Accuracy'].idxmax()]
    
    print('\\n' + '='*100)
    print('BEST PARAMETERS')
    print('='*100)
    print(f'Best AUROC: {best_auroc[\"AUROC\"]:.2f}%')
    print(f'Parameters: π₁={best_auroc[\"PI_1\"]}, π₂={best_auroc[\"PI_2\"]}, LR={best_auroc[\"Learning_Rate\"]}')
    print(f'\\nBest Accuracy: {best_acc[\"Accuracy\"]:.2f}%')
    print(f'Parameters: π₁={best_acc[\"PI_1\"]}, π₂={best_acc[\"PI_2\"]}, LR={best_acc[\"Learning_Rate\"]}')
    print('='*100)
else:
    print('No results found!')
"
