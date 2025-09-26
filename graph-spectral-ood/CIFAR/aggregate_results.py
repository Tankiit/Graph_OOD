#!/usr/bin/env python3
"""
Results aggregation script for spectral monitoring experiments
Combines results from multiple experiments into a single CSV
"""

import os
import glob
import pickle
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import json
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate experiment results')
    parser.add_argument('--experiment_dir', type=str, required=True,
                      help='Directory containing experiment results')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Output CSV file path')
    parser.add_argument('--log_dir', type=str, default=None,
                      help='Directory containing log files (default: experiment_dir/logs)')
    return parser.parse_args()

def parse_experiment_name(exp_name):
    """Parse experiment name to extract configuration"""
    # Expected format: dataset_wild_test_method_pi0.x_0.y
    pattern = r'(.+?)_(.+?)_(.+?)_(.+?)_pi([\d.]+)_([\d.]+)'
    match = re.match(pattern, exp_name)

    if match:
        return {
            'id_dataset': match.group(1),
            'wild_dataset': match.group(2),
            'test_dataset': match.group(3),
            'method': match.group(4),
            'pi_1': float(match.group(5)),
            'pi_2': float(match.group(6))
        }
    else:
        # Fallback parsing
        parts = exp_name.split('_')
        if len(parts) >= 6:
            return {
                'id_dataset': parts[0],
                'wild_dataset': parts[1],
                'test_dataset': parts[2],
                'method': parts[3],
                'pi_1': 0.0,
                'pi_2': 0.0
            }
        return {}

def extract_metrics_from_pkl(file_path):
    """Extract metrics from pickle file"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        metrics = {}

        # Extract final epoch metrics
        if 'fpr95_test' in data and data['fpr95_test']:
            metrics['final_fpr95'] = data['fpr95_test'][-1]

        if 'auroc_test' in data and data['auroc_test']:
            metrics['final_auroc'] = data['auroc_test'][-1]

        if 'test_accuracy' in data and data['test_accuracy']:
            metrics['final_test_acc'] = data['test_accuracy'][-1]

        if 'test_accuracy_cor' in data and data['test_accuracy_cor']:
            metrics['final_test_acc_cor'] = data['test_accuracy_cor'][-1]

        if 'train_accuracy' in data and data['train_accuracy']:
            metrics['final_train_acc'] = data['train_accuracy'][-1]

        # Extract spectral metrics
        if 'spectral_gaps' in data and data['spectral_gaps']:
            final_spectral = data['spectral_gaps'][-1]
            metrics['final_gap_pure'] = final_spectral.get('gap_pure_in', np.nan)
            metrics['final_gap_mixed'] = final_spectral.get('gap_mixed', np.nan)
            metrics['final_degradation'] = final_spectral.get('degradation', np.nan)
            metrics['final_wild_ratio'] = final_spectral.get('wild_ratio', np.nan)

            # Calculate average spectral metrics
            all_gaps = [g.get('gap_pure_in', 0) for g in data['spectral_gaps']]
            all_degradations = [g.get('degradation', 0) for g in data['spectral_gaps']]
            metrics['avg_gap_pure'] = np.mean(all_gaps) if all_gaps else np.nan
            metrics['avg_degradation'] = np.mean(all_degradations) if all_degradations else np.nan
            metrics['max_degradation'] = np.max(all_degradations) if all_degradations else np.nan

        # Extract training curves statistics
        if 'fpr95_test' in data and data['fpr95_test']:
            metrics['best_fpr95'] = np.min(data['fpr95_test'])
            metrics['avg_fpr95'] = np.mean(data['fpr95_test'])

        if 'auroc_test' in data and data['auroc_test']:
            metrics['best_auroc'] = np.max(data['auroc_test'])
            metrics['avg_auroc'] = np.mean(data['auroc_test'])

        # Extract constraint violations (for SCONE/WOODS)
        if 'in_dist_constraint' in data and data['in_dist_constraint']:
            violations = [v for v in data['in_dist_constraint'] if v > 0]
            metrics['constraint_violations'] = len(violations)
            metrics['max_constraint_violation'] = np.max(violations) if violations else 0

        # Extract validation metrics
        if 'val_wild_total' in data and data['val_wild_total']:
            metrics['val_wild_total'] = data['val_wild_total'][-1] if data['val_wild_total'] else 0

        if 'val_wild_class_as_in' in data and data['val_wild_class_as_in']:
            metrics['val_wild_classified_in'] = data['val_wild_class_as_in'][-1] if data['val_wild_class_as_in'] else 0
            if metrics.get('val_wild_total', 0) > 0:
                metrics['val_wild_in_ratio'] = metrics['val_wild_classified_in'] / metrics['val_wild_total']

        # Extract data sizes
        for key in ['train_in_size', 'train_aux_in_size', 'train_aux_out_size',
                    'valid_in_size', 'valid_aux_size', 'test_in_size',
                    'test_in_cor_size', 'test_out_size']:
            if key in data:
                metrics[key] = data[key]

        return metrics

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def extract_metrics_from_log(log_path):
    """Extract additional metrics from log file"""
    metrics = {}

    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()

        # Extract training time (look for epoch timing)
        epoch_times = []
        for line in lines:
            if 'Epoch time:' in line or 'epoch time' in line.lower():
                # Extract time value
                time_match = re.search(r'(\d+\.?\d*)\s*(s|sec|seconds)', line.lower())
                if time_match:
                    epoch_times.append(float(time_match.group(1)))

        if epoch_times:
            metrics['avg_epoch_time'] = np.mean(epoch_times)
            metrics['total_training_time'] = sum(epoch_times)

        # Extract final metrics from log
        for line in reversed(lines):  # Start from the end
            if 'final fpr95' in line.lower():
                match = re.search(r'(\d+\.?\d*)', line)
                if match:
                    metrics['log_final_fpr95'] = float(match.group(1))
                    break

            if 'final auroc' in line.lower():
                match = re.search(r'(\d+\.?\d*)', line)
                if match:
                    metrics['log_final_auroc'] = float(match.group(1))

        # Count warnings and errors
        metrics['num_warnings'] = sum(1 for line in lines if 'warning' in line.lower())
        metrics['num_errors'] = sum(1 for line in lines if 'error' in line.lower())

    except Exception as e:
        print(f"Error reading log {log_path}: {e}")

    return metrics

def aggregate_results(experiment_dir, log_dir=None):
    """Aggregate all results into a DataFrame"""
    results = []

    # Find all .pkl files
    pkl_files = glob.glob(os.path.join(experiment_dir, '*.pkl'))

    print(f"Found {len(pkl_files)} result files")

    for pkl_path in pkl_files:
        exp_name = os.path.basename(pkl_path).replace('.pkl', '')

        # Parse experiment configuration
        config = parse_experiment_name(exp_name)
        if not config:
            print(f"Could not parse: {exp_name}")
            continue

        # Extract metrics from pickle
        metrics = extract_metrics_from_pkl(pkl_path)

        # Extract metrics from log if available
        if log_dir:
            log_path = os.path.join(log_dir, f"{exp_name}.log")
            if os.path.exists(log_path):
                log_metrics = extract_metrics_from_log(log_path)
                metrics.update(log_metrics)

        # Combine configuration and metrics
        row = {**config, **metrics}
        row['experiment_name'] = exp_name
        row['pkl_file'] = os.path.basename(pkl_path)

        results.append(row)

    return pd.DataFrame(results)

def calculate_summary_statistics(df):
    """Calculate summary statistics across experiments"""
    summary = {}

    # Group by method
    if 'method' in df.columns:
        method_stats = df.groupby('method').agg({
            'final_fpr95': ['mean', 'std', 'min', 'max'],
            'final_auroc': ['mean', 'std', 'min', 'max'],
            'final_test_acc': ['mean', 'std'],
            'final_degradation': ['mean', 'std', 'min', 'max'],
            'avg_degradation': 'mean'
        })
        summary['by_method'] = method_stats

    # Group by dataset
    if 'id_dataset' in df.columns:
        dataset_stats = df.groupby('id_dataset').agg({
            'final_fpr95': ['mean', 'std'],
            'final_auroc': ['mean', 'std'],
            'final_degradation': ['mean', 'std']
        })
        summary['by_dataset'] = dataset_stats

    # Group by pi values
    if 'pi_1' in df.columns and 'pi_2' in df.columns:
        pi_stats = df.groupby(['pi_1', 'pi_2']).agg({
            'final_fpr95': 'mean',
            'final_auroc': 'mean',
            'final_degradation': 'mean',
            'final_gap_mixed': 'mean'
        })
        summary['by_pi'] = pi_stats

    # Overall best configurations
    if 'final_fpr95' in df.columns:
        best_fpr95 = df.nsmallest(5, 'final_fpr95')[['experiment_name', 'method',
                                                       'id_dataset', 'pi_1', 'pi_2',
                                                       'final_fpr95', 'final_auroc']]
        summary['best_fpr95'] = best_fpr95

    if 'final_degradation' in df.columns:
        best_degradation = df.nsmallest(5, 'final_degradation')[['experiment_name', 'method',
                                                                  'id_dataset', 'pi_1', 'pi_2',
                                                                  'final_degradation', 'final_gap_mixed']]
        summary['best_degradation'] = best_degradation

    return summary

def main():
    args = parse_args()

    # Set log directory
    if args.log_dir is None:
        args.log_dir = os.path.join(args.experiment_dir, 'logs')

    print(f"Aggregating results from: {args.experiment_dir}")
    if os.path.exists(args.log_dir):
        print(f"Using log directory: {args.log_dir}")

    # Aggregate results
    df = aggregate_results(args.experiment_dir, args.log_dir)

    if df.empty:
        print("No results found to aggregate!")
        return

    print(f"Aggregated {len(df)} experiments")

    # Sort by key columns
    sort_columns = []
    for col in ['method', 'id_dataset', 'wild_dataset', 'pi_1', 'pi_2']:
        if col in df.columns:
            sort_columns.append(col)
    if sort_columns:
        df = df.sort_values(sort_columns)

    # Save full results
    df.to_csv(args.output_file, index=False)
    print(f"✓ Saved aggregated results to {args.output_file}")

    # Calculate and save summary statistics
    summary = calculate_summary_statistics(df)

    # Save summary to text file
    summary_file = args.output_file.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("AGGREGATED RESULTS SUMMARY\n")
        f.write("="*60 + "\n\n")

        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Unique methods: {df['method'].nunique() if 'method' in df.columns else 0}\n")
        f.write(f"Unique datasets: {df['id_dataset'].nunique() if 'id_dataset' in df.columns else 0}\n\n")

        for key, value in summary.items():
            f.write(f"\n{key.upper().replace('_', ' ')}\n")
            f.write("-"*40 + "\n")
            f.write(str(value))
            f.write("\n")

    print(f"✓ Saved summary to {summary_file}")

    # Print brief summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)

    if 'method' in df.columns and 'final_fpr95' in df.columns:
        print("\nBest FPR95 by Method:")
        best_by_method = df.groupby('method')['final_fpr95'].min()
        for method, fpr in best_by_method.items():
            print(f"  {method}: {fpr:.4f}")

    if 'final_degradation' in df.columns:
        print(f"\nOverall Statistics:")
        print(f"  Avg Degradation: {df['final_degradation'].mean():.2%}")
        print(f"  Min Degradation: {df['final_degradation'].min():.2%}")
        print(f"  Max Degradation: {df['final_degradation'].max():.2%}")

    print(f"\nOutput saved to: {args.output_file}")

if __name__ == "__main__":
    main()