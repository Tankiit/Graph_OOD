#!/usr/bin/env python3
"""
Visualization script for spectral monitoring results
Generates comprehensive plots from experiment data
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import argparse
import json
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize spectral monitoring results')
    parser.add_argument('--experiment_dir', type=str, required=True,
                      help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save visualizations')
    parser.add_argument('--file_pattern', type=str, default='*.pkl',
                      help='Pattern to match result files')
    return parser.parse_args()

def load_experiment_results(experiment_dir, pattern='*.pkl'):
    """Load all experiment result files"""
    results = {}
    result_files = glob.glob(os.path.join(experiment_dir, pattern))

    for file_path in result_files:
        exp_name = os.path.basename(file_path).replace('.pkl', '')
        try:
            with open(file_path, 'rb') as f:
                results[exp_name] = pickle.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return results

def extract_spectral_metrics(results):
    """Extract spectral gap metrics from results"""
    metrics = []

    for exp_name, data in results.items():
        if 'spectral_gaps' not in data:
            continue

        # Parse experiment name
        parts = exp_name.split('_')
        id_data = parts[0]
        wild_data = parts[1]
        test_data = parts[2]
        method = parts[3]
        pi_values = '_'.join(parts[4:])

        # Extract metrics for each epoch
        for gap_data in data['spectral_gaps']:
            metrics.append({
                'experiment': exp_name,
                'id_data': id_data,
                'wild_data': wild_data,
                'test_data': test_data,
                'method': method,
                'pi_values': pi_values,
                'epoch': gap_data.get('epoch', 0),
                'gap_pure': gap_data.get('gap_pure_in', 0),
                'gap_mixed': gap_data.get('gap_mixed', 0),
                'degradation': gap_data.get('degradation', 0),
                'wild_ratio': gap_data.get('wild_ratio', 0),
                'pi_1': gap_data.get('effective_pi1', 0),
                'pi_2': gap_data.get('effective_pi2', 0)
            })

    return pd.DataFrame(metrics)

def plot_spectral_evolution(df, output_dir):
    """Plot spectral gap evolution over epochs"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Spectral Gap Evolution During Training', fontsize=16)

    # Group by method
    for method in df['method'].unique():
        method_df = df[df['method'] == method]

        # Plot 1: Gap evolution
        ax = axes[0, 0]
        for exp in method_df['experiment'].unique():
            exp_df = method_df[method_df['experiment'] == exp]
            ax.plot(exp_df['epoch'], exp_df['gap_pure'],
                   alpha=0.7, label=f"{method}_{exp[:20]}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Pure IN Gap')
        ax.set_title('Pure In-Distribution Gap Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # Plot 2: Mixed gap
        ax = axes[0, 1]
        for exp in method_df['experiment'].unique():
            exp_df = method_df[method_df['experiment'] == exp]
            ax.plot(exp_df['epoch'], exp_df['gap_mixed'], alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mixed Gap')
        ax.set_title('Mixed Data Gap Evolution')

        # Plot 3: Degradation
        ax = axes[1, 0]
        for exp in method_df['experiment'].unique():
            exp_df = method_df[method_df['experiment'] == exp]
            ax.plot(exp_df['epoch'], exp_df['degradation'] * 100, alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Degradation (%)')
        ax.set_title('Spectral Degradation Over Time')
        ax.axhline(y=30, color='r', linestyle='--', label='Threshold (30%)')

        # Plot 4: Wild ratio
        ax = axes[1, 1]
        for exp in method_df['experiment'].unique():
            exp_df = method_df[method_df['experiment'] == exp]
            ax.plot(exp_df['epoch'], exp_df['wild_ratio'], alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Wild Ratio')
        ax.set_title('Wild/IN Ratio')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectral_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved spectral evolution plot")

def plot_pi_impact_heatmap(df, output_dir):
    """Create heatmap showing impact of pi values on spectral degradation"""

    # Calculate average degradation for each pi combination
    pi_impact = df.groupby(['pi_1', 'pi_2'])['degradation'].mean().reset_index()

    # Create pivot table
    pivot = pi_impact.pivot(index='pi_2', columns='pi_1', values='degradation')

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.2%', cmap='RdYlGn_r',
                center=0.3, vmin=0, vmax=1,
                cbar_kws={'label': 'Average Degradation'})
    ax.set_title('Impact of π₁ and π₂ on Spectral Degradation')
    ax.set_xlabel('π₁ (Covariate Shift)')
    ax.set_ylabel('π₂ (Semantic Shift)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pi_impact_heatmap.png'), dpi=300)
    plt.close()
    print(f"✓ Saved pi impact heatmap")

def plot_method_comparison(df, output_dir):
    """Compare methods across different metrics"""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig)

    # Calculate final metrics (last epoch)
    final_df = df.groupby(['experiment', 'method']).last().reset_index()

    # Plot 1: Final gap comparison
    ax1 = fig.add_subplot(gs[0, :])
    sns.boxplot(data=final_df, x='method', y='gap_mixed', ax=ax1)
    ax1.set_title('Final Mixed Gap by Method')
    ax1.set_ylabel('Spectral Gap')

    # Plot 2: Degradation comparison
    ax2 = fig.add_subplot(gs[1, 0])
    sns.violinplot(data=final_df, x='method', y='degradation', ax=ax2)
    ax2.set_title('Degradation Distribution')
    ax2.set_ylabel('Degradation')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

    # Plot 3: ID dataset comparison
    ax3 = fig.add_subplot(gs[1, 1])
    id_comparison = final_df.groupby(['id_data', 'method'])['gap_mixed'].mean().unstack()
    id_comparison.plot(kind='bar', ax=ax3)
    ax3.set_title('Gap by ID Dataset')
    ax3.set_ylabel('Average Gap')
    ax3.legend(title='Method')

    # Plot 4: Wild dataset impact
    ax4 = fig.add_subplot(gs[1, 2])
    wild_comparison = final_df.groupby(['wild_data', 'method'])['degradation'].mean().unstack()
    wild_comparison.plot(kind='bar', ax=ax4)
    ax4.set_title('Degradation by Wild Dataset')
    ax4.set_ylabel('Average Degradation')
    ax4.legend(title='Method')

    # Plot 5: Time series comparison
    ax5 = fig.add_subplot(gs[2, :])
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        avg_degradation = method_df.groupby('epoch')['degradation'].mean()
        ax5.plot(avg_degradation.index, avg_degradation.values,
                label=method, linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Average Degradation')
    ax5.set_title('Average Degradation Over Training')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.suptitle('Method Comparison - Spectral Metrics', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved method comparison plots")

def plot_performance_vs_spectral(results, output_dir):
    """Plot OOD performance metrics vs spectral gap"""
    metrics = []

    for exp_name, data in results.items():
        if 'spectral_gaps' in data and 'fpr95_test' in data:
            # Get final epoch data
            final_gap_data = data['spectral_gaps'][-1] if data['spectral_gaps'] else {}
            final_fpr95 = data['fpr95_test'][-1] if data['fpr95_test'] else None
            final_auroc = data['auroc_test'][-1] if 'auroc_test' in data and data['auroc_test'] else None

            if final_fpr95 is not None:
                metrics.append({
                    'experiment': exp_name,
                    'gap_mixed': final_gap_data.get('gap_mixed', 0),
                    'degradation': final_gap_data.get('degradation', 0),
                    'fpr95': final_fpr95,
                    'auroc': final_auroc
                })

    if not metrics:
        print("No performance metrics found")
        return

    perf_df = pd.DataFrame(metrics)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: FPR95 vs Gap
    ax = axes[0, 0]
    ax.scatter(perf_df['gap_mixed'], perf_df['fpr95'], alpha=0.6)
    z = np.polyfit(perf_df['gap_mixed'], perf_df['fpr95'], 1)
    p = np.poly1d(z)
    ax.plot(perf_df['gap_mixed'].sort_values(),
           p(perf_df['gap_mixed'].sort_values()),
           "r-", alpha=0.8, label=f'Trend')
    ax.set_xlabel('Spectral Gap (Mixed)')
    ax.set_ylabel('FPR95')
    ax.set_title('OOD Detection vs Spectral Gap')
    ax.legend()

    # Plot 2: FPR95 vs Degradation
    ax = axes[0, 1]
    ax.scatter(perf_df['degradation'], perf_df['fpr95'], alpha=0.6)
    ax.set_xlabel('Spectral Degradation')
    ax.set_ylabel('FPR95')
    ax.set_title('OOD Detection vs Degradation')

    # Plot 3: AUROC vs Gap
    if perf_df['auroc'].notna().any():
        ax = axes[1, 0]
        ax.scatter(perf_df['gap_mixed'], perf_df['auroc'], alpha=0.6)
        ax.set_xlabel('Spectral Gap (Mixed)')
        ax.set_ylabel('AUROC')
        ax.set_title('AUROC vs Spectral Gap')

    # Plot 4: Correlation matrix
    ax = axes[1, 1]
    corr_matrix = perf_df[['gap_mixed', 'degradation', 'fpr95']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Matrix')

    plt.suptitle('Performance vs Spectral Metrics', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_vs_spectral.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved performance vs spectral plots")

def create_summary_report(df, results, output_dir):
    """Create a text summary report"""
    report_path = os.path.join(output_dir, 'spectral_analysis_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SPECTRAL MONITORING ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total experiments: {len(results)}\n")
        f.write(f"Total epochs analyzed: {len(df)}\n")
        f.write(f"Methods tested: {', '.join(df['method'].unique())}\n")
        f.write(f"ID datasets: {', '.join(df['id_data'].unique())}\n")
        f.write(f"Wild datasets: {', '.join(df['wild_data'].unique())}\n\n")

        # Best configurations
        f.write("BEST CONFIGURATIONS (Minimum Degradation)\n")
        f.write("-"*40 + "\n")

        final_df = df.groupby('experiment').last().reset_index()
        best_configs = final_df.nsmallest(5, 'degradation')

        for idx, row in best_configs.iterrows():
            f.write(f"\n{idx+1}. {row['experiment'][:50]}\n")
            f.write(f"   Degradation: {row['degradation']:.2%}\n")
            f.write(f"   Gap (mixed): {row['gap_mixed']:.6f}\n")
            f.write(f"   π₁={row['pi_1']:.2f}, π₂={row['pi_2']:.2f}\n")

        # Method comparison
        f.write("\n\nMETHOD COMPARISON\n")
        f.write("-"*40 + "\n")

        method_stats = final_df.groupby('method').agg({
            'degradation': ['mean', 'std', 'min', 'max'],
            'gap_mixed': ['mean', 'std']
        })

        f.write(method_stats.to_string())

        # Pi impact summary
        f.write("\n\nPI IMPACT SUMMARY\n")
        f.write("-"*40 + "\n")

        pi_stats = final_df.groupby(['pi_1', 'pi_2']).agg({
            'degradation': 'mean',
            'gap_mixed': 'mean'
        }).round(4)

        f.write(pi_stats.to_string())

        # Dataset-specific insights
        f.write("\n\nDATASET-SPECIFIC INSIGHTS\n")
        f.write("-"*40 + "\n")

        for id_data in df['id_data'].unique():
            id_df = final_df[final_df['id_data'] == id_data]
            f.write(f"\n{id_data}:\n")
            f.write(f"  Avg degradation: {id_df['degradation'].mean():.2%}\n")
            f.write(f"  Best wild data: {id_df.loc[id_df['degradation'].idxmin(), 'wild_data']}\n")
            f.write(f"  Worst wild data: {id_df.loc[id_df['degradation'].idxmax(), 'wild_data']}\n")

    print(f"✓ Saved analysis report to {report_path}")

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading experiment results...")
    results = load_experiment_results(args.experiment_dir, args.file_pattern)

    if not results:
        print("No results found!")
        return

    print(f"Loaded {len(results)} experiments")

    # Extract spectral metrics
    print("Extracting spectral metrics...")
    df = extract_spectral_metrics(results)

    if df.empty:
        print("No spectral metrics found in results!")
        return

    print(f"Extracted {len(df)} data points")

    # Generate visualizations
    print("Generating visualizations...")

    try:
        plot_spectral_evolution(df, args.output_dir)
    except Exception as e:
        print(f"Error in spectral evolution plot: {e}")

    try:
        plot_pi_impact_heatmap(df, args.output_dir)
    except Exception as e:
        print(f"Error in pi impact heatmap: {e}")

    try:
        plot_method_comparison(df, args.output_dir)
    except Exception as e:
        print(f"Error in method comparison: {e}")

    try:
        plot_performance_vs_spectral(results, args.output_dir)
    except Exception as e:
        print(f"Error in performance vs spectral plot: {e}")

    # Create summary report
    try:
        create_summary_report(df, results, args.output_dir)
    except Exception as e:
        print(f"Error creating report: {e}")

    # Save processed data
    df.to_csv(os.path.join(args.output_dir, 'spectral_metrics.csv'), index=False)
    print(f"✓ Saved processed metrics to spectral_metrics.csv")

    print(f"\nVisualization complete! Check {args.output_dir} for outputs.")

if __name__ == "__main__":
    main()