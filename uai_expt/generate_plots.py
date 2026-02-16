"""
Generate plots from evaluation results
Creates the gamma vs AUROC plots (the "money figure")
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup plotting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load results
results_df = pd.read_csv('results/evaluation_results.csv')

# Create plots directory
Path('plots').mkdir(exist_ok=True)

# ============================================================
# Plot 1: γ vs AUROC (The "Money Figure")
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(r'$\hat{\gamma}$ vs AUROC: Feature Separation Ratio vs OOD Detection Performance',
             fontsize=16, fontweight='bold')

detectors = ['auroc_energy', 'auroc_mahalanobis', 'auroc_knn', 'auroc_spectral']
detector_names = ['Energy', 'Mahalanobis', 'k-NN', 'Spectral']

for idx, (detector, name) in enumerate(zip(detectors, detector_names)):
    ax = axes[idx // 2, idx % 2]

    # Group by architecture
    for arch in ['wrn', 'clip', 'dinov2']:
        arch_data = results_df[results_df['architecture'] == arch]

        if len(arch_data) > 0:
            arch_labels = {'wrn': 'WRN-40-2 (SAL)', 'clip': 'CLIP ViT-L', 'dinov2': 'DINOv2 ViT-L'}
            ax.scatter(arch_data['gamma'], arch_data[detector],
                      label=arch_labels[arch], s=200, alpha=0.7, edgecolors='black', linewidth=2)

    ax.set_xlabel(r'$\hat{\gamma} = \hat{\delta}/\hat{\epsilon}$', fontsize=14)
    ax.set_ylabel(f'{name} AUROC', fontsize=14)
    ax.set_title(f'{name} OOD Detector', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Add correlation coefficient
    all_gamma = results_df['gamma'].values
    all_auroc = results_df[detector].values
    corr = np.corrcoef(all_gamma, all_auroc)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('plots/gamma_vs_auroc.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/gamma_vs_auroc.png")

# ============================================================
# Plot 2: η vs AUROC
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(r'$\hat{\eta}$ vs AUROC: Cross-edge Fraction vs OOD Detection Performance',
             fontsize=16, fontweight='bold')

for idx, (detector, name) in enumerate(zip(detectors, detector_names)):
    ax = axes[idx // 2, idx % 2]

    for arch in ['wrn', 'clip', 'dinov2']:
        arch_data = results_df[results_df['architecture'] == arch]

        if len(arch_data) > 0:
            arch_labels = {'wrn': 'WRN-40-2 (SAL)', 'clip': 'CLIP ViT-L', 'dinov2': 'DINOv2 ViT-L'}
            ax.scatter(arch_data['eta'], arch_data[detector],
                      label=arch_labels[arch], s=200, alpha=0.7, edgecolors='black', linewidth=2)

    ax.set_xlabel(r'$\hat{\eta}$ (Cross-edge Fraction)', fontsize=14)
    ax.set_ylabel(f'{name} AUROC', fontsize=14)
    ax.set_title(f'{name} OOD Detector', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Add correlation coefficient
    all_eta = results_df['eta'].values
    all_auroc = results_df[detector].values
    corr = np.corrcoef(all_eta, all_auroc)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('plots/eta_vs_auroc.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/eta_vs_auroc.png")

# ============================================================
# Plot 3: λ₂ vs AUROC
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(r'$\lambda_2$ vs AUROC: Algebraic Connectivity vs OOD Detection Performance',
             fontsize=16, fontweight='bold')

for idx, (detector, name) in enumerate(zip(detectors, detector_names)):
    ax = axes[idx // 2, idx % 2]

    for arch in ['wrn', 'clip', 'dinov2']:
        arch_data = results_df[results_df['architecture'] == arch]

        if len(arch_data) > 0:
            arch_labels = {'wrn': 'WRN-40-2 (SAL)', 'clip': 'CLIP ViT-L', 'dinov2': 'DINOv2 ViT-L'}
            ax.scatter(arch_data['lambda2_id'], arch_data[detector],
                      label=arch_labels[arch], s=200, alpha=0.7, edgecolors='black', linewidth=2)

    ax.set_xlabel(r'$\lambda_2^{ID}$ (Algebraic Connectivity)', fontsize=14)
    ax.set_ylabel(f'{name} AUROC', fontsize=14)
    ax.set_title(f'{name} OOD Detector', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Add correlation coefficient
    all_lambda2 = results_df['lambda2_id'].values
    all_auroc = results_df[detector].values
    corr = np.corrcoef(all_lambda2, all_auroc)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('plots/lambda2_vs_auroc.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/lambda2_vs_auroc.png")

# ============================================================
# Summary Table
# ============================================================
print("\n" + "="*80)
print("EVALUATION RESULTS SUMMARY")
print("="*80)

# Group by architecture
for arch in ['wrn', 'clip', 'dinov2']:
    arch_name = {'wrn': 'WRN-40-2 (SAL)', 'clip': 'CLIP ViT-L', 'dinov2': 'DINOv2 ViT-L'}[arch]
    arch_data = results_df[results_df['architecture'] == arch]

    print(f"\n{arch_name}:")
    print("-" * 60)

    for _, row in arch_data.iterrows():
        print(f"\n  {row['id_dataset'].upper()} vs {row['ood_dataset'].upper()}:")
        print(f"    γ̂ = {row['gamma']:.4f}, η̂ = {row['eta']:.4f}")
        print(f"    λ₂(ID) = {row['lambda2_id']:.6f}, λ₂(OOD) = {row['lambda2_ood']:.6f}")
        print(f"    V1 holds: {row['V1_holds']}, V2 holds: {row['V2_holds']}")
        print(f"    AUROCs -> Energy: {row['auroc_energy']:.4f}, "
              f"Mahalanobis: {row['auroc_mahalanobis']:.4f}, "
              f"k-NN: {row['auroc_knn']:.4f}, "
              f"Spectral: {row['auroc_spectral']:.4f}")

print("\n" + "="*80)
print("✓ All plots generated successfully!")
print("="*80)
print("\nGenerated files:")
print("  - plots/gamma_vs_auroc.png (γ̂ vs AUROC - THE MONEY FIGURE)")
print("  - plots/eta_vs_auroc.png (η̂ vs AUROC)")
print("  - plots/lambda2_vs_auroc.png (λ₂ vs AUROC)")
print("  - results/evaluation_results.csv (complete results)")
