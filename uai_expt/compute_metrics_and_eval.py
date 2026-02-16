"""
compute_metrics_and_eval.py
Compute γ, η, λ₂ from features and run OOD detection evaluation.

This script:
1. Loads extracted features from all architectures
2. Computes γ̂ = δ̂/ε̂ (feature separation)
3. Computes η̂ (cross-edge fraction)
4. Computes λ₂ for ID, OOD, and joint graphs
5. Runs SAL/Energy/Mahalanobis/k-NN OOD detection
6. Generates γ vs AUROC plots
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import MinCovDet
from scipy.spatial.distance import cdist
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags as spdiags
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import logging
import argparse
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path('/Users/tanmoy/research/data')
FEATURE_DIR = Path('./features')
RESULTS_DIR = Path('./results')
RESULTS_DIR.mkdir(exist_ok=True)

PLOT_DIR = Path('./plots')
PLOT_DIR.mkdir(exist_ok=True)

# Datasets
ID_DATASETS = ['cifar10', 'cifar100']
OOD_DATASETS = ['svhn', 'textures', 'lsun_c']

# Architectures
ARCHITECTURES = ['wrn', 'clip', 'dinov2']

# ============================================================
# Feature Loading
# ============================================================

def load_features(dataset_name, architecture, split='test'):
    """Load features from disk"""
    arch_name = architecture.lower()
    load_path = FEATURE_DIR / f'{dataset_name}_{split}_{arch_name}.npz'

    if not load_path.exists():
        logger.warning(f"Features not found: {load_path}")
        return None, None

    data = np.load(load_path)
    features = data['features']
    labels = data['labels']

    logger.info(f"Loaded {len(features)} features from {load_path}")
    return features, labels


# ============================================================
# Metric Computation
# ============================================================

def compute_verification_conditions(features_in, features_out, k=10):
    """
    Compute V1 (γ) and V2 (η) from feature arrays.

    Args:
        features_in: In-distribution features (N_in x D)
        features_out: Out-of-distribution features (N_out x D)
        k: Number of neighbors for k-NN graph

    Returns:
        dict with gamma, delta, epsilon, eta, lambda2_in, lambda2_out, lambda2_joint
    """
    n_in, n_out = len(features_in), len(features_out)

    # --- V1: Feature Separation ---
    # ε: intra-class spread (average k-NN distance)
    logger.info("Computing ε (intra-class spread)...")

    nn_in = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(features_in)
    dists_in, _ = nn_in.kneighbors(features_in)
    eps_in = np.median(dists_in[:, k])  # k-th neighbor distance, median

    nn_out = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(features_out)
    dists_out, _ = nn_out.kneighbors(features_out)
    eps_out = np.median(dists_out[:, k])

    eps = max(eps_in, eps_out)

    # δ: inter-manifold distance (5th percentile for robustness)
    logger.info("Computing δ (inter-manifold distance)...")

    n_sub = min(2000, n_in, n_out)
    idx_in = np.random.choice(n_in, n_sub, replace=False)
    idx_out = np.random.choice(n_out, n_sub, replace=False)

    cross = cdist(features_in[idx_in], features_out[idx_out], metric='cosine')
    delta = np.percentile(cross.min(axis=1), 5)

    gamma = delta / eps if eps > 0 else float('inf')

    # --- V2: Neighborhood Preservation (cross-edge fraction) ---
    logger.info("Computing η (cross-edge fraction)...")

    features_joint = np.vstack([features_in, features_out])
    labels_joint = np.array([0]*n_in + [1]*n_out)

    nn_joint = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn_joint.fit(features_joint)
    _, indices = nn_joint.kneighbors(features_joint)

    cross_edges = 0
    total_edges = 0
    for i in range(len(features_joint)):
        for j in indices[i, 1:]:  # skip self
            total_edges += 1
            if labels_joint[i] != labels_joint[j]:
                cross_edges += 1

    eta = cross_edges / total_edges if total_edges > 0 else 0

    # --- Spectral: λ₂ for each graph ---
    logger.info("Computing λ₂ (algebraic connectivity)...")

    lambda2_in = compute_lambda2(features_in, k)
    lambda2_out = compute_lambda2(features_out, k)
    lambda2_joint = compute_lambda2(features_joint, k)

    return {
        'gamma': gamma,
        'delta': delta,
        'epsilon': eps,
        'eps_in': eps_in,
        'eps_out': eps_out,
        'eta': eta,
        'lambda2_in': lambda2_in,
        'lambda2_out': lambda2_out,
        'lambda2_joint': lambda2_joint,
        'V1_holds': gamma > 1.0,
        'V2_holds': eta < 0.5,
    }


def compute_lambda2(features, k=10):
    """
    Compute algebraic connectivity of k-NN graph.

    Args:
        features: Feature matrix (N x D)
        k: Number of neighbors

    Returns:
        λ₂ (second smallest eigenvalue of normalized Laplacian)
    """
    n = len(features)

    # Handle small graphs
    if n < k + 1:
        k = max(3, n // 2)

    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(features)
    dists, indices = nn.kneighbors(features)

    # Build adjacency (symmetric)
    W = lil_matrix((n, n))
    for i in range(n):
        for j_idx in range(1, k+1):
            j = indices[i, j_idx]
            sim = max(1 - dists[i, j_idx], 1e-8)  # cosine similarity
            W[i, j] = sim
            W[j, i] = sim

    W = W.tocsr()
    D = np.array(W.sum(axis=1)).flatten()
    D_sp = spdiags(D, 0, shape=(n, n))

    # Unnormalized Laplacian
    L = D_sp - W

    # For normalized: use generalized eigenvalue problem
    # L v = λ D v
    try:
        eigenvalues = eigsh(L.tocsc(), k=2, M=D_sp.tocsc(), sigma=0,
                           which='LM', return_eigenvectors=False)
        eigenvalues.sort()
        return eigenvalues[1]  # λ₂
    except:
        # Fallback for small graphs
        logger.warning(f"Failed to compute λ₂ for graph with {n} nodes")
        return 0.0


# ============================================================
# OOD Detection Methods
# ============================================================

class OODDetector:
    """Base class for OOD detection"""

    def __init__(self, features_id, labels_id):
        self.features_id = features_id
        self.labels_id = labels_id

    def compute_scores(self, features_ood):
        """Compute OOD scores (higher = more OOD)"""
        raise NotImplementedError


class EnergyDetector(OODDetector):
    """Energy-based OOD detection"""

    def __init__(self, features_id, labels_id, temperature=1.0):
        super().__init__(features_id, labels_id)
        self.temperature = temperature

        # Fit a simple Gaussian model to ID features
        self.mean = np.mean(features_id, axis=0)
        self.cov = np.cov(features_id.T)
        self.cov_inv = np.linalg.inv(self.cov + np.eye(len(self.mean)) * 1e-6)

    def compute_scores(self, features_ood):
        # Negative log-likelihood as OOD score
        diff = features_ood - self.mean
        scores = np.sum(diff @ self.cov_inv * diff, axis=1)
        return scores


class MahalanobisDetector(OODDetector):
    """Mahalanobis distance-based OOD detection"""

    def __init__(self, features_id, labels_id):
        super().__init__(features_id, labels_id)

        # Class-conditional Mahalanobis
        self.classes = np.unique(labels_id)
        self.class_params = {}

        for c in self.classes:
            mask = labels_id == c
            features_c = features_id[mask]

            self.class_params[c] = {
                'mean': np.mean(features_c, axis=0),
                'cov_inv': self._compute_cov_inv(features_c)
            }

    def _compute_cov_inv(self, features, reg=1e-6):
        """Compute regularized inverse covariance"""
        cov = np.cov(features.T)
        cov += np.eye(cov.shape[0]) * reg
        return np.linalg.inv(cov)

    def compute_scores(self, features_ood):
        scores = []

        for feat in features_ood:
            min_score = float('inf')

            for c in self.classes:
                params = self.class_params[c]
                diff = feat - params['mean']
                score = diff @ params['cov_inv'] @ diff.T
                min_score = min(min_score, score)

            scores.append(min_score)

        return np.array(scores)


class KNNDetector(OODDetector):
    """k-NN distance-based OOD detection"""

    def __init__(self, features_id, labels_id, k=5):
        super().__init__(features_id, labels_id)
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k, metric='cosine')
        self.nn.fit(features_id)

    def compute_scores(self, features_ood):
        dists, _ = self.nn.kneighbors(features_ood)
        # Use average distance to k nearest neighbors
        return dists.mean(axis=1)


class SpectralDetector(OODDetector):
    """
    Spectral Graph-based OOD Detection (SAL)
    Similar to the method in the SAL paper
    """

    def __init__(self, features_id, labels_id, k=10):
        super().__init__(features_id, labels_id)
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
        self.nn.fit(features_id)

        # Build ID graph and compute spectral properties
        self.lambda2_id = compute_lambda2(features_id, k)

    def compute_scores(self, features_ood):
        scores = []

        for feat in features_ood:
            # Temporarily add this sample to ID set
            features_temp = np.vstack([self.features_id, feat.reshape(1, -1)])

            # Compute new λ₂
            lambda2_new = compute_lambda2(features_temp, self.k)

            # Score based on change in λ₂
            # Larger decrease = more likely OOD
            score = self.lambda2_id - lambda2_new
            scores.append(score)

        return np.array(scores)


def compute_auroc(scores_id, scores_ood):
    """Compute AUROC for OOD detection"""
    # ID scores should be low, OOD scores should be high
    labels = np.array([0]*len(scores_id) + [1]*len(scores_ood))
    scores = np.concatenate([scores_id, scores_ood])

    try:
        auroc = roc_auc_score(labels, scores)
        return auroc
    except:
        return None


# ============================================================
# Main Evaluation Pipeline
# ============================================================

def evaluate_pair(id_dataset, ood_dataset, architecture):
    """
    Evaluate OOD detection for a specific (ID, OOD, architecture) combination.

    Returns:
        dict with metrics and AUROCs
    """
    logger.info(f"\nEvaluating {id_dataset} vs {ood_dataset} ({architecture})")

    # Load features
    feat_id, labels_id = load_features(id_dataset, architecture)
    feat_ood, labels_ood = load_features(ood_dataset, architecture)

    if feat_id is None or feat_ood is None:
        logger.warning(f"Skipping {id_dataset} vs {ood_dataset} ({architecture}) - missing features")
        return None

    # Compute verification conditions
    logger.info(f"Computing γ, η, λ₂ for {id_dataset} vs {ood_dataset}")
    metrics = compute_verification_conditions(feat_id, feat_ood)

    # Run OOD detectors
    results = {
        'id_dataset': id_dataset,
        'ood_dataset': ood_dataset,
        'architecture': architecture,
        'gamma': metrics['gamma'],
        'eta': metrics['eta'],
        'lambda2_id': metrics['lambda2_in'],
        'lambda2_ood': metrics['lambda2_out'],
        'lambda2_joint': metrics['lambda2_joint'],
        'V1_holds': metrics['V1_holds'],
        'V2_holds': metrics['V2_holds'],
    }

    # Energy
    logger.info("Running Energy detector...")
    detector = EnergyDetector(feat_id, labels_id)
    scores_id = detector.compute_scores(feat_id)
    scores_ood = detector.compute_scores(feat_ood)
    results['auroc_energy'] = compute_auroc(scores_id, scores_ood)

    # Mahalanobis
    logger.info("Running Mahalanobis detector...")
    detector = MahalanobisDetector(feat_id, labels_id)
    scores_id = detector.compute_scores(feat_id)
    scores_ood = detector.compute_scores(feat_ood)
    results['auroc_mahalanobis'] = compute_auroc(scores_id, scores_ood)

    # k-NN
    logger.info("Running k-NN detector...")
    detector = KNNDetector(feat_id, labels_id, k=5)
    scores_id = detector.compute_scores(feat_id)
    scores_ood = detector.compute_scores(feat_ood)
    results['auroc_knn'] = compute_auroc(scores_id, scores_ood)

    # Spectral (SAL)
    logger.info("Running Spectral detector...")
    detector = SpectralDetector(feat_id, labels_id, k=10)
    scores_id = detector.compute_scores(feat_id[:min(100, len(feat_id))])  # Subsample for speed
    scores_ood = detector.compute_scores(feat_ood[:min(100, len(feat_ood))])
    results['auroc_spectral'] = compute_auroc(scores_id, scores_ood)

    logger.info(f"AUROCs - Energy: {results['auroc_energy']:.4f}, "
               f"Mahalanobis: {results['auroc_mahalanobis']:.4f}, "
               f"k-NN: {results['auroc_knn']:.4f}, "
               f"Spectral: {results['auroc_spectral']:.4f}")

    return results


def run_all_evaluations():
    """Run evaluation for all (ID, OOD, architecture) combinations"""
    all_results = []

    for arch in ARCHITECTURES:
        for id_ds in ID_DATASETS:
            for ood_ds in OOD_DATASETS:
                result = evaluate_pair(id_ds, ood_ds, arch)

                if result:
                    all_results.append(result)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / 'evaluation_results.csv', index=False)

    # Save as JSON
    with open(RESULTS_DIR / 'evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {RESULTS_DIR}")

    return results_df


# ============================================================
# Plotting
# ============================================================

def plot_gamma_vs_auroc(results_df):
    """
    Create the key plot: γ vs AUROC
    This is the "money figure" from the SAL paper
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(r'$\hat{\gamma}$ vs AUROC: Feature Separation and OOD Detection Performance',
                 fontsize=16, fontweight='bold')

    detectors = ['auroc_energy', 'auroc_mahalanobis', 'auroc_knn', 'auroc_spectral']
    titles = ['Energy', 'Mahalanobis', 'k-NN', 'Spectral']

    for ax, detector, title in zip(axes.flat, detectors, titles):
        for arch in ARCHITECTURES:
            arch_data = results_df[results_df['architecture'] == arch]

            if len(arch_data) > 0:
                ax.scatter(arch_data['gamma'], arch_data[detector],
                          label=arch.upper(), s=100, alpha=0.7)

        ax.set_xlabel(r'$\hat{\gamma} = \hat{\delta} / \hat{\epsilon}$', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')

    plt.tight_layout()
    save_path = PLOT_DIR / 'gamma_vs_auroc.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {save_path}")

    return fig


def plot_eta_vs_auroc(results_df):
    """Plot η vs AUROC"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(r'$\hat{\eta}$ vs AUROC: Cross-edge Fraction and OOD Detection',
                 fontsize=16, fontweight='bold')

    detectors = ['auroc_energy', 'auroc_mahalanobis', 'auroc_knn', 'auroc_spectral']
    titles = ['Energy', 'Mahalanobis', 'k-NN', 'Spectral']

    for ax, detector, title in zip(axes.flat, detectors, titles):
        for arch in ARCHITECTURES:
            arch_data = results_df[results_df['architecture'] == arch]

            if len(arch_data) > 0:
                ax.scatter(arch_data['eta'], arch_data[detector],
                          label=arch.upper(), s=100, alpha=0.7)

        ax.set_xlabel(r'$\hat{\eta}$ (cross-edge fraction)', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')

    plt.tight_layout()
    save_path = PLOT_DIR / 'eta_vs_auroc.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {save_path}")

    return fig


def plot_lambda2_vs_auroc(results_df):
    """Plot λ₂ vs AUROC"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(r'$\lambda_2$ vs AUROC: Spectral Gap and OOD Detection',
                 fontsize=16, fontweight='bold')

    detectors = ['auroc_energy', 'auroc_mahalanobis', 'auroc_knn', 'auroc_spectral']
    titles = ['Energy', 'Mahalanobis', 'k-NN', 'Spectral']

    for ax, detector, title in zip(axes.flat, detectors, titles):
        for arch in ARCHITECTURES:
            arch_data = results_df[results_df['architecture'] == arch]

            if len(arch_data) > 0:
                ax.scatter(arch_data['lambda2_joint'], arch_data[detector],
                          label=arch.upper(), s=100, alpha=0.7)

        ax.set_xlabel(r'$\lambda_2$ (joint graph)', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')

    plt.tight_layout()
    save_path = PLOT_DIR / 'lambda2_vs_auroc.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {save_path}")

    return fig


def main():
    """Main evaluation and plotting pipeline"""

    logger.info("\n" + "="*60)
    logger.info("UAI OOD Detection Evaluation Pipeline")
    logger.info("="*60)

    # Run all evaluations
    results_df = run_all_evaluations()

    # Generate plots
    logger.info("\nGenerating plots...")

    plot_gamma_vs_auroc(results_df)
    plot_eta_vs_auroc(results_df)
    plot_lambda2_vs_auroc(results_df)

    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*60)

    for arch in ARCHITECTURES:
        arch_data = results_df[results_df['architecture'] == arch]

        if len(arch_data) > 0:
            logger.info(f"\n{arch.upper()}:")
            logger.info(f"  Mean γ: {arch_data['gamma'].mean():.4f} ± {arch_data['gamma'].std():.4f}")
            logger.info(f"  Mean η: {arch_data['eta'].mean():.4f} ± {arch_data['eta'].std():.4f}")
            logger.info(f"  Mean AUROC (Energy): {arch_data['auroc_energy'].mean():.4f}")
            logger.info(f"  Mean AUROC (Mahalanobis): {arch_data['auroc_mahalanobis'].mean():.4f}")
            logger.info(f"  Mean AUROC (k-NN): {arch_data['auroc_knn'].mean():.4f}")
            logger.info(f"  Mean AUROC (Spectral): {arch_data['auroc_spectral'].mean():.4f}")

    logger.info("\n" + "="*60)
    logger.info("Evaluation complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
