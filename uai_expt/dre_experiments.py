"""
UAI 2026 — Density Ratio Estimation under Wild Contamination
Experiments (i) Provable Mixtures and (ii) Deployment Risk Quantification

Usage:
    python dre_experiments.py --features_dir ./features --output_dir ./results/dre

Expects .npz files in features_dir with keys 'features' and optionally 'labels'.
Naming convention: {dataset}_test_{arch}.npz  (e.g., cifar10_test_clip.npz)
"""

import numpy as np
import os
import argparse
import json
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_score
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Core DRE utilities
# =============================================================================

def load_features(path):
    """Load features from .npz file."""
    data = np.load(path, allow_pickle=True)
    # Try common key names
    for key in ['features', 'feat', 'embeddings', 'data', 'arr_0']:
        if key in data:
            return data[key]
    # Fall back to first array
    return data[data.files[0]]


# -------------------------------
# Utilities
# -------------------------------

def _kth_radius(nn: NearestNeighbors, X_query: np.ndarray, k: int, drop_self: bool) -> np.ndarray:
    """
    Return distance to the k-th neighbor. If drop_self=True, we query k+1
    and drop the first neighbor when it is (near-)zero.
    """
    n_req = k + 1 if drop_self else k
    dists, _ = nn.kneighbors(X_query, n_neighbors=n_req, return_distance=True)

    if not drop_self:
        return dists[:, -1]

    # If the first neighbor is self (distance ~0), drop it; else keep neighbors as-is.
    first = dists[:, 0]
    is_self = first <= 1e-12
    # default: take k-th among k+1 (index k)
    r = dists[:, k].copy()
    # if not self, then the k-th neighbor is at index k-1 in the returned list
    r[~is_self] = dists[~is_self, k-1]
    return r


class KNNDensityRatioScorer:
    """
    Scores x by an estimate monotone in r_wild(x)=p_wild(x)/p_in(x).

    score(x) = log( rho_in(x) / rho_wild(x) )
    (Log is numerically stable and preserves ranking.)
    """
    def __init__(self, D_in: np.ndarray, D_wild_ref: np.ndarray, k: int = 50, metric: str = "euclidean"):
        self.k = int(k)
        self.metric = metric

        self.nn_in = NearestNeighbors(n_neighbors=self.k + 1, metric=self.metric, algorithm="auto")
        self.nn_wild = NearestNeighbors(n_neighbors=self.k + 1, metric=self.metric, algorithm="auto")

        self.nn_in.fit(D_in)
        self.nn_wild.fit(D_wild_ref)

    def score(self, X: np.ndarray, drop_self_in: bool = False, drop_self_wild: bool = False) -> np.ndarray:
        rho_in = _kth_radius(self.nn_in, X, self.k, drop_self=drop_self_in)
        rho_wild = _kth_radius(self.nn_wild, X, self.k, drop_self=drop_self_wild)

        rho_in = np.maximum(rho_in, 1e-12)
        rho_wild = np.maximum(rho_wild, 1e-12)

        return np.log(rho_in) - np.log(rho_wild)   # higher => more OOD


def estimate_pi_from_id_floor(scores_id: np.ndarray, floor_q: float = 1.0) -> float:
    """
    Theory-aligned π estimator under Huber contamination:
      r_wild(z) = (1-π) + π r*(z)
    For deep-ID points where r*(z) ~ 0: r_wild(z) ~ (1-π).
    If score is monotone in r_wild, we need an *approximately calibrated* ratio.

    Here we assume we have a ratio-like quantity in linear scale:
      ratio_hat ~ r_wild
    If you use log scores, convert back by exp.

    We estimate:
      (1-π) ≈ low-quantile(ratio_hat on pure ID)
      π̂ = 1 - lowq
    """
    # Convert log-score back to "ratio-like" scale:
    ratio_hat = np.exp(scores_id)

    lowq = np.percentile(ratio_hat, floor_q)  # e.g., 1st percentile (more conservative than min)
    pi_hat = 1.0 - lowq
    return float(np.clip(pi_hat, 0.0, 0.99))


# Legacy functions for backward compatibility
def knn_density_ratio(z, D_in, D_wild, k=50):
    """
    Estimate r_wild(z) = p_wild(z) / p_in(z) via k-NN distance ratio.

    Returns rho_k(z; D_in) / rho_k(z; D_wild), which is proportional to
    r_wild(z)^{1/d} (monotone transform — preserves ranking for AUROC).

    Parameters
    ----------
    z : array (n_query, m) — points to score
    D_in : array (n_in, m) — in-distribution reference set
    D_wild : array (n_wild, m) — wild (deployment) data
    k : int — number of neighbors

    Returns
    -------
    scores : array (n_query,) — density ratio estimates (higher = more OOD)
    """
    scorer = KNNDensityRatioScorer(D_in, D_wild, k=k, metric='cosine')
    return scorer.score(z)


def estimate_pi(scores_id_cal, method='minimum'):
    """
    Estimate mixing proportion π from density ratio scores on ID calibration data.

    Under Huber model: r_wild(z) = (1-π) + π·r*(z)
    For z deep inside M_in: r*(z) ≈ 0, so r_wild(z) ≈ (1-π)
    Therefore: π̂ = 1 - min(r_wild over ID points)

    In practice with k-NN ratio (monotone transform), we use
    the quantile approach for robustness.

    Parameters
    ----------
    scores_id_cal : array — DRE scores on held-out ID calibration data
    method : str — 'minimum' or 'quantile'

    Returns
    -------
    pi_hat : float — estimated mixing proportion
    """
    # Use the new theory-aligned estimator
    return estimate_pi_from_id_floor(scores_id_cal, floor_q=1.0)


def compute_separation_ratio(D_in, D_out, k=50):
    """
    Compute separation ratio γ = δ / (6ε).

    δ: inter-manifold distance (5th percentile of cross-distances)
    ε: intra-manifold spread (average k-NN distance within ID)
    """
    # Intra-manifold spread: average k-NN distance within ID
    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(D_in)
    dists, _ = nn.kneighbors(D_in)
    epsilon = np.mean(dists[:, -1])

    # Inter-manifold distance: distances from OOD to ID
    dists_cross, _ = nn.kneighbors(D_out)
    delta = np.percentile(dists_cross[:, 0], 5)  # 5th percentile of nearest distances

    gamma = delta / (6 * epsilon + 1e-10)
    return gamma, delta, epsilon


def l2_normalize(X, eps=1e-12):
    """L2 normalize feature vectors."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


# =============================================================================
# Experiment (i): Provable Mixtures
# =============================================================================

def experiment_provable_mixtures(
    D_in_train, D_in_held, D_out,
    pi_values=(0.1, 0.2, 0.3, 0.4, 0.5),
    k=50, n_trials=5, seed=42,
    metric="euclidean"
):
    """
    Construct wild data with known π, estimate π̂, compare to ground truth.

    Improved version with proper train/val splits to avoid leakage.

    Parameters
    ----------
    D_in_train : array — ID training features (reference set)
    D_in_held : array — held-out ID features (to construct wild mixtures)
    D_out : array — OOD features
    pi_values : tuple — ground truth mixing proportions to test
    k : int — k-NN parameter
    n_trials : int — number of random trials per π
    seed : int — random seed
    metric : str — distance metric for k-NN

    Returns
    -------
    results : dict — {pi_true: {pi_hat_mean, pi_hat_std, error_mean, ...}}
    """
    rng = np.random.RandomState(seed)
    results = {}
    n_wild = min(len(D_in_held), len(D_out), 2000)

    # Pure ID calibration set for pi (held-out from D_in_held!)
    n_pi_cal = min(800, len(D_in_held))
    idx_pi_cal = rng.choice(len(D_in_held), size=n_pi_cal, replace=False)
    D_id_pi_cal = D_in_held[idx_pi_cal]

    for pi_true in pi_values:
        pi_hats = []

        for _ in range(n_trials):
            n_ood = int(pi_true * n_wild)
            n_id = n_wild - n_ood

            idx_id = rng.choice(len(D_in_held), size=n_id, replace=False)
            idx_ood = rng.choice(len(D_out), size=n_ood, replace=False)

            D_wild = np.vstack([D_in_held[idx_id], D_out[idx_ood]])
            labels_wild = np.concatenate([np.zeros(n_id), np.ones(n_ood)])

            perm = rng.permutation(len(D_wild))
            D_wild = D_wild[perm]
            labels_wild = labels_wild[perm]

            # Split wild into reference vs query to avoid self leakage
            split = len(D_wild) // 2
            D_wild_ref = D_wild[:split]
            D_wild_query = D_wild[split:]
            y_query = labels_wild[split:]

            scorer = KNNDensityRatioScorer(D_in=D_in_train, D_wild_ref=D_wild_ref, k=k, metric=metric)

            # scores on pure ID calibration for pi
            s_id = scorer.score(D_id_pi_cal, drop_self_in=False, drop_self_wild=False)
            pi_hat = estimate_pi_from_id_floor(s_id, floor_q=1.0)
            pi_hats.append(pi_hat)

        pi_hats = np.array(pi_hats)
        results[float(pi_true)] = {
            "pi_hat_mean": float(pi_hats.mean()),
            "pi_hat_std": float(pi_hats.std()),
            "error_mean": float(np.mean(np.abs(pi_hats - pi_true))),
            "error_std": float(np.std(np.abs(pi_hats - pi_true))),
            "pi_hats": pi_hats.tolist(),
        }

        print(f"  π_true={pi_true:.2f}: π̂={pi_hats.mean():.3f} ± {pi_hats.std():.3f}, "
              f"|error|={np.mean(np.abs(pi_hats - pi_true)):.3f}")
    return results


# =============================================================================
# Experiment (ii): Deployment Risk / Precision
# =============================================================================

def experiment_deployment_risk(
    D_in_train, D_in_test, D_out,
    pi_values=(0.1, 0.2, 0.3, 0.4, 0.5),
    alpha_values=(0.01, 0.05, 0.10),
    k=50, n_trials=5, seed=42,
    metric="euclidean"
):
    """
    Show that AUROC alone is insufficient for deployment decisions.
    Estimate π̂ and compute calibrated vs uncalibrated precision.

    Improved version with proper train/val splits to avoid leakage.

    For each (π, α):
    - Construct wild data with known π
    - Score with k-NN DRE
    - Set threshold at FPR=α on ID data
    - Compare: (a) true precision, (b) precision estimated with π̂,
               (c) precision assuming π=0.5 (uncalibrated)

    Returns
    -------
    results : dict — nested {pi: {alpha: {auroc, precision_true, precision_calibrated, ...}}}
    """
    rng = np.random.RandomState(seed)
    results = {}
    n_wild = min(len(D_in_test), len(D_out), 2000)

    # Pure ID calibration set for threshold and pi
    n_cal = min(1000, len(D_in_test))
    idx_cal = rng.choice(len(D_in_test), size=n_cal, replace=False)
    D_id_cal = D_in_test[idx_cal]

    for pi_true in pi_values:
        results[float(pi_true)] = {}

        for alpha in alpha_values:
            trial_results = defaultdict(list)

            for _ in range(n_trials):
                n_ood = int(pi_true * n_wild)
                n_id = n_wild - n_ood

                idx_id = rng.choice(len(D_in_test), size=n_id, replace=False)
                idx_ood = rng.choice(len(D_out), size=n_ood, replace=False)

                D_wild = np.vstack([D_in_test[idx_id], D_out[idx_ood]])
                y_wild = np.concatenate([np.zeros(n_id), np.ones(n_ood)])

                perm = rng.permutation(len(D_wild))
                D_wild = D_wild[perm]
                y_wild = y_wild[perm]

                # split wild into ref/query
                split = len(D_wild) // 2
                D_wild_ref = D_wild[:split]
                D_wild_query = D_wild[split:]
                y_query = y_wild[split:]

                scorer = KNNDensityRatioScorer(D_in=D_in_train, D_wild_ref=D_wild_ref, k=k, metric=metric)

                # Score query wild
                s_query = scorer.score(D_wild_query, drop_self_in=False, drop_self_wild=False)

                # AUROC (rank-only)
                auroc = roc_auc_score(y_query, s_query) if len(np.unique(y_query)) > 1 else 0.5

                # Score pure ID calibration (for threshold + pi)
                s_id = scorer.score(D_id_cal, drop_self_in=False, drop_self_wild=False)

                # Threshold for target FPR alpha on pure ID
                tau = np.quantile(s_id, 1 - alpha, method="higher")

                flagged = s_query > tau
                n_flagged = int(flagged.sum())

                if n_flagged > 0:
                    precision_true = float(np.mean(y_query[flagged]))
                    fpr_actual = float(np.mean((s_query[y_query == 0] > tau))) if np.any(y_query == 0) else 0.0
                    tpr = float(np.mean((s_query[y_query == 1] > tau))) if np.any(y_query == 1) else 0.0

                    # π̂ from deep-ID floor (theory-aligned)
                    pi_hat = estimate_pi_from_id_floor(s_id, floor_q=1.0)

                    # Precision calibration formula (Theorem 3)
                    precision_cal = (pi_hat * tpr) / max(pi_hat * tpr + (1 - pi_hat) * alpha, 1e-12)
                    precision_naive = (0.5 * tpr) / max(0.5 * tpr + 0.5 * alpha, 1e-12)
                    precision_oracle = (pi_true * tpr) / max(pi_true * tpr + (1 - pi_true) * alpha, 1e-12)
                else:
                    precision_true = precision_cal = precision_naive = precision_oracle = 0.0
                    fpr_actual = tpr = 0.0
                    pi_hat = 0.0

                trial_results["auroc"].append(auroc)
                trial_results["precision_true"].append(precision_true)
                trial_results["precision_calibrated"].append(float(precision_cal))
                trial_results["precision_naive"].append(float(precision_naive))
                trial_results["precision_oracle"].append(float(precision_oracle))
                trial_results["fpr_actual"].append(fpr_actual)
                trial_results["tpr"].append(tpr)
                trial_results["pi_hat"].append(float(pi_hat))
                trial_results["n_flagged"].append(float(n_flagged))

            results[float(pi_true)][float(alpha)] = {
                k_: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k_, v in trial_results.items()
            }

            r = results[float(pi_true)][float(alpha)]
            print(f"  π={pi_true:.2f}, α={alpha:.2f}: "
                  f"AUROC={r['auroc']['mean']:.3f}, "
                  f"Prec_true={r['precision_true']['mean']:.3f}, "
                  f"Prec_cal={r['precision_calibrated']['mean']:.3f}, "
                  f"Prec_naive={r['precision_naive']['mean']:.3f}")

    return results


# =============================================================================
# Main runner
# =============================================================================

def get_dataset_pairs():
    """Define ID → OOD dataset pairs and file mappings."""
    return [
        {
            'name': 'CIFAR-10 → SVHN',
            'id_file': 'cifar10_test_{arch}.npz',
            'ood_file': 'svhn_test_{arch}.npz',
        },
        {
            'name': 'CIFAR-10 → CIFAR-100',
            'id_file': 'cifar10_test_{arch}.npz',
            'ood_file': 'cifar100_test_{arch}.npz',
        },
        {
            'name': 'CIFAR-10 → Textures',
            'id_file': 'cifar10_test_{arch}.npz',
            'ood_file': 'textures_test_{arch}.npz',
        },
        {
            'name': 'CIFAR-100 → SVHN',
            'id_file': 'cifar100_test_{arch}.npz',
            'ood_file': 'svhn_test_{arch}.npz',
        },
    ]


def run_all(features_dir, output_dir, k=50, n_trials=5):
    """Run all experiments across architectures and dataset pairs."""

    os.makedirs(output_dir, exist_ok=True)

    architectures = ['wrn', 'clip', 'dinov2']
    pairs = get_dataset_pairs()
    pi_values = (0.1, 0.2, 0.3, 0.4, 0.5)
    alpha_values = (0.01, 0.05, 0.10)

    all_results = {}

    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"Architecture: {arch.upper()}")
        print(f"{'='*60}")

        for pair in pairs:
            pair_name = pair['name']
            id_path = os.path.join(features_dir, pair['id_file'].format(arch=arch))
            ood_path = os.path.join(features_dir, pair['ood_file'].format(arch=arch))

            if not os.path.exists(id_path) or not os.path.exists(ood_path):
                print(f"  Skipping {pair_name} ({arch}): files not found")
                print(f"    Looked for: {id_path}")
                print(f"    Looked for: {ood_path}")
                continue

            print(f"\n--- {pair_name} ({arch}) ---")

            # Load features
            D_id = load_features(id_path)
            D_ood = load_features(ood_path)

            print(f"  ID features: {D_id.shape}, OOD features: {D_ood.shape}")

            # L2 normalize features
            D_id = l2_normalize(D_id.astype(np.float32))
            D_ood = l2_normalize(D_ood.astype(np.float32))

            # Split ID into train / held-out / test
            n = len(D_id)
            idx = np.random.RandomState(42).permutation(n)
            n_train = n // 3
            n_held = n // 3

            D_in_train = D_id[idx[:n_train]]
            D_in_held = D_id[idx[n_train:n_train+n_held]]
            D_in_test = D_id[idx[n_train+n_held:]]

            # Compute separation ratio γ
            gamma, delta, epsilon = compute_separation_ratio(D_in_train, D_ood, k=k)
            print(f"  γ = {gamma:.3f} (δ={delta:.4f}, ε={epsilon:.4f})")

            key = f"{arch}_{pair_name}"
            all_results[key] = {
                'arch': arch,
                'pair': pair_name,
                'gamma': float(gamma),
                'delta': float(delta),
                'epsilon': float(epsilon),
            }

            # Experiment (i): Provable mixtures
            print(f"\n  Experiment (i): Provable Mixtures")
            mix_results = experiment_provable_mixtures(
                D_in_train, D_in_held, D_ood,
                pi_values=pi_values, k=k, n_trials=n_trials,
                metric="euclidean"
            )
            all_results[key]['mixture_estimation'] = mix_results

            # Experiment (ii): Deployment risk
            print(f"\n  Experiment (ii): Deployment Risk")
            risk_results = experiment_deployment_risk(
                D_in_train, D_in_test, D_ood,
                pi_values=pi_values, alpha_values=alpha_values,
                k=k, n_trials=n_trials,
                metric="euclidean"
            )
            all_results[key]['deployment_risk'] = risk_results

    # Save results
    results_path = os.path.join(output_dir, 'dre_results.json')

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {results_path}")
    print(f"{'='*60}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UAI DRE Experiments')
    parser.add_argument('--features_dir', type=str, default='./features',
                        help='Directory containing .npz feature files')
    parser.add_argument('--output_dir', type=str, default='./results/dre',
                        help='Output directory for results')
    parser.add_argument('--k', type=int, default=50,
                        help='k for k-NN (default: 50)')
    parser.add_argument('--n_trials', type=int, default=5,
                        help='Number of random trials per setting')

    args = parser.parse_args()
    run_all(args.features_dir, args.output_dir, k=args.k, n_trials=args.n_trials)
