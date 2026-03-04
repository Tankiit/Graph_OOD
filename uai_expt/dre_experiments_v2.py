"""
UAI 2026 — Density Ratio Estimation under Wild Contamination
v2: Adds NNR (Nearest Neighbor Ratio) scorer grounded in Noshad et al. (2017)

Two DRE mechanisms:
  1. Distance-based k-NN DRE (ratio of k-th neighbor distances)
  2. NNR: Joint neighborhood, label-counting DRE (Noshad et al. 2017)
     - Theoretically grounded: consistent estimator of density ratio
     - MSE rate O(N^{-2γ/(γ+d)}) under Hölder smoothness
     - Directly implements the "joint neighborhood" construction

Usage:
    python dre_experiments_v2.py --features_dir ./features --output_dir ./results/dre_v2

Expects .npz files with keys like 'features', 'feat', 'embeddings', etc.
Naming: {dataset}_test_{arch}.npz  (e.g., cifar10_test_clip.npz)
"""

import numpy as np
import os
import argparse
import json
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Feature I/O
# =============================================================================

def load_features(path):
    """Load features from .npz file."""
    data = np.load(path, allow_pickle=True)
    for key in ['features', 'feat', 'embeddings', 'data', 'arr_0']:
        if key in data:
            return data[key]
    return data[data.files[0]]


def l2_normalize(X, eps=1e-12):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


# =============================================================================
# Scorer 1: Distance-based k-NN DRE (your existing approach)
# =============================================================================

class DistanceBasedDREScorer:
    """
    Estimates log(rho_k(x; D_in) / rho_k(x; D_wild)).
    Monotone in r_wild but requires separate density estimation from each pop.
    """
    def __init__(self, D_in, D_wild_ref, k=50, metric="euclidean"):
        self.k = k
        self.nn_in = NearestNeighbors(n_neighbors=k+1, metric=metric, algorithm="auto").fit(D_in)
        self.nn_wild = NearestNeighbors(n_neighbors=k+1, metric=metric, algorithm="auto").fit(D_wild_ref)

    def score(self, X):
        """Returns scores: higher = more OOD."""
        d_in, _ = self.nn_in.kneighbors(X, n_neighbors=self.k)
        d_wild, _ = self.nn_wild.kneighbors(X, n_neighbors=self.k)
        rho_in = np.maximum(d_in[:, -1], 1e-12)
        rho_wild = np.maximum(d_wild[:, -1], 1e-12)
        return np.log(rho_in) - np.log(rho_wild)  # higher => more OOD


# =============================================================================
# Scorer 2: NNR — Joint Neighborhood DRE (Noshad et al. 2017)
# =============================================================================

class NNRScorer:
    """
    Nearest Neighbor Ratio scorer based on Noshad et al. (2017).

    Given reference set X ~ p_in and wild set Y ~ p_wild, pools Z = X ∪ Y
    and builds k-NN in the joint set. For each query point y_i, counts:
        N_i = number of X-points among k-NN of y_i in Z
        M_i = number of Y-points among k-NN of y_i in Z

    The ratio N_i / (M_i + 1) converges to p_in(y_i) / p_wild(y_i) = 1/r_wild(y_i).

    So score = -log(N_i / (M_i + 1)) is monotone in r_wild (higher = more OOD).

    Theoretical backing:
        - Consistent under Hölder smoothness (Theorem 2.1 in Noshad et al.)
        - MSE rate O(N^{-2γ/(γ+d)}) for γ-Hölder densities
        - No boundary correction needed
        - Automatic bias cancellation at support boundaries

    Connection to V1/V2:
        - V1 (separation): ensures r_wild departs from 1, so N_i/(M_i+1) is informative
        - V2 (neighborhood preservation): ensures local neighborhoods are pure enough
          for the label ratio to reflect density contrast, not noise
    """

    def __init__(self, D_in, D_wild_ref, k=50, metric="euclidean"):
        """
        Parameters
        ----------
        D_in : array (n_in, m) — clean reference bank (in-distribution)
        D_wild_ref : array (n_wild, m) — wild reference data
        k : int — number of neighbors in joint graph
        metric : str — distance metric
        """
        self.k = k
        self.n_in = len(D_in)
        self.n_wild = len(D_wild_ref)
        self.eta = self.n_wild / self.n_in  # sample ratio M/N

        # Pool Z = D_in ∪ D_wild, track labels
        self.Z = np.vstack([D_in, D_wild_ref])
        # label: 0 = in-distribution, 1 = wild
        self.labels = np.concatenate([
            np.zeros(self.n_in, dtype=int),
            np.ones(self.n_wild, dtype=int)
        ])

        # Build k-NN on pooled set
        # Request k+1 to handle self-matches for points in Z
        self.nn_joint = NearestNeighbors(
            n_neighbors=k + 1, metric=metric, algorithm="auto"
        ).fit(self.Z)

    def _count_neighbors(self, X, drop_self=False):
        """
        For each row in X, find k neighbors in Z and count labels.

        Returns
        -------
        N_i : array — count of in-distribution neighbors
        M_i : array — count of wild neighbors
        """
        n_req = self.k + 1 if drop_self else self.k
        # Always query k+1 to be safe
        dists, indices = self.nn_joint.kneighbors(X, n_neighbors=self.k + 1)

        N_counts = np.zeros(len(X), dtype=int)
        M_counts = np.zeros(len(X), dtype=int)

        for i in range(len(X)):
            nbrs = indices[i]
            d = dists[i]

            if drop_self:
                # Remove self-match (distance ~ 0)
                mask = d > 1e-12
                nbrs = nbrs[mask][:self.k]
            else:
                nbrs = nbrs[:self.k]

            nbr_labels = self.labels[nbrs]
            N_counts[i] = np.sum(nbr_labels == 0)  # in-distribution
            M_counts[i] = np.sum(nbr_labels == 1)  # wild

        return N_counts, M_counts

    def score(self, X, drop_self=False):
        """
        Score query points. Higher = more OOD.

        NNR ratio: N_i / (M_i + 1) ∝ p_in(x) / p_wild(x) = 1/r_wild(x)
        So score = -log(N_i / (M_i + 1)) = log(r_wild) approximately.
        """
        N_i, M_i = self._count_neighbors(X, drop_self=drop_self)

        # Noshad ratio: N_i / (M_i + 1)
        # This estimates (1/η) · p_in(x)/p_wild(x) up to the sample ratio correction
        ratio = N_i.astype(float) / (M_i.astype(float) + 1.0)

        # Score = -log(ratio) so that higher = more OOD
        # (ratio is small when p_wild >> p_in, i.e., OOD regions)
        return -np.log(np.maximum(ratio, 1e-12))

    def raw_ratio(self, X, drop_self=False):
        """
        Return the raw NNR ratio N_i/(M_i+1) without log transform.
        Useful for π estimation where we need the linear-scale ratio.
        """
        N_i, M_i = self._count_neighbors(X, drop_self=drop_self)
        return N_i.astype(float) / (M_i.astype(float) + 1.0)

    def neighbor_fractions(self, X, drop_self=False):
        """
        Return fraction of in-distribution neighbors for each query point.
        This is the classification-based DRE view: P(neighbor is ID | x).
        """
        N_i, M_i = self._count_neighbors(X, drop_self=drop_self)
        total = N_i + M_i
        return N_i.astype(float) / np.maximum(total.astype(float), 1.0)


# =============================================================================
# π estimation
# =============================================================================

def estimate_pi_nnr(scorer_nnr, D_id_cal, floor_q=1.0):
    """
    Estimate π using NNR on pure ID calibration data.

    Theory: For z deep in M_in, r*(z) ≈ 0, so r_wild(z) ≈ (1-π).
    The NNR ratio N/(M+1) estimates (1/η)·(1/r_wild) = (1/η)·1/(1-π) at these points.

    Equivalently, the fraction of ID neighbors for a deep-ID point should be:
        p_in(z) / (p_in(z) + (η)·p_wild(z))
    For deep ID where p_wild(z) ≈ (1-π)·p_in(z):
        frac ≈ 1 / (1 + η·(1-π))

    Simpler approach: use the scores directly.
    For deep-ID points, score = -log(N/(M+1)) should be minimal.
    The minimum score on ID data corresponds to r_wild ≈ (1-π).
    """
    # Get raw ratios on pure ID calibration points
    # These points are NOT in the joint graph, so no self-match issue
    ratios = scorer_nnr.raw_ratio(D_id_cal, drop_self=False)

    # For deep-ID points: ratio ≈ (1/η) · 1/(1-π)
    # So (1-π) ≈ (1/η) · 1/max(ratio)
    # But this depends on η normalization...

    # More robust: use neighbor fractions
    fracs = scorer_nnr.neighbor_fractions(D_id_cal, drop_self=False)

    # For deep-ID: frac_in ≈ n_in / (n_in + n_wild · (1-π))
    # Under equal sample sizes (n_in ≈ n_wild): frac ≈ 1/(1 + (1-π)) = 1/(2-π)
    # So π ≈ 2 - 1/frac

    # Use high quantile of fractions (deepest ID points)
    high_frac = np.percentile(fracs, 100 - floor_q)  # e.g., 99th percentile

    n_in = scorer_nnr.n_in
    n_wild = scorer_nnr.n_wild

    # Exact formula: frac = n_in / (n_in + n_wild · r_wild(z))
    # For deep ID: r_wild(z) = (1-π), so:
    # frac = n_in / (n_in + n_wild · (1-π))
    # Solving: (1-π) = (n_in / n_wild) · (1/frac - 1)
    # π = 1 - (n_in / n_wild) · (1/frac - 1)

    if high_frac > 0.01:
        one_minus_pi = (n_in / n_wild) * (1.0 / high_frac - 1.0)
        pi_hat = 1.0 - one_minus_pi
    else:
        pi_hat = 0.99  # fractions near 0 => almost all wild = OOD

    return float(np.clip(pi_hat, 0.0, 0.99))


def estimate_pi_distance(scores_id, floor_q=1.0):
    """
    Estimate π from distance-based DRE scores on pure ID calibration data.
    (Your existing approach, kept for comparison.)
    """
    ratio_hat = np.exp(scores_id)
    lowq = np.percentile(ratio_hat, floor_q)
    pi_hat = 1.0 - lowq
    return float(np.clip(pi_hat, 0.0, 0.99))


# =============================================================================
# Geometric diagnostics
# =============================================================================

def compute_separation_ratio(D_in, D_out, k=50):
    """Compute γ = δ/(6ε). δ: inter-manifold, ε: intra-manifold spread."""
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(D_in)
    dists_intra, _ = nn.kneighbors(D_in)
    epsilon = np.mean(dists_intra[:, -1])

    dists_cross, _ = nn.kneighbors(D_out)
    delta = np.percentile(dists_cross[:, 0], 5)

    gamma = delta / (6 * epsilon + 1e-10)
    return gamma, delta, epsilon


def compute_cross_edge_fraction(D_in, D_wild, labels_wild, k=50):
    """
    V2 diagnostic: fraction of cross-distribution edges in joint k-NN graph.
    labels_wild: 0=ID, 1=OOD for points in D_wild.
    """
    Z = np.vstack([D_in, D_wild])
    z_labels = np.concatenate([np.zeros(len(D_in)), labels_wild])

    nn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(Z)
    _, indices = nn.kneighbors(Z)

    cross_edges = 0
    total_edges = 0
    for i in range(len(Z)):
        for j in indices[i]:
            if j != i:
                total_edges += 1
                if z_labels[i] != z_labels[j]:
                    cross_edges += 1

    return cross_edges / max(total_edges, 1)


# =============================================================================
# Experiment (i): Provable Mixtures — both scorers
# =============================================================================

def experiment_provable_mixtures(
    D_in_train, D_in_held, D_out,
    pi_values=(0.1, 0.2, 0.3, 0.4, 0.5),
    k=50, n_trials=5, seed=42
):
    """
    Construct wild data with known π, estimate π̂ with both scorers.
    """
    rng = np.random.RandomState(seed)
    results = {}
    n_wild = min(len(D_in_held), len(D_out), 2000)

    # Pure ID calibration (separate from mixture construction)
    n_cal = min(800, len(D_in_held) // 2)
    idx_cal = rng.choice(len(D_in_held), size=n_cal, replace=False)
    D_id_cal = D_in_held[idx_cal]
    # Mask these out from mixture construction
    mask_held = np.ones(len(D_in_held), dtype=bool)
    mask_held[idx_cal] = False
    D_in_held_mix = D_in_held[mask_held]

    for pi_true in pi_values:
        trial_data = defaultdict(list)

        for t in range(n_trials):
            n_ood = int(pi_true * n_wild)
            n_id = n_wild - n_ood

            n_id = min(n_id, len(D_in_held_mix))
            n_ood = min(n_ood, len(D_out))

            idx_id = rng.choice(len(D_in_held_mix), size=n_id, replace=False)
            idx_ood = rng.choice(len(D_out), size=n_ood, replace=False)

            D_wild = np.vstack([D_in_held_mix[idx_id], D_out[idx_ood]])
            labels_wild = np.concatenate([np.zeros(n_id), np.ones(n_ood)])

            perm = rng.permutation(len(D_wild))
            D_wild = D_wild[perm]
            labels_wild = labels_wild[perm]

            # Split wild into ref/query
            split = len(D_wild) // 2
            D_wild_ref = D_wild[:split]
            D_wild_query = D_wild[split:]
            y_query = labels_wild[split:]

            # --- Distance-based scorer ---
            dist_scorer = DistanceBasedDREScorer(D_in_train, D_wild_ref, k=k)
            s_dist_query = dist_scorer.score(D_wild_query)
            s_dist_cal = dist_scorer.score(D_id_cal)

            auroc_dist = roc_auc_score(y_query, s_dist_query) if len(np.unique(y_query)) > 1 else 0.5
            pi_hat_dist = estimate_pi_distance(s_dist_cal, floor_q=1.0)

            # --- NNR scorer ---
            nnr_scorer = NNRScorer(D_in_train, D_wild_ref, k=k)
            s_nnr_query = nnr_scorer.score(D_wild_query)
            pi_hat_nnr = estimate_pi_nnr(nnr_scorer, D_id_cal, floor_q=1.0)

            auroc_nnr = roc_auc_score(y_query, s_nnr_query) if len(np.unique(y_query)) > 1 else 0.5

            # V2 diagnostic on this mixture
            eta = compute_cross_edge_fraction(
                D_in_train[:500], D_wild_ref[:500],
                labels_wild[:split][:500], k=min(k, 20)
            )

            trial_data["auroc_dist"].append(auroc_dist)
            trial_data["auroc_nnr"].append(auroc_nnr)
            trial_data["pi_hat_dist"].append(pi_hat_dist)
            trial_data["pi_hat_nnr"].append(pi_hat_nnr)
            trial_data["cross_edge_frac"].append(eta)

        results[float(pi_true)] = {}
        for key, vals in trial_data.items():
            vals = np.array(vals)
            results[float(pi_true)][key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "values": vals.tolist()
            }

        r = results[float(pi_true)]
        print(f"  π={pi_true:.2f}: "
              f"AUROC(dist)={r['auroc_dist']['mean']:.3f} "
              f"AUROC(nnr)={r['auroc_nnr']['mean']:.3f} | "
              f"π̂(dist)={r['pi_hat_dist']['mean']:.3f}±{r['pi_hat_dist']['std']:.3f} "
              f"π̂(nnr)={r['pi_hat_nnr']['mean']:.3f}±{r['pi_hat_nnr']['std']:.3f} | "
              f"η={r['cross_edge_frac']['mean']:.3f}")

    return results


# =============================================================================
# Experiment (ii): Deployment Risk — both scorers
# =============================================================================

def experiment_deployment_risk(
    D_in_train, D_in_test, D_out,
    pi_values=(0.1, 0.2, 0.3, 0.4, 0.5),
    alpha_values=(0.01, 0.05, 0.10),
    k=50, n_trials=5, seed=42
):
    """
    AUROC invariance + calibrated precision depends on π̂.
    Runs both scorers side by side.
    """
    rng = np.random.RandomState(seed)
    results = {}
    n_wild = min(len(D_in_test), len(D_out), 2000)

    n_cal = min(1000, len(D_in_test) // 2)
    idx_cal = rng.choice(len(D_in_test), size=n_cal, replace=False)
    D_id_cal = D_in_test[idx_cal]

    for pi_true in pi_values:
        results[float(pi_true)] = {}

        for alpha in alpha_values:
            trial_data = defaultdict(list)

            for _ in range(n_trials):
                n_ood = int(pi_true * n_wild)
                n_id = n_wild - n_ood

                idx_id = rng.choice(len(D_in_test), size=n_id, replace=False)
                idx_ood = rng.choice(len(D_out), size=n_ood, replace=False)

                D_wild = np.vstack([D_in_test[idx_id], D_out[idx_ood]])
                y_wild = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
                perm = rng.permutation(len(D_wild))
                D_wild, y_wild = D_wild[perm], y_wild[perm]

                split = len(D_wild) // 2
                D_wild_ref, D_wild_query = D_wild[:split], D_wild[split:]
                y_query = y_wild[split:]

                for scorer_name, ScorerClass in [("dist", DistanceBasedDREScorer), ("nnr", NNRScorer)]:
                    scorer = ScorerClass(D_in_train, D_wild_ref, k=k)
                    s_query = scorer.score(D_wild_query)

                    auroc = roc_auc_score(y_query, s_query) if len(np.unique(y_query)) > 1 else 0.5

                    # Threshold from ID calibration
                    if scorer_name == "nnr":
                        s_cal = scorer.score(D_id_cal)
                        pi_hat = estimate_pi_nnr(scorer, D_id_cal, floor_q=1.0)
                    else:
                        s_cal = scorer.score(D_id_cal)
                        pi_hat = estimate_pi_distance(s_cal, floor_q=1.0)

                    tau = np.quantile(s_cal, 1 - alpha)
                    flagged = s_query > tau
                    n_flagged = int(flagged.sum())

                    if n_flagged > 0:
                        prec_true = float(np.mean(y_query[flagged]))
                        tpr = float(np.mean(s_query[y_query == 1] > tau)) if np.any(y_query == 1) else 0.0
                        fpr_actual = float(np.mean(s_query[y_query == 0] > tau)) if np.any(y_query == 0) else 0.0

                        prec_cal = (pi_hat * tpr) / max(pi_hat * tpr + (1 - pi_hat) * alpha, 1e-12)
                        prec_naive = (0.5 * tpr) / max(0.5 * tpr + 0.5 * alpha, 1e-12)
                        prec_oracle = (pi_true * tpr) / max(pi_true * tpr + (1 - pi_true) * alpha, 1e-12)
                    else:
                        prec_true = prec_cal = prec_naive = prec_oracle = 0.0
                        fpr_actual = tpr = pi_hat = 0.0

                    trial_data[f"auroc_{scorer_name}"].append(auroc)
                    trial_data[f"prec_true_{scorer_name}"].append(prec_true)
                    trial_data[f"prec_cal_{scorer_name}"].append(float(prec_cal))
                    trial_data[f"prec_naive_{scorer_name}"].append(float(prec_naive))
                    trial_data[f"prec_oracle_{scorer_name}"].append(float(prec_oracle))
                    trial_data[f"pi_hat_{scorer_name}"].append(float(pi_hat))
                    trial_data[f"tpr_{scorer_name}"].append(tpr)
                    trial_data[f"fpr_{scorer_name}"].append(fpr_actual)

            results[float(pi_true)][float(alpha)] = {
                k_: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k_, v in trial_data.items()
            }

            r = results[float(pi_true)][float(alpha)]
            print(f"  π={pi_true:.2f}, α={alpha:.2f}: "
                  f"AUROC(dist/nnr)={r['auroc_dist']['mean']:.3f}/{r['auroc_nnr']['mean']:.3f} | "
                  f"Prec_true={r['prec_true_dist']['mean']:.3f}/{r['prec_true_nnr']['mean']:.3f} | "
                  f"π̂={r['pi_hat_dist']['mean']:.3f}/{r['pi_hat_nnr']['mean']:.3f}")

    return results


# =============================================================================
# Experiment (iii): AUROC invariance across π (key theoretical claim)
# =============================================================================

def experiment_auroc_invariance(
    D_in_train, D_in_held, D_out,
    pi_values=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7),
    k=50, n_trials=10, seed=42
):
    """
    Key theoretical claim: AUROC is invariant to π under Huber contamination.
    r_wild = (1-π) + π·r* is a monotone transform of r* for fixed π,
    so the ranking (and thus AUROC) is preserved.

    This experiment verifies: AUROC should be ~constant across all π values.
    """
    rng = np.random.RandomState(seed)
    results = {}
    n_wild = min(len(D_in_held), len(D_out), 2000)

    for pi_true in pi_values:
        aurocs_dist, aurocs_nnr = [], []

        for _ in range(n_trials):
            n_ood = int(pi_true * n_wild)
            n_id = n_wild - n_ood
            n_id = min(n_id, len(D_in_held))
            n_ood = min(n_ood, len(D_out))

            idx_id = rng.choice(len(D_in_held), size=n_id, replace=False)
            idx_ood = rng.choice(len(D_out), size=n_ood, replace=False)

            D_wild = np.vstack([D_in_held[idx_id], D_out[idx_ood]])
            y = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
            perm = rng.permutation(len(D_wild))
            D_wild, y = D_wild[perm], y[perm]

            split = len(D_wild) // 2
            D_ref, D_q = D_wild[:split], D_wild[split:]
            y_q = y[split:]

            if len(np.unique(y_q)) < 2:
                continue

            # Distance scorer
            ds = DistanceBasedDREScorer(D_in_train, D_ref, k=k)
            aurocs_dist.append(roc_auc_score(y_q, ds.score(D_q)))

            # NNR scorer
            ns = NNRScorer(D_in_train, D_ref, k=k)
            aurocs_nnr.append(roc_auc_score(y_q, ns.score(D_q)))

        results[float(pi_true)] = {
            "auroc_dist": {"mean": float(np.mean(aurocs_dist)), "std": float(np.std(aurocs_dist))},
            "auroc_nnr": {"mean": float(np.mean(aurocs_nnr)), "std": float(np.std(aurocs_nnr))},
        }

        r = results[float(pi_true)]
        print(f"  π={pi_true:.2f}: AUROC(dist)={r['auroc_dist']['mean']:.3f}±{r['auroc_dist']['std']:.3f}  "
              f"AUROC(nnr)={r['auroc_nnr']['mean']:.3f}±{r['auroc_nnr']['std']:.3f}")

    return results


# =============================================================================
# Main
# =============================================================================

def get_dataset_pairs():
    return [
        {'name': 'CIFAR10→SVHN', 'id': 'cifar10', 'ood': 'svhn'},
        {'name': 'CIFAR10→CIFAR100', 'id': 'cifar10', 'ood': 'cifar100'},
        {'name': 'CIFAR10→Textures', 'id': 'cifar10', 'ood': 'textures'},
        {'name': 'CIFAR100→SVHN', 'id': 'cifar100', 'ood': 'svhn'},
    ]


def find_feature_file(features_dir, dataset, arch):
    """Try common naming patterns to find feature files."""
    patterns = [
        f"{dataset}_test_{arch}.npz",
        f"{dataset}_{arch}.npz",
        f"{arch}_{dataset}.npz",
        f"{dataset}_test.npz",
    ]
    for p in patterns:
        path = os.path.join(features_dir, p)
        if os.path.exists(path):
            return path

    # Try to find any file matching dataset and arch
    for f in os.listdir(features_dir):
        if dataset in f.lower() and arch in f.lower() and f.endswith('.npz'):
            return os.path.join(features_dir, f)
    return None


def run_all(features_dir, output_dir, k=50, n_trials=5):
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect architectures and files
    print(f"Scanning {features_dir} for feature files...")
    available_files = [f for f in os.listdir(features_dir) if f.endswith('.npz')]
    print(f"Found {len(available_files)} .npz files: {available_files[:10]}...")

    architectures = ['wrn', 'clip', 'dinov2']
    pairs = get_dataset_pairs()
    pi_values = (0.1, 0.2, 0.3, 0.4, 0.5)
    alpha_values = (0.01, 0.05, 0.10)

    all_results = {}

    for arch in architectures:
        print(f"\n{'='*70}")
        print(f"Architecture: {arch.upper()}")
        print(f"{'='*70}")

        for pair in pairs:
            pair_name = pair['name']

            id_path = find_feature_file(features_dir, pair['id'], arch)
            ood_path = find_feature_file(features_dir, pair['ood'], arch)

            if id_path is None or ood_path is None:
                print(f"  Skipping {pair_name} ({arch}): files not found")
                continue

            print(f"\n--- {pair_name} ({arch}) ---")
            print(f"  ID: {id_path}")
            print(f"  OOD: {ood_path}")

            D_id = l2_normalize(load_features(id_path).astype(np.float32))
            D_ood = l2_normalize(load_features(ood_path).astype(np.float32))
            print(f"  Shapes: ID={D_id.shape}, OOD={D_ood.shape}")

            # Split ID: train / held / test
            n = len(D_id)
            idx = np.random.RandomState(42).permutation(n)
            s1, s2 = n // 3, 2 * n // 3
            D_in_train = D_id[idx[:s1]]
            D_in_held = D_id[idx[s1:s2]]
            D_in_test = D_id[idx[s2:]]

            # Geometric diagnostics
            gamma, delta, epsilon = compute_separation_ratio(D_in_train, D_ood, k=k)
            print(f"  γ={gamma:.4f} (δ={delta:.5f}, ε={epsilon:.5f})")

            key = f"{arch}_{pair_name}"
            all_results[key] = {
                'arch': arch, 'pair': pair_name,
                'gamma': float(gamma), 'delta': float(delta), 'epsilon': float(epsilon),
            }

            # Exp (iii): AUROC invariance (fast, important theoretical claim)
            print(f"\n  [Exp iii] AUROC Invariance across π:")
            inv_results = experiment_auroc_invariance(
                D_in_train, D_in_held, D_ood,
                pi_values=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7),
                k=k, n_trials=n_trials, seed=42
            )
            all_results[key]['auroc_invariance'] = inv_results

            # Exp (i): Provable mixtures
            print(f"\n  [Exp i] Provable Mixtures (π estimation):")
            mix_results = experiment_provable_mixtures(
                D_in_train, D_in_held, D_ood,
                pi_values=pi_values, k=k, n_trials=n_trials
            )
            all_results[key]['mixture_estimation'] = mix_results

            # Exp (ii): Deployment risk
            print(f"\n  [Exp ii] Deployment Risk (precision calibration):")
            risk_results = experiment_deployment_risk(
                D_in_train, D_in_test, D_ood,
                pi_values=pi_values, alpha_values=alpha_values,
                k=k, n_trials=n_trials
            )
            all_results[key]['deployment_risk'] = risk_results

    # Save
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        return obj

    out_path = os.path.join(output_dir, 'dre_v2_results.json')
    with open(out_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\n{'='*70}")
    print(f"All results saved to {out_path}")
    print(f"{'='*70}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: γ vs AUROC across architectures")
    print(f"{'='*70}")
    print(f"{'Setting':<35} {'γ':>8} {'AUROC(dist)':>12} {'AUROC(nnr)':>12}")
    print("-" * 70)
    for key, res in all_results.items():
        if 'auroc_invariance' in res:
            # Use π=0.3 as representative
            inv = res['auroc_invariance']
            pi_key = '0.3' if '0.3' in inv else list(inv.keys())[len(inv)//2]
            a_d = inv[pi_key]['auroc_dist']['mean']
            a_n = inv[pi_key]['auroc_nnr']['mean']
            print(f"{key:<35} {res['gamma']:>8.4f} {a_d:>12.3f} {a_n:>12.3f}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UAI DRE v2 — Distance + NNR')
    parser.add_argument('--features_dir', type=str, default='./features')
    parser.add_argument('--output_dir', type=str, default='./results/dre_v2')
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--n_trials', type=int, default=5)
    args = parser.parse_args()
    run_all(args.features_dir, args.output_dir, k=args.k, n_trials=args.n_trials)
