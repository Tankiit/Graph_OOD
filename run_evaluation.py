"""
run_evaluation.py

Full evaluation pipeline for the paper.
Works directly from cached .npz feature files — no network forward pass needed.

Standard metrics via OpenOOD's compute_all_metrics (AUROC, AUPR, FPR@95).
DRE calibration metrics (pi_hat, prec_oracle, prec_hat, prec_naive).

Feature file structure expected:
    {feature_root}/{arch}/{dataset}_train.npz   -> keys: 'features', (optional) 'labels'
    {feature_root}/{arch}/{dataset}_test.npz    -> keys: 'features', (optional) 'labels'

Usage:
    python run_evaluation.py --feature_root /path/to/features --output results/

Architectures:  clip_vitg14 | dinov2_vitg14
ID datasets:    cifar10 | cifar100
OOD datasets:   svhn_test | dtd_test | imagenet_resize | imagenet_1k_val | lsun_resize |
                places365_val | tinyimagenet_val
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from itertools import product
from sklearn.neighbors import NearestNeighbors

# ── OpenOOD metrics (pip install openood) ──────────────────────────────────
try:
    from openood.evaluators.metrics import compute_all_metrics
    OPENOOD_AVAILABLE = True
except ImportError:
    OPENOOD_AVAILABLE = False
    print("WARNING: openood not installed. Falling back to sklearn metrics.")
    print("         pip install git+https://github.com/Jingkang50/OpenOOD.git")

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


# =============================================================================
# 1. LOADERS
# =============================================================================

def load_features(path: str) -> np.ndarray:
    """Load features from .npz, normalise to unit sphere (cosine similarity)."""
    data = np.load(path)
    feats = data['feats'] if 'feats' in data else data['features']
    feats = feats.astype(np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.clip(norms, 1e-8, None)


def load_split(feature_root, arch, filename):
    """
    Load features by exact filename stem.
    e.g. load_split(root, 'clip_vitg14', 'cifar10_train')
         load_split(root, 'clip_vitg14', 'places365_val')
    """
    path = Path(feature_root) / arch / f"{filename}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return load_features(str(path))


# =============================================================================
# 2. OOD SCORERS  (operate on normalised features)
# =============================================================================

def score_knn(feats_train, feats_test, k=10):
    """k-NN distance score. Higher = more OOD."""
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    nn.fit(feats_train)
    dists, _ = nn.kneighbors(feats_test, n_neighbors=k)
    return dists[:, -1]   # k-th NN distance


def score_mahalanobis(feats_train, feats_test):
    """Class-agnostic Mahalanobis (single Gaussian). Higher = more OOD."""
    mu  = feats_train.mean(axis=0)
    cov = np.cov(feats_train.T) + 1e-6 * np.eye(feats_train.shape[1])
    cov_inv = np.linalg.pinv(cov)
    delta = feats_test - mu
    scores = np.einsum('ij,jk,ik->i', delta, cov_inv, delta)
    return scores


def score_vim(feats_train, feats_test, d=None):
    """
    Virtual-logit Matching (ViM). Uses PCA residual energy.
    d defaults to feature_dim // 2, capped at 512.
    """
    dim = feats_train.shape[1]
    if d is None:
        d = min(dim // 2, 512)
    mu = feats_train.mean(0)
    _, _, Vt = np.linalg.svd(feats_train - mu, full_matrices=False)
    P     = Vt[:d]                          # (d, dim)
    proj  = (feats_test - mu) @ P.T @ P    # projection onto ID subspace
    resid = (feats_test - mu) - proj        # out-of-subspace component
    return np.linalg.norm(resid, axis=1)    # higher = more OOD


def score_energy(feats_train, feats_test, T=1.0):
    """
    Cosine-similarity energy score against train set.
    s(z) = -T * log sum_i exp(z · z_i / T)
    Negated so higher = more OOD.
    """
    sim = feats_test @ feats_train.T          # (n_test, n_train) cosine sims
    energy = -T * torch.logsumexp(
        torch.tensor(sim / T), dim=1).numpy()
    return energy


SCORERS = {
    'knn':         score_knn,
    'mahalanobis': score_mahalanobis,
    'vim':         score_vim,
    'energy':      score_energy,
}


# =============================================================================
# 3. STANDARD METRICS  (via OpenOOD or sklearn fallback)
# =============================================================================

def compute_metrics(scores_id, scores_ood):
    """
    Returns dict with AUROC, AUPR_IN, AUPR_OUT, FPR@95.
    scores: higher value = more OOD.
    """
    labels = np.concatenate([
        np.zeros(len(scores_id)),   # 0 = ID
        np.ones(len(scores_ood))    # 1 = OOD
    ])
    scores = np.concatenate([scores_id, scores_ood])

    if OPENOOD_AVAILABLE:
        # compute_all_metrics expects: conf scores (higher=ID), labels (1=OOD)
        # so we negate: conf = -score
        conf = -scores
        results = compute_all_metrics(conf, labels, verbose=False)
        # OpenOOD returns: [FPR, AUROC, AUPR_IN, AUPR_OUT, OSCR, ACC]
        fpr, auroc, aupr_in, aupr_out = results[:4]
        return {
            'auroc':    float(auroc),
            'aupr_in':  float(aupr_in),
            'aupr_out': float(aupr_out),
            'fpr95':    float(fpr),
        }
    else:
        # sklearn fallback
        auroc    = roc_auc_score(labels, scores)
        aupr_out = average_precision_score(labels, scores)
        aupr_in  = average_precision_score(1 - labels, -scores)
        fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
        idx  = np.searchsorted(tpr_arr, 0.95)
        fpr95 = float(fpr_arr[min(idx, len(fpr_arr)-1)])
        return {
            'auroc':    float(auroc),
            'aupr_in':  float(aupr_in),
            'aupr_out': float(aupr_out),
            'fpr95':    fpr95,
        }


# =============================================================================
# 4. DRE CALIBRATION METRICS
# =============================================================================

def estimate_pi_floor(feats_cal, feats_wild, k=10, batch_size=512):
    """
    Vectorised floor estimator.
    pi_hat = 1 - median_z r_hat_wild(z)

    r_hat_wild(z) = [count of wild points within rho_wild(z)] / n_wild
                  / [count of cal  points within rho_cal(z) ] * n_cal

    where rho_*(z) = distance to k-th nearest neighbour in that set.

    Vectorised: query all cal points in batches against pre-built index.
    Complexity: O(n_cal * n_wild / batch) vs O(n_cal^2) for the naive loop.
    """
    n_cal  = len(feats_cal)
    n_wild = len(feats_wild)

    nn_cal  = NearestNeighbors(n_neighbors=k, metric='cosine',
                               algorithm='auto', n_jobs=-1).fit(feats_cal)
    nn_wild = NearestNeighbors(n_neighbors=k, metric='cosine',
                               algorithm='auto', n_jobs=-1).fit(feats_wild)

    r_hat = np.empty(n_cal, dtype=np.float32)

    for start in range(0, n_cal, batch_size):
        end   = min(start + batch_size, n_cal)
        batch = feats_cal[start:end]              # (B, d)

        # k-th NN distances define the ball radius for each query point
        d_cal,  _ = nn_cal.kneighbors( batch, n_neighbors=k)  # (B, k)
        d_wild, _ = nn_wild.kneighbors(batch, n_neighbors=k)  # (B, k)
        rho_cal  = d_cal[:,  -1]   # (B,) radius in cal set
        rho_wild = d_wild[:, -1]   # (B,) radius in wild set

        # Count points within each ball:
        # Use the pairwise distance matrix within each batch vs full set.
        # For cosine: sim = batch @ set.T, distance = 1 - sim
        sim_cal  = 1.0 - batch @ feats_cal.T    # (B, n_cal)  cosine dist
        sim_wild = 1.0 - batch @ feats_wild.T   # (B, n_wild)

        t_cal  = (sim_cal  <= rho_cal[:,  None]).sum(axis=1).astype(np.float32)
        t_wild = (sim_wild <= rho_wild[:, None]).sum(axis=1).astype(np.float32)

        r_hat[start:end] = (t_wild / n_wild) / np.maximum(t_cal / n_cal, 1e-9)

    return float(np.clip(1.0 - np.median(r_hat), 0.0, 1.0))


def precision_from_components(tpr, fpr, pi):
    """Prec(tau, pi) = pi*TPR / (pi*TPR + (1-pi)*FPR)  [Proposition 1]."""
    num = pi * tpr
    den = pi * tpr + (1.0 - pi) * fpr
    return float(num / max(den, 1e-9))


def compute_dre_metrics(feats_id_cal, feats_id_test, feats_ood,
                        scores_id_cal, scores_id_test, scores_ood,
                        pi_values=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
                        k=10, fpr_target=0.05):
    """
    For each pi: build wild mixture, estimate pi_hat, compute precision metrics.

    Returns list of dicts (one per pi value).
    """
    # Threshold calibrated once on ID cal set (pi-independent)
    tau = float(np.quantile(scores_id_cal, 1.0 - fpr_target))

    rows = []
    n_ood_total = len(feats_ood)

    for pi in pi_values:
        rng = np.random.default_rng(42)
        n   = len(feats_id_test)
        n_ood_requested = int(n * pi)
        n_safe = n - n_ood_requested

        # Cap n_ood to available samples and skip if not enough
        n_ood = min(n_ood_requested, n_ood_total)
        n_safe_actual = n - n_ood

        # Skip if not enough OOD samples or no safe samples
        if n_ood == 0 or n_safe_actual == 0:
            rows.append({
                'pi_true':      pi,
                'pi_hat':       np.nan,
                'pi_error':     np.nan,
                'auroc':        np.nan,
                'fpr95':        np.nan,
                'aupr_in':      np.nan,
                'aupr_out':     np.nan,
                'tau':          tau,
                'tpr_at_tau':   np.nan,
                'fpr_at_tau':   np.nan,
                'prec_oracle':  np.nan,
                'prec_hat':     np.nan,
                'prec_naive':   np.nan,
                'gap':          np.nan,
            })
            continue

        idx_safe = rng.choice(n, n_safe_actual, replace=False)
        idx_ood  = rng.choice(n_ood_total, n_ood, replace=False)

        feats_wild  = np.concatenate([feats_id_test[idx_safe],  feats_ood[idx_ood]])
        scores_wild = np.concatenate([scores_id_test[idx_safe], scores_ood[idx_ood]])
        labels      = np.array([0]*n_safe_actual + [1]*n_ood)

        # --- standard metrics on the wild mixture ---
        std = compute_metrics(scores_id_cal, scores_wild)   # treat wild as "OOD set"

        # --- pi estimation ---
        pi_hat = estimate_pi_floor(feats_id_cal, feats_wild, k=k)

        # --- TPR / FPR at threshold tau ---
        pred_ood = (scores_wild > tau)
        tp = np.sum(pred_ood & (labels == 1))
        fp = np.sum(pred_ood & (labels == 0))
        fn = np.sum(~pred_ood & (labels == 1))
        tn = np.sum(~pred_ood & (labels == 0))
        tpr_val = float(tp / max(tp + fn, 1))
        fpr_val = float(fp / max(fp + tn, 1))

        rows.append({
            'pi_true':      pi,
            'pi_hat':       pi_hat,
            'pi_error':     abs(pi_hat - pi),
            # standard (pi-invariant)
            'auroc':        std['auroc'],
            'fpr95':        std['fpr95'],
            'aupr_in':      std['aupr_in'],
            'aupr_out':     std['aupr_out'],
            # DRE calibration
            'tau':          tau,
            'tpr_at_tau':   tpr_val,
            'fpr_at_tau':   fpr_val,
            'prec_oracle':  precision_from_components(tpr_val, fpr_val, pi),
            'prec_hat':     precision_from_components(tpr_val, fpr_val, pi_hat),
            'prec_naive':   precision_from_components(tpr_val, fpr_val, 0.5),
            'gap':          (precision_from_components(tpr_val, fpr_val, pi_hat)
                            - precision_from_components(tpr_val, fpr_val, 0.5)),
        })

    return rows


# =============================================================================
# 5. MAIN LOOP OVER ALL (arch, id_dataset, ood_dataset, scorer) COMBINATIONS
# =============================================================================

ARCHS       = ['clip_vitg14', 'dinov2_vitg14']
ID_DATASETS = {
    'cifar10':  {'train': 'cifar10_train',  'test': 'cifar10_test'},
    'cifar100': {'train': 'cifar100_train', 'test': 'cifar100_test'},
}
# exact filename stems (no extension) present in every arch folder
OOD_FILES = [
    'svhn_test',
    'dtd_test',
    'imagenet_resize',
    'imagenet_1k_val',
    'lsun_resize',
    'tinyimagenet_val',
    'places365_val',
]
PI_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


def run_all(feature_root, output_dir, scorers=None, k_knn=10, k_dre=10,
            cal_frac=0.2, fpr_target=0.05, pi_values=None):
    """
    Run full evaluation grid. Saves one JSON per (arch, id, ood, scorer).
    """
    if scorers is None:
        scorers = list(SCORERS.keys())
    if pi_values is None:
        pi_values = PI_VALUES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for arch, (id_name, id_files) in product(ARCHS, ID_DATASETS.items()):
        print(f"\n{'='*60}")
        print(f"  arch={arch}  id={id_name}")
        print(f"{'='*60}")

        try:
            feats_id_train = load_split(feature_root, arch, id_files['train'])
            feats_id_test  = load_split(feature_root, arch, id_files['test'])
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        # Cal / eval split (first cal_frac of test as S_cal)
        n_test = len(feats_id_test)
        n_cal  = int(n_test * cal_frac)
        feats_id_cal  = feats_id_test[:n_cal]
        feats_id_eval = feats_id_test[n_cal:]
        print(f"  ID: train={len(feats_id_train)}  cal={n_cal}  eval={n_test-n_cal}")

        for ood_file in OOD_FILES:
            try:
                feats_ood = load_split(feature_root, arch, ood_file)
            except FileNotFoundError as e:
                print(f"  SKIP {ood_file}: {e}")
                continue

            print(f"\n  OOD={ood_file}  n={len(feats_ood)}")

            for scorer_name in scorers:
                scorer_fn = SCORERS[scorer_name]
                print(f"    scorer={scorer_name} ...", end=' ', flush=True)

                try:
                    kw = {'k': k_knn} if scorer_name == 'knn' else {}
                    scores_id_cal  = scorer_fn(feats_id_train, feats_id_cal,  **kw)
                    scores_id_eval = scorer_fn(feats_id_train, feats_id_eval, **kw)
                    scores_ood     = scorer_fn(feats_id_train, feats_ood,     **kw)
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

                # Standard OOD metrics (no mixture, fixed test sets)
                std_metrics = compute_metrics(scores_id_eval, scores_ood)
                print(f"AUROC={std_metrics['auroc']:.3f}  FPR95={std_metrics['fpr95']:.3f}")

                # DRE calibration metrics (pi sweep)
                dre_rows = compute_dre_metrics(
                    feats_id_cal, feats_id_eval, feats_ood,
                    scores_id_cal, scores_id_eval, scores_ood,
                    pi_values=pi_values, k=k_dre, fpr_target=fpr_target,
                )

                result = {
                    'arch':        arch,
                    'id_dataset':  id_name,
                    'ood_dataset': ood_file,
                    'scorer':      scorer_name,
                    'std_metrics': std_metrics,
                    'dre_metrics': dre_rows,
                }
                all_results.append(result)

                fname = f"{arch}__{id_name}__{ood_file}__{scorer_name}.json"
                with open(output_dir / fname, 'w') as f:
                    json.dump(result, f, indent=2)

    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone. {len(all_results)} result files in {output_dir}/")
    return all_results


# =============================================================================
# 6. TABLE + FIGURE GENERATION
# =============================================================================

def make_standard_table(all_results, save_path=None):
    """
    LaTeX table: rows = (id, ood), cols = (arch x scorer), cells = AUROC.
    Structure matches paper Table 2.
    """
    from collections import defaultdict
    cells = defaultdict(dict)
    for r in all_results:
        row_key = (r['id_dataset'], r['ood_dataset'])
        col_key = (r['arch'], r['scorer'])
        cells[row_key][col_key] = r['std_metrics']['auroc']

    archs   = sorted({r['arch']   for r in all_results})
    scorers = sorted({r['scorer'] for r in all_results})
    rows    = sorted(cells.keys())

    header_cols = ' & '.join(
        f"\\multicolumn{{1}}{{c}}{{{a.split('_')[0]}\\\\{s}}}"
        for a in archs for s in scorers
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{AUROC across architectures and scorers. "
        r"All methods evaluated on identical feature sets from the same $k$-NN graph.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{ll" + "c"*len(archs)*len(scorers) + "}",
        r"\toprule",
        r"ID & OOD & " + header_cols + r" \\",
        r"\midrule",
    ]
    for (id_ds, ood_ds) in rows:
        vals = [f"{cells[(id_ds,ood_ds)].get((a,s), float('nan')):.3f}"
                for a in archs for s in scorers]
        lines.append(f"{id_ds} & {ood_ds} & " + " & ".join(vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)

    if save_path:
        Path(save_path).write_text(tex)
    return tex


def make_dre_figure(all_results, arch='clip_vitg14',
                    id_ds='cifar10', ood_ds='svhn_test',
                    scorer='knn', save_path=None):
    """Three-panel figure: AUROC flat, pi_hat accuracy, precision calibration."""
    import matplotlib.pyplot as plt

    record = next(
        (r for r in all_results
         if r['arch']==arch and r['id_dataset']==id_ds
         and r['ood_dataset']==ood_ds and r['scorer']==scorer),
        None
    )
    if record is None:
        print(f"No result for {arch}/{id_ds}/{ood_ds}/{scorer}")
        return

    rows = record['dre_metrics']
    pi_true    = [r['pi_true']    for r in rows]
    auroc      = [r['auroc']      for r in rows]
    pi_hat     = [r['pi_hat']     for r in rows]
    prec_ora   = [r['prec_oracle'] for r in rows]
    prec_hat   = [r['prec_hat']   for r in rows]
    prec_naive = [r['prec_naive'] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel A: AUROC vs pi  (should be flat — Theorem 1)
    ax = axes[0]
    ax.plot(pi_true, auroc, 'o-', color='steelblue', lw=2)
    ax.axhline(np.mean(auroc), color='gray', ls='--',
               label=f'Mean={np.mean(auroc):.3f}', alpha=0.7)
    ax.set(xlabel=r'True $\pi$', ylabel='AUROC',
           title=r'AUROC is $\pi$-invariant (Thm 1)', ylim=(0, 1.05))
    ax.legend(fontsize=9)

    # Panel B: pi_hat vs pi_true (Theorem 3)
    ax = axes[1]
    ax.plot([0, 0.5], [0, 0.5], 'k--', alpha=0.4, label='Identity')
    ax.plot(pi_true, pi_hat, 'o-', color='darkorange', lw=2, label=r'$\hat\pi$')
    ax.set(xlabel=r'True $\pi$', ylabel=r'Estimated $\hat\pi$',
           title=r'$\hat\pi$ consistency (Thm 3)')
    ax.legend(fontsize=9)

    # Panel C: Precision curves (main contribution)
    ax = axes[2]
    ax.plot(pi_true, prec_ora,   'o-',  color='green',      lw=2, label='Oracle $\\pi$')
    ax.plot(pi_true, prec_hat,   's--', color='darkorange',  lw=2, label=r'Estimated $\hat\pi$ (ours)')
    ax.plot(pi_true, prec_naive, '^:',  color='crimson',     lw=2, label=r'Naive $\pi{=}0.5$')
    ax.set(xlabel=r'True $\pi$', ylabel='Deployment Precision',
           title='Precision calibration (Thm 2+5)', ylim=(0, 1.05))
    ax.legend(fontsize=9)

    plt.suptitle(f'{arch} | {id_ds} → {ood_ds} | scorer: {scorer}', fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


# =============================================================================
# 7. CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_root', required=True,
                        help='Root dir containing clip_vitg14/ dinov2_vitg14/')
    parser.add_argument('--output',       default='results/',
                        help='Output directory for JSON + tables + figures')
    parser.add_argument('--scorers', nargs='+',
                        default=['knn', 'mahalanobis', 'vim', 'energy'],
                        choices=list(SCORERS.keys()))
    parser.add_argument('--k_knn',  type=int, default=10)
    parser.add_argument('--k_dre',  type=int, default=10)
    parser.add_argument('--pi_values', nargs='+', type=float,
                        default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument('--table_only', action='store_true',
                        help='Skip DRE metrics, only compute standard table')
    args = parser.parse_args()

    results = run_all(
        feature_root = args.feature_root,
        output_dir   = args.output,
        scorers      = args.scorers,
        k_knn        = args.k_knn,
        k_dre        = args.k_dre,
        pi_values    = args.pi_values,
    )

    # Generate outputs
    out = Path(args.output)
    tex = make_standard_table(results, save_path=str(out / 'table_main.tex'))
    print("\nStandard results table:")
    print(tex)

    # DRE figure for representative setting
    make_dre_figure(results, save_path=str(out / 'fig_dre_calibration.pdf'))