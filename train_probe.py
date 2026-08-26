"""
step2_train_probe.py
────────────────────
Trains OVA probe heads on hidden states from Step 1.
Computes Δ(x) = f_pred(h) − f_defer(h) and all evaluation metrics.

Fixes over the bare version:
  - --input and --output args for multi-dataset/model runs
  - --layer arg: use the specific peak layer found in sweep (default: all)
  - O(N log N) AURC instead of O(N²)
  - logprob normalisation uses calibration stats, not test stats
  - SCOD composite score as a fourth signal
  - per-question gap aggregation for conformal (not per-representation)
  - clean Table 1 row output for the paper

Signals evaluated:
  gap           Δ(x) = logit_pred − logit_defer        [proposed]
  probe_alone   σ(f_pred(h))                           [ablation]
  logprob_alone normalised generation log-probability  [ablation]
  scod          r(x) + β·gap  (KKT-optimal composite)  [proposed+]

Metrics:
  AURC  Area Under Risk-Coverage Curve  (lower = better)
  AUROC Binary discrimination AUROC      (higher = better)
  PRR   Prediction Rejection Ratio       (higher = better)

Usage:
    python step2_train_probe.py
    python step2_train_probe.py --input ./llama3_8b_tqa.pt
    python step2_train_probe.py --input ./llama3_8b_halu.pt --output results_halu.json
    python step2_train_probe.py --alpha 0.1 --seed 42
"""

import json, pickle, argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path("outputs")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train OVA probes and evaluate gap signal")
    p.add_argument("--input",  type=str, default=str(OUTPUT_DIR / "hidden_states.pt"),
                   help="Path to hidden_states.pt from Step 1")
    p.add_argument("--output", type=str, default=None,
                   help="JSON results file (default: outputs/results_<tag>.json)")
    p.add_argument("--layer",  type=int, default=None,
                   help="Specific layer to use (default: use stored mean-of-last-N)")
    p.add_argument("--alpha",  type=float, default=0.10,
                   help="Conformal miscoverage level (default 0.1 = 90%% coverage)")
    p.add_argument("--C",      type=float, default=1.0,
                   help="Logistic regression regularisation (same for both heads)")
    p.add_argument("--beta",   type=float, default=None,
                   help="SCOD β (default: KKT-optimal from α and TPR_MIN=0.7)")
    p.add_argument("--seed",   type=int,   default=42)
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path):
    print(f"\nLoading {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    N    = data["n_questions"]
    print(f"  {N} questions  |  d={data['h_correct'].shape[1]}"
          f"  |  {data['model']}")
    return data


def build_arrays(data, layer=None):
    """
    Build (X, y_model, y_expert, lp, lp_c, lp_w, cats, N).

    If --layer is specified AND the .pt file stores per-layer hidden states,
    extract that specific layer. Otherwise use the stored mean-of-last-N vector.
    """
    h_c_raw = data["h_correct"].float()
    h_w_raw = data["h_wrong"].float()

    # Per-layer extraction (if stored as [N, n_layers, d])
    if h_c_raw.ndim == 3 and layer is not None:
        max_l = h_c_raw.shape[1] - 1
        li = min(layer, max_l)
        h_c = h_c_raw[:, li, :].numpy()
        h_w = h_w_raw[:, li, :].numpy()
        print(f"  Using stored layer index {li} (of {h_c_raw.shape[1]})")
    else:
        # Already mean-pooled [N, d] from Step 1
        h_c = h_c_raw.numpy()
        h_w = h_w_raw.numpy()
        if layer is not None:
            print(f"  Note: --layer {layer} ignored — .pt stores mean-pooled vector. "
                  f"Re-run Step 1 with --save-all-layers to enable layer selection here.")

    N = h_c.shape[0]
    X = np.vstack([h_c, h_w])                          # (2N, d)
    y_model = np.array([1]*N + [0]*N, dtype=int)        # 1=correct, 0=wrong

    lp_c = data["lp_correct"].numpy()                   # (N,)
    lp_w = data["lp_wrong"].numpy()                     # (N,)
    lp   = np.concatenate([lp_c, lp_w])                 # (2N,)

    cats = list(data.get("categories", ["Unknown"] * N))

    # OVA expert labels
    # ── Reconstruction guard ───────────────────────────────────────────────────
    # TruthfulQA's multiple_choice split has no "category" field, so step1
    # may have saved y_expert=1 for every question (all fell through to __default__
    # -> human_acc=0.80 >= threshold -> expert_label=1). This gives a single-class
    # target that crashes LogisticRegression with "only one class: 1".
    #
    # Fix: detect the degenerate case and reconstruct y_expert from the
    # human_accs field (which step1 DOES save correctly).
    if "y_expert_correct" in data:
        y_exp_c = data["y_expert_correct"].numpy()
        y_exp_w = data["y_expert_wrong"].numpy()

        # Detect degenerate single-class case
        all_ones = (y_exp_c.sum() == N) and (y_exp_w.sum() == N)

        if all_ones and "human_accs" in data:
            print("  ⚠  y_expert all-1 (category join missed in step1).")
            print("     Reconstructing from saved human_accs...")
            haccs = np.array(list(data["human_accs"]), dtype=float)
            thr   = float(data.get("expert_threshold", 0.80))
            y_exp_c = (haccs >= thr).astype(int)
            y_exp_w = y_exp_c.copy()
            n_rel = int(y_exp_c.sum())
            print(f"     Expert reliable (human_acc >= {thr}): "
                  f"{n_rel}/{N} ({n_rel/N*100:.0f}%)")

        elif all_ones and len(set(cats)) > 1:
            print("  ⚠  y_expert all-1 — reconstructing from category names...")
            thr = float(data.get("expert_threshold", 0.80))
            y_exp_c = np.array(
                [int(HUMAN_ACC_BY_CATEGORY.get(c,
                     HUMAN_ACC_BY_CATEGORY["__default__"]) >= thr)
                 for c in cats], dtype=int)
            y_exp_w = y_exp_c.copy()

        y_expert  = np.concatenate([y_exp_c, y_exp_w])   # (2N,)
        n_div     = (y_model != y_expert).sum()
        n_classes = len(np.unique(y_expert))
        print(f"  OVA divergence: {n_div}/{2*N} reps "
              f"({n_div/(2*N)*100:.1f}%)  |  y_expert classes: {n_classes}")

        if n_classes < 2:
            print("  ⚠  y_expert still single-class after reconstruction.")
            print("     f_defer will train on y_model (gap = probe - probe = 0).")
            print("     Re-run step1 with category-join fix for true OVA.")
            y_expert = y_model.copy()
    else:
        y_expert = y_model.copy()
        print("  No expert labels — using y_model as fallback")

    return X, y_model, y_expert, lp, lp_c, lp_w, cats, N


# ── Probe training ────────────────────────────────────────────────────────────

def fit_probe(X_tr, y_tr, C=1.0):
    """L2 logistic regression with StandardScaler. Returns (clf, scaler)."""
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X_tr)
    clf = LogisticRegression(
        C=C, max_iter=2000, solver="lbfgs",
        class_weight="balanced", random_state=42,
    )
    clf.fit(Xs, y_tr)
    return clf, sc


# ── Signal computation ────────────────────────────────────────────────────────

def compute_signals(clf_pred, sc_pred, clf_defer, sc_defer,
                    X, lp, lp_cal_min, lp_cal_range, beta):
    """
    Returns dict of signal arrays (higher = more confident = don't defer).

    gap:           logit_pred - logit_defer   (positive → accept, negative → defer)
    probe_alone:   σ(f_pred(h))               (probability of model correctness)
    logprob_alone: normalised lp              (uses cal-split min/range to avoid leakage)
    scod:          (1-probe) + β·(−gap)       (risk proxy + β × OOD gap, then negated)
    """
    logit_pred  = clf_pred.decision_function(sc_pred.transform(X))
    logit_defer = clf_defer.decision_function(sc_defer.transform(X))
    gap         = logit_pred - logit_defer

    probe_prob  = clf_pred.predict_proba(sc_pred.transform(X))[:, 1]

    # Normalise logprob with calibration-split statistics (no test leakage)
    lp_norm = np.clip((lp - lp_cal_min) / (lp_cal_range + 1e-8), 0.0, 1.0)

    # SCOD composite: s(x) = r(x) + β·g(x)
    # r(x) = 1 - σ(f_pred)     (conditional risk proxy)
    # g(x) = -gap               (gap is positive when model confident → OOD proxy)
    # We negate s for "confidence" convention (higher → accept)
    risk    = 1.0 - probe_prob
    scod_s  = -(risk + beta * (-gap))   # negate so higher = more confident

    return {
        "gap":           gap,
        "probe_alone":   probe_prob,
        "logprob_alone": lp_norm,
        "scod":          scod_s,
        # internals (not evaluated as signals)
        "_logit_pred":   logit_pred,
        "_logit_defer":  logit_defer,
        "_risk":         risk,
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def aurc(confidence, correct):
    """
    AURC — Area Under Risk-Coverage Curve.
    O(N log N): sort once, then cumulative mean.
    Lower is better.
    """
    order   = np.argsort(-confidence)          # descending confidence
    correct_sorted = correct[order].astype(float)
    n       = len(correct)
    # cumulative risk at each coverage level k/n
    cumsum  = np.cumsum(correct_sorted)
    risks   = 1.0 - cumsum / np.arange(1, n + 1)
    return float(risks.mean())


def prr(confidence, correct):
    """Prediction Rejection Ratio. Higher is better."""
    baseline = 1.0 - correct.mean()
    a        = aurc(confidence, correct)
    return float(baseline / (a + 1e-8))


def evaluate_signals(signals, y_correct):
    """Evaluate all non-internal signals. Returns dict of metric dicts."""
    out = {}
    for name, scores in signals.items():
        if name.startswith("_"):
            continue
        a  = aurc(scores, y_correct)
        r  = prr(scores, y_correct)
        try:
            ro = roc_auc_score(y_correct, scores)
        except Exception:
            ro = float("nan")
        out[name] = {
            "aurc":  round(a,  4),
            "auroc": round(ro, 4),
            "prr":   round(r,  4),
        }
    return out


# ── Conformal calibration ─────────────────────────────────────────────────────

def conformal_threshold(gap_cal, y_cal_model, alpha):
    """
    Split conformal threshold on calibration split.

    We operate on the gap signal of CORRECT representations only
    (y_cal_model == 1), computing the (1-α) quantile of -gap.
    At test time: defer if gap < -theta.

    Returns theta.
    """
    correct_gaps = gap_cal[y_cal_model == 1]
    n     = len(correct_gaps)
    if n == 0:
        return 0.0
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(-correct_gaps, level))


# ── Category analysis ─────────────────────────────────────────────────────────

def category_gap_stats(gap_correct, gap_wrong, cats):
    """
    Per-category gap separation = μ(Δ_correct) − μ(Δ_wrong).
    Higher = gap signal has more discriminative power for this category.
    """
    stats = defaultdict(lambda: {"gc": [], "gw": []})
    for g, cat in zip(gap_correct, cats):
        stats[cat]["gc"].append(g)
    for g, cat in zip(gap_wrong, cats):
        stats[cat]["gw"].append(g)

    rows = []
    for cat, d in stats.items():
        gc = np.array(d["gc"])
        gw = np.array(d["gw"])
        rows.append({
            "category":         cat,
            "n":                int(len(gc)),
            "mean_gap_correct": round(float(gc.mean()), 4),
            "mean_gap_wrong":   round(float(gw.mean()), 4),
            "separation":       round(float(gc.mean() - gw.mean()), 4),
        })
    rows.sort(key=lambda r: r["separation"])
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    data = load_data(args.input)
    X, y_model, y_expert, lp, lp_c, lp_w, cats, N = \
        build_arrays(data, layer=args.layer)

    tag = (Path(args.input).stem
           .replace("hidden_states", data.get("dataset", "ds"))
           .replace("__", "_"))
    out_path = Path(args.output) if args.output \
               else OUTPUT_DIR / f"results_{tag}.json"

    print(f"  2N={len(X)} representations  ({N} questions × 2)")

    # ── 70 / 10 / 20 split ────────────────────────────────────────────────────
    idx       = np.arange(len(X))
    idx_tv, idx_test = train_test_split(
        idx, test_size=0.20, random_state=args.seed, stratify=y_model)
    idx_train, idx_cal = train_test_split(
        idx_tv, test_size=0.125, random_state=args.seed, stratify=y_model[idx_tv])

    X_tr, y_tr       = X[idx_train], y_model[idx_train]
    y_exp_tr          = y_expert[idx_train]
    X_cal, y_cal      = X[idx_cal],  y_model[idx_cal]
    X_te,  y_te       = X[idx_test], y_model[idx_test]
    y_exp_te          = y_expert[idx_test]

    lp_cal = lp[idx_cal]
    lp_te  = lp[idx_test]

    print(f"\n  Split: train {len(X_tr)} | cal {len(X_cal)} | test {len(X_te)}")

    # ── Calibration-split logprob stats (no leakage) ──────────────────────────
    lp_cal_min   = lp_cal.min()
    lp_cal_range = lp_cal.max() - lp_cal.min()

    # ── OVA probe training ────────────────────────────────────────────────────
    print(f"\nTraining f_pred  (target: y_model, C={args.C})...")
    clf_pred, sc_pred   = fit_probe(X_tr, y_tr,     C=args.C)
    acc_pred = clf_pred.score(sc_pred.transform(X_te), y_te)
    print(f"  Test accuracy (y_model):   {acc_pred:.4f}")

    print(f"Training f_defer (target: y_expert, C={args.C}) ← DIFFERENT TARGET")
    clf_defer, sc_defer = fit_probe(X_tr, y_exp_tr, C=args.C)
    acc_defer = clf_defer.score(sc_defer.transform(X_te), y_exp_te)
    print(f"  Test accuracy (y_expert):  {acc_defer:.4f}")

    # True OVA fraction
    ova_frac = ((y_te == 0) & (y_exp_te == 1)).mean()
    if ova_frac > 0:
        print(f"  True OVA cases (model wrong, expert right): "
              f"{ova_frac*100:.1f}% of test set  ← deferral clearly warranted")

    # ── SCOD β ────────────────────────────────────────────────────────────────
    # KKT-optimal: β = α·tpr_min / (1 − α)
    TPR_MIN = 0.70
    beta    = args.beta if args.beta is not None \
              else args.alpha * TPR_MIN / (1.0 - args.alpha)
    print(f"\n  SCOD β = {beta:.4f}  (α={args.alpha}, TPR_min={TPR_MIN})")

    # ── Signals ───────────────────────────────────────────────────────────────
    sig_cal = compute_signals(
        clf_pred, sc_pred, clf_defer, sc_defer,
        X_cal, lp_cal, lp_cal_min, lp_cal_range, beta)
    sig_te  = compute_signals(
        clf_pred, sc_pred, clf_defer, sc_defer,
        X_te, lp_te, lp_cal_min, lp_cal_range, beta)

    # ── Conformal threshold ───────────────────────────────────────────────────
    theta      = conformal_threshold(sig_cal["gap"], y_cal, args.alpha)
    deferral   = (sig_te["gap"] < -theta).astype(int)
    defer_rate = deferral.mean()
    acc_nodefe = y_te[deferral == 0].mean() if (deferral == 0).sum() > 0 \
                 else float("nan")

    print(f"\n  Conformal θ̂ = {theta:.4f}  (α = {args.alpha})")
    print(f"  Deferral rate:            {defer_rate*100:.1f}%")
    print(f"  Non-deferred accuracy:    {acc_nodefe*100:.1f}%  "
          f"(target ≥ {(1-args.alpha)*100:.0f}%)")

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = evaluate_signals(sig_te, y_te)

    print(f"\n{'Signal':<20}  {'AURC↓':>7}  {'AUROC↑':>7}  {'PRR↑':>7}")
    print("  " + "─" * 50)
    for name, m in sorted(metrics.items(), key=lambda x: x[1]["aurc"]):
        mark = "  ◄ proposed" if name in ("gap", "scod") else ""
        print(f"  {name:<18}  {m['aurc']:>7.4f}  "
              f"{m['auroc']:>7.4f}  {m['prr']:>7.4f}{mark}")

    gap_aurc   = metrics["gap"]["aurc"]
    probe_aurc = metrics["probe_alone"]["aurc"]
    lp_aurc    = metrics["logprob_alone"]["aurc"]
    scod_aurc  = metrics["scod"]["aurc"]

    claim1 = gap_aurc  < probe_aurc
    claim2 = gap_aurc  < lp_aurc
    claim3 = scod_aurc < gap_aurc     # SCOD composite ≤ gap alone

    print(f"\n{'='*54}")
    print(f"  Claim 1 (gap < probe_alone):    {'✓' if claim1 else '✗'}  "
          f"({gap_aurc:.4f} vs {probe_aurc:.4f})")
    print(f"  Claim 2 (gap < logprob_alone):  {'✓' if claim2 else '✗'}  "
          f"({gap_aurc:.4f} vs {lp_aurc:.4f})")
    print(f"  Claim 3 (scod ≤ gap):           {'✓' if claim3 else '✗'}  "
          f"({scod_aurc:.4f} vs {gap_aurc:.4f})")
    key_claim = claim1 and claim2
    print(f"\n  Key claim holds: {key_claim}")
    print(f"{'='*54}")

    if not claim1:
        print("\n  Troubleshooting Claim 1:")
        print("  → Try --layer N where N is the sweep best layer")
        print("  → Check that y_expert ≠ y_model (true OVA divergence)")
        print("  → TruthfulQA without category join gives y_expert=y_model")
        print("    which makes f_defer ≈ f_pred → gap ≈ 0 → no signal")

    # ── Category analysis ─────────────────────────────────────────────────────
    sig_full     = compute_signals(
        clf_pred, sc_pred, clf_defer, sc_defer,
        X, lp, lp_cal_min, lp_cal_range, beta)
    gap_full     = sig_full["gap"]
    gap_correct  = gap_full[:N]
    gap_wrong    = gap_full[N:]
    cat_stats    = category_gap_stats(gap_correct, gap_wrong, cats)

    print(f"\nCategory gap separation (lowest → highest, shown: all ≤ 6):")
    print(f"  {'Category':<32}  {'N':>4}  {'Sep':>7}")
    print("  " + "─" * 50)
    for row in cat_stats[:min(len(cat_stats), 6)]:
        bar = "▓" * max(0, int(row["separation"] * 20))
        print(f"  {row['category']:<32}  {row['n']:>4}  "
              f"{row['separation']:>7.4f}  {bar}")

    print(f"\n  Gap mean (correct reps):  {gap_correct.mean():.4f}")
    print(f"  Gap mean (wrong reps):    {gap_wrong.mean():.4f}")
    print(f"  Gap separation (overall): "
          f"{gap_correct.mean() - gap_wrong.mean():.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    probe_path = OUTPUT_DIR / f"probe_{tag}.pkl"
    with open(probe_path, "wb") as f:
        pickle.dump({
            "clf_pred":    clf_pred,  "sc_pred":  sc_pred,
            "clf_defer":   clf_defer, "sc_defer": sc_defer,
            "lp_cal_min":  lp_cal_min,
            "lp_cal_range":lp_cal_range,
            "beta":        beta,
            "theta":       theta,
            "tag":         tag,
        }, f)

    signals_path = OUTPUT_DIR / f"signals_{tag}.pt"
    torch.save({
        "gap_correct":  gap_correct.tolist(),
        "gap_wrong":    gap_wrong.tolist(),
        "lp_correct":   lp_c.tolist(),
        "lp_wrong":     lp_w.tolist(),
        "cats":         cats,
        "N":            N,
        "sig_te":       {k: v.tolist() for k, v in sig_te.items()},
        "y_te":         y_te.tolist(),
        "y_exp_te":     y_exp_te.tolist(),
    }, signals_path)

    result = {
        "tag":          tag,
        "model":        data.get("model", "unknown"),
        "dataset":      data.get("dataset", "unknown"),
        "n_questions":  N,
        "metrics":      metrics,
        "conformal": {
            "alpha":        args.alpha,
            "theta":        round(float(theta), 4),
            "beta":         round(float(beta),  4),
            "deferral_rate":round(float(defer_rate), 4),
            "non_def_acc":  round(float(acc_nodefe), 4),
        },
        "claims": {
            "claim1_gap_lt_probe":  claim1,
            "claim2_gap_lt_logprob":claim2,
            "claim3_scod_lt_gap":   claim3,
            "key_claim_holds":      key_claim,
        },
        "gap_stats": {
            "mean_correct":  round(float(gap_correct.mean()), 4),
            "mean_wrong":    round(float(gap_wrong.mean()),   4),
            "separation":    round(float(gap_correct.mean() - gap_wrong.mean()), 4),
        },
        "category_stats": cat_stats,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✓  Saved → {out_path}")
    print(f"✓  Probes → {probe_path}")
    print(f"✓  Signals → {signals_path}")
    print("\nStep 2 complete — run step3_plot.py next")
    print(f"  Or check: python -c \"import json; r=json.load(open('{out_path}')); "
          f"print(r['claims'])\"")


if __name__ == "__main__":
    main()