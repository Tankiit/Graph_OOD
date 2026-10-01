"""Evaluate steering swarms with the instrument of the AISTATS 2027 draft.

    uv run python -m actdist.eval_steer outputs/steering/<name>      # re-run evaluation of a trained experiment
    uv run python -m actdist.eval_steer outputs/steering/<name> --detectors energy knn mahalanobis
    uv run python -m actdist.eval_steer outputs/steering/<name> --eval-vectors 128 --out all_vectors

--detectors evaluates with other detectors than the ones the run was configured with; the override is
recorded in <name>/eval_overrides.json and config.json is left as trained.

For each swarm, three families of unit directions start from the same held-out probes:
learned (a random subset of `eval_vectors` of the swarm), toward (v_toward: OOD-dev mean minus
ID-dev mean at the layer, sign flipped for ood2id) and random (`n_random` matched random directions).
Each direction is walked along the grid alpha in [0, alpha_max] (the trained radius is on the grid).

Reported separately, never derived from one another (draft, "What is measured"):
  standard     AUROC, FPR@95, flip rate at t_D, for the clean model and at alpha = radius
  path         first-flip length (alpha* first rejection for id2ood, alpha-dagger first acceptance
               for ood2id), censoring, currently / ever flipped fraction per alpha, recrossings,
               ratio of learned to random first-flip length
  variability  sigma_b: spread of the median first-flip length over bootstrap resamples of the
               calibration split that sets t_D
"""

import json
from pathlib import Path

import numpy as np
import torch

from .detectors import auroc, fpr_at_tpr
from .steer import steer

FAMILIES = ("learned", "toward", "random")


@torch.no_grad()
def walk(ctx, h: torch.Tensor, dirs: torch.Tensor, alphas: np.ndarray) -> dict:
    """Scores and predictions of every probe along every direction: [M, A, N] per detector."""
    cfg, split = ctx.cfg, ctx.split
    vecs = (torch.as_tensor(alphas, dtype=torch.float32, device=ctx.device)[None, :, None]
            * dirs[:, None, :]).reshape(-1, dirs.shape[1])                     # [M*A, D]
    M, A, N = len(dirs), len(alphas), len(h)
    budget = cfg.vec_chunk * max(cfg.bs_id, cfg.bs_ood) * 2               # samples per suffix call
    pb = min(N, max(cfg.bs_id, cfg.bs_ood))
    vb = max(1, budget // pb)
    scores = {d: np.empty((M * A, N), np.float32) for d in cfg.detectors}
    pred = np.empty((M * A, N), np.int8)
    for p in range(0, N, pb):
        hp = h[p:p + pb]
        for c in range(0, len(vecs), vb):
            v = vecs[c:c + vb]
            with ctx.autocast():
                feat, logits = split.suffix(steer(hp, v))
            logits = logits.float().view(len(v), len(hp), -1)
            feat = feat.float().view(len(v), len(hp), -1)
            for d in cfg.detectors:
                scores[d][c:c + len(v), p:p + len(hp)] = ctx.score(d, logits, feat)
            pred[c:c + len(v), p:p + len(hp)] = logits.argmax(-1).cpu().numpy()
    return {"scores": {d: s.reshape(M, A, N) for d, s in scores.items()}, "pred": pred.reshape(M, A, N)}


def first_flip(flip: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """flip [..., A, N] -> first alpha at which each probe flips, inf when censored: [..., N]."""
    first = alphas[flip.argmax(-2)]
    return np.where(flip.any(-2), first, np.inf)


def path_metrics(flip: np.ndarray, alphas: np.ndarray) -> dict:
    """flip [M, A, N] for one family -> path metrics (per-direction arrays summarised over directions)."""
    fstar = first_flip(flip, alphas)                                  # [M, N]
    med = np.quantile(fstar, 0.5, axis=1, method="lower")              # median first-flip per direction
    trans = np.abs(np.diff(flip.astype(np.int8), axis=1)).sum(1)      # [M, N] state changes along the path
    recross = trans - (~flip[:, 0] & flip.any(1))                     # changes after the first flip
    return {
        "first_flip_median": _summ(med),
        "censored": float((~flip.any(1)).mean()),
        "recrossings_mean": float(recross.mean()),
        "current": flip.mean(2).mean(0).tolist(),                     # fraction flipped at alpha
        "ever": np.maximum.accumulate(flip, axis=1).mean(2).mean(0).tolist(),
        "_median_per_dir": med,
    }


def _summ(x: np.ndarray) -> dict:
    """Quantile summary; 'lower' quantiles so censored (inf) values never interpolate to nan."""
    x = np.asarray(x, float)
    q = lambda p: float(np.quantile(x, p, method="lower"))
    return {"median": q(0.5), "q25": q(0.25), "q75": q(0.75), "min": float(x.min()), "max": float(x.max())}


def sigma_b(scores: np.ndarray, cal: np.ndarray, sign: float, alphas, tpr: float, B: int, seed: int) -> np.ndarray:
    """Std over B calibration resamples of the median first-flip length, per direction: [M]."""
    rng = np.random.default_rng(seed)
    meds = []
    for _ in range(B):
        t = np.quantile(rng.choice(cal, len(cal), replace=True), tpr)
        flip = scores > t if sign > 0 else scores <= t
        meds.append(np.quantile(first_flip(flip, alphas), 0.5, axis=1, method="lower"))
    meds = np.stack(meds)                                               # [B, M]
    finite = np.isfinite(meds).all(0)
    out = np.full(meds.shape[1], np.inf)
    out[finite] = meds[:, finite].std(0)
    return out


def evaluate(ctx, directions: dict, out_dir: Path):
    cfg = ctx.cfg
    r = ctx.radius
    a_max = cfg.alpha_max or 4 * r
    alphas = np.unique(np.concatenate([np.linspace(0, a_max, cfg.alpha_steps), [r]]))
    i_r = int(np.where(alphas == r)[0][0])
    rng = np.random.default_rng(cfg.seed + 1)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    rand = torch.nn.functional.normalize(torch.randn(cfg.n_random, ctx.dim, generator=gen), dim=1).to(ctx.device)

    probes = {"id": ctx.encode(ctx.id_probe), "ood": ctx.encode(ctx.ood_probe)}
    clean_scores = {s: {d: ctx.score(d, p["logits"], p["feat"]) for d in cfg.detectors}
                    for s, p in probes.items()}
    labels = probes["id"]["label"].numpy()
    metrics = {
        "radius": r, "act_norm_median": ctx.act_norm_median, "alphas": alphas.tolist(), "radius_index": i_r,
        "thresholds": ctx.thresholds,
        "clean": {d: {"auroc": auroc(clean_scores["id"][d], clean_scores["ood"][d]),
                      "fpr95": fpr_at_tpr(clean_scores["id"][d], clean_scores["ood"][d], cfg.tpr),
                      "id_accepted_at_t": float((clean_scores["id"][d] <= ctx.thresholds[d]).mean()),
                      "ood_accepted_at_t": float((clean_scores["ood"][d] <= ctx.thresholds[d]).mean())}
                  for d in cfg.detectors},
        "clean_accuracy": float((probes["id"]["logits"].argmax(1).numpy() == labels).mean()),
        "swarms": {},
    }
    arrays = {"alphas": alphas}

    for name, dirs_all in directions.items():
        sign = 1.0 if name == "id2ood" else -1.0
        src, tgt = ("id", "ood") if name == "id2ood" else ("ood", "id")
        sub = np.sort(rng.choice(len(dirs_all), min(cfg.eval_vectors, len(dirs_all)), replace=False))
        fam_dirs = {"learned": dirs_all[sub].to(ctx.device), "toward": sign * ctx.v_toward[None], "random": rand}
        stacked = torch.cat([fam_dirs[f] for f in FAMILIES])
        bounds = np.cumsum([0] + [len(fam_dirs[f]) for f in FAMILIES])
        res = walk(ctx, probes[src]["h"], stacked, alphas)

        U = torch.nn.functional.normalize(dirs_all.float(), dim=1)
        cos = (U @ U.T).cpu().numpy()
        off = cos[~np.eye(len(cos), dtype=bool)]
        m = {"diversity": {"abs_cos_mean": float(np.abs(off).mean()) if len(off) else 0.0,
                           "cos_max": float(off.max()) if len(off) else 1.0,
                           "cos_to_toward": _summ((U @ (sign * ctx.v_toward)).cpu().numpy())},
             "eval_vector_indices": sub.tolist()}
        for d in cfg.detectors:
            S, t = res["scores"][d], ctx.thresholds[d]
            flip = S > t if sign > 0 else S <= t
            sb = sigma_b(S, ctx.cal_scores[d], sign, alphas, cfg.tpr, cfg.bootstrap_B, cfg.seed)
            md = {}
            for f, (lo, hi) in zip(FAMILIES, zip(bounds[:-1], bounds[1:])):
                pm = path_metrics(flip[lo:hi], alphas)
                med = pm.pop("_median_per_dir")
                at_r = S[lo:hi, i_r]                                    # steered scores at alpha = radius
                other = clean_scores[tgt][d]
                if sign > 0:   # steered ID vs clean OOD
                    au = [auroc(s, other) for s in at_r]
                    fp = [fpr_at_tpr(s, other, cfg.tpr) for s in at_r]
                else:          # clean ID vs steered OOD
                    au = [auroc(other, s) for s in at_r]
                    fp = [fpr_at_tpr(other, s, cfg.tpr) for s in at_r]
                md[f] = {"standard_at_radius": {"flip_rate": _summ(flip[lo:hi, i_r].mean(1)),
                                                "auroc": _summ(au), "fpr95": _summ(fp)},
                         "path": pm,
                         "sigma_b": _summ(sb[lo:hi][np.isfinite(sb[lo:hi])]) if np.isfinite(sb[lo:hi]).any() else None,
                         "sigma_b_censored_dirs": int((~np.isfinite(sb[lo:hi])).sum())}
                md[f]["_med"] = med
            lm, rm = md["learned"].pop("_med"), md["random"].pop("_med")
            md["toward"].pop("_med")
            ml, mr = (float(np.quantile(x, 0.5, method="lower")) for x in (lm, rm))
            finite = np.isfinite(ml) and np.isfinite(mr) and mr > 0
            md["ratio_learned_to_random"] = ml / mr if finite else None  # None: a side is censored (or flips at 0)
            m[d] = md
            arrays[f"{name}__{d}__scores"] = S.astype(np.float32)  # float16 overflows for mahalanobis
        if name == "id2ood":
            acc = (res["pred"] == labels[None, None, :]).mean(2)            # [M, A]
            m["accuracy_at_radius"] = {f: _summ(acc[lo:hi, i_r]) for f, (lo, hi) in zip(FAMILIES, zip(bounds[:-1], bounds[1:]))}
            arrays[f"{name}__accuracy"] = acc.astype(np.float32)
        arrays[f"{name}__family_bounds"] = bounds
        metrics["swarms"][name] = m

    np.savez_compressed(out_dir / "paths.npz", **arrays)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _print_summary(metrics, cfg)


def _print_summary(m: dict, cfg):
    d = cfg.detectors[0]
    c = m["clean"][d]
    print(f"[{d}] clean: AUROC {c['auroc']:.3f}  FPR95 {c['fpr95']:.3f}  ID accepted {c['id_accepted_at_t']:.3f}  "
          f"OOD accepted {c['ood_accepted_at_t']:.3f}  accuracy {m['clean_accuracy']:.4f}")
    for name, s in m["swarms"].items():
        for f in FAMILIES:
            x = s[d][f]
            fr = x["standard_at_radius"]["flip_rate"]
            print(f"[{d}] {name:7s} {f:8s} flip@r median {fr['median']:.3f} (max {fr['max']:.3f})  "
                  f"AUROC@r {x['standard_at_radius']['auroc']['median']:.3f}  "
                  f"first-flip {x['path']['first_flip_median']['median']:.3g}  censored {x['path']['censored']:.3f}")
        ratio = s[d]["ratio_learned_to_random"]
        print(f"[{d}] {name:7s} learned/random first-flip ratio {'censored' if ratio is None else f'{ratio:.3f}'}  "
              f"|cos| between vectors {s['diversity']['abs_cos_mean']:.3f}")


def main():
    import argparse
    from dataclasses import replace
    from datetime import datetime

    from .steer_config import SteerConfig
    from .train_steer import Context

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp", type=Path, help="outputs/steering/<name>")
    p.add_argument("--detectors", nargs="+", help="evaluate with these detectors instead of the configured ones")
    p.add_argument("--eval-vectors", type=int, help="learned directions walked per swarm (>= K: every vector)")
    p.add_argument("--out", help="write the results to <exp>/<out>/ instead of overwriting <exp>/")
    args = p.parse_args()
    exp = args.exp
    cfg = SteerConfig(**json.loads((exp / "config.json").read_text())).validate()
    overrides = {}
    if args.detectors:
        overrides["detectors"] = args.detectors
    if args.eval_vectors:
        overrides["eval_vectors"] = args.eval_vectors
    cfg = replace(cfg, **overrides).validate()
    out_dir = exp / args.out if args.out else exp
    out_dir.mkdir(parents=True, exist_ok=True)
    if overrides:
        (out_dir / "eval_overrides.json").write_text(json.dumps(
            {**overrides, "knn_k": cfg.knn_k, "ref_size": cfg.ref_size,
             "date": datetime.now().isoformat(timespec="seconds")}, indent=2))
    ctx = Context(cfg)
    saved = torch.load(exp / "swarms.pt", map_location="cpu")
    evaluate(ctx, {n: d.to(ctx.device) for n, d in saved["directions"].items()}, out_dir)


if __name__ == "__main__":
    main()
