"""Paired synthetic experiments. Run ``python run_synth.py --help``."""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from analytic import alpha_star_maha, alpha_star_maha_contaminated
from detectors import KNN, Linear, Maha, Spectral
from paths import first_rejection, random_dirs
from src.data.synth import contaminant_count, draw, make_world, paired_pool, splits


@dataclass
class Config:
    d: int = 3
    sep: float = 3.0
    n: int = 32
    n_calib: int = 64
    n_probe: int = 2
    n_contaminant: int = 64
    n_oracle: int = 64
    B: int = 20
    R: int = 3
    m: int = 1
    pis: tuple = (0.0, 0.1, 0.2)
    detectors: tuple = ("maha", "knn", "spectral", "linear")
    quantile: float = 0.95
    alpha_max: float = 12.0
    alpha_steps: int = 25
    refine_steps: int = 24
    k: int = 5
    seed: int = 0

    def validate(self):
        for name in (
            "d",
            "n",
            "n_calib",
            "n_probe",
            "n_contaminant",
            "n_oracle",
            "B",
            "R",
            "alpha_steps",
            "k",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.B < 2 or self.R < 2 or self.n < 4 or self.alpha_steps < 2:
            raise ValueError("require B,R >= 2, n >= 4, alpha_steps >= 2")
        if (
            not isinstance(self.m, (int, np.integer))
            or self.m < 0
            or not isinstance(self.refine_steps, (int, np.integer))
            or self.refine_steps < 0
        ):
            raise ValueError("m and refine_steps must be nonnegative integers")
        if not np.isfinite(self.sep) or self.sep < 0:
            raise ValueError("sep must be finite and nonnegative")
        if (
            not 0 < self.quantile < 1
            or not np.isfinite(self.alpha_max)
            or self.alpha_max <= 0
        ):
            raise ValueError("require quantile in (0,1) and positive finite alpha_max")
        if not self.pis or 0 not in self.pis or len(set(self.pis)) != len(self.pis):
            raise ValueError(
                "pis must be unique and include zero for paired differences"
            )
        if any(not np.isfinite(pi) or not 0 <= pi < 1 for pi in self.pis):
            raise ValueError("all pis must lie in [0,1)")
        if (
            not self.detectors
            or len(set(self.detectors)) != len(self.detectors)
            or set(self.detectors) - {"maha", "knn", "spectral", "linear"}
        ):
            raise ValueError(
                "detectors must be unique names from maha, knn, spectral, linear"
            )
        if "knn" in self.detectors and self.k > self.n // 2:
            raise ValueError("k must not exceed n//2")
        if self.n_oracle < 2 or ("knn" in self.detectors and self.n_oracle < self.k):
            raise ValueError("n_oracle must be at least 2 and at least k for kNN")


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resample(X, rng, n=None):
    n = len(X) if n is None else n
    return X[rng.integers(len(X), size=n)] if n else X[:0]


def _auroc(clean_scores, other_scores):
    scores = np.concatenate((clean_scores, other_scores))
    ranks = rankdata(scores)
    n0, n1 = len(clean_scores), len(other_scores)
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def _factory(name, reference, config):
    return {
        "maha": lambda: Maha(),
        "knn": lambda: KNN(config.k),
        "spectral": lambda: Spectral(),
        "linear": lambda: Linear(reference),
    }[name]()


def _oracle_targets(world, data, path_specs, rng, config, alphas):
    """Independent finite-reference targets, or exact population Maha targets."""
    calibration = draw(world, config.n_oracle, 0.0, rng)[0]
    reference = draw(world, config.n_oracle, 1.0, rng)[0]
    fractions = {}
    for n in (config.n, config.n // 2):
        for pi in config.pis:
            count = contaminant_count(n, pi)
            fractions[n, pi] = count / (n + count)
    injection = draw(
        world, contaminant_count(config.n_oracle, max(fractions.values())), 1.0, rng
    )[0]
    cached, targets = {}, {}
    for (n, pi), actual in fractions.items():
        for name in config.detectors:
            cache_key = (actual, name)
            if cache_key not in cached:
                entries = {}
                if name == "maha":
                    for protocol in ("A", "B"):
                        for probe, direction, x, v in path_specs:
                            alpha = alpha_star_maha_contaminated(
                                x, v, world, actual, protocol, quantile=config.quantile
                            )
                            entries[protocol, probe, direction] = {
                                "oracle_alpha": alpha,
                                "oracle_censored": False,
                                "oracle_kind": "population",
                                "oracle_pi_actual": actual,
                                "oracle_n_pool": None,
                            }
                else:
                    pool, labels = paired_pool(data["oracle"], injection, actual)
                    fitted = _factory(name, reference, config).fit(pool)
                    for protocol, calibration_data in (("A", calibration), ("B", pool)):
                        threshold = float(
                            np.quantile(fitted.score(calibration_data), config.quantile)
                        )
                        for probe, direction, x, v in path_specs:
                            alpha, censored, _ = first_rejection(
                                fitted.score,
                                threshold,
                                x,
                                v,
                                alphas,
                                refine_steps=config.refine_steps,
                            )
                            entries[protocol, probe, direction] = {
                                "oracle_alpha": alpha,
                                "oracle_censored": censored,
                                "oracle_kind": "finite_reference",
                                "oracle_pi_actual": float(labels.mean()),
                                "oracle_n_pool": len(pool),
                            }
                cached[cache_key] = entries
            for (protocol, probe, direction), entry in cached[cache_key].items():
                targets[n, pi, name, protocol, probe, direction] = entry
    return targets


def run(config, output, *, plots=True):
    """Write raw paths, diagnostics, analysis tables, metadata and optional PDFs.

    Each outer replication draws an independent world and datasets. Inside a
    world, B nonparametric bootstrap replicates resample dev, clean calib,
    contaminant reference, and the separate Q injection reservoir. Probe
    points/directions stay fixed. n/2 uses prefixes of the same bootstrap
    samples; all pi levels retain the exact same clean rows at a given n.
    """
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    alphas = np.linspace(0, config.alpha_max, config.alpha_steps)
    records, spectral_records = [], []
    seeds = np.random.SeedSequence(config.seed).spawn(config.R)
    world_arrays = {}
    for r, seed in enumerate(seeds):
        world_seed, data_seed, bootstrap_seed = seed.spawn(3)
        world = make_world(config.d, config.sep, world_seed)
        rng, boot_rng = (
            np.random.default_rng(data_seed),
            np.random.default_rng(bootstrap_seed),
        )
        data = splits(
            world,
            {
                "dev": config.n,
                "calib": config.n_calib,
                "probe": config.n_probe,
                "contaminant": config.n_contaminant,
                "oracle": config.n_oracle,
            },
            rng,
        )
        injection = draw(world, contaminant_count(config.n, max(config.pis)), 1.0, rng)[
            0
        ]
        oracle_q = draw(world, config.n_oracle, 1.0, rng)[0]
        path_specs = []
        for probe, x in enumerate(data["probe"]):
            toward = world.mu + world.Delta - x
            norm = np.linalg.norm(toward)
            toward = toward / norm if norm else random_dirs(x, 1, rng)[0]
            directions = [("toward", toward)] + [
                (f"random_{j}", v) for j, v in enumerate(random_dirs(x, config.m, rng))
            ]
            path_specs.extend((probe, name, x, v) for name, v in directions)
        world_arrays.update(
            {
                f"world_{r}_mu": world.mu,
                f"world_{r}_Sigma": world.Sigma,
                f"world_{r}_Delta": world.Delta,
                f"world_{r}_probes": data["probe"],
                f"world_{r}_directions": np.stack([v for _, _, _, v in path_specs]),
            }
        )
        # Oracle fits and targets stay fixed across bootstrap replicates.
        targets = _oracle_targets(world, data, path_specs, rng, config, alphas)
        for b in range(config.B):
            clean_boot = _resample(data["dev"], boot_rng)
            q_boot = _resample(injection, boot_rng)
            calib_boot = _resample(data["calib"], boot_rng)
            reference_boot = _resample(data["contaminant"], boot_rng)
            for n in (config.n, config.n // 2):
                for pi in sorted(config.pis):
                    pool, labels = paired_pool(clean_boot[:n], q_boot, pi)
                    actual = float(labels.mean())
                    common = {
                        "world": r,
                        "b": b,
                        "n_clean": n,
                        "n_pool": len(pool),
                        "pi": pi,
                        "pi_actual": actual,
                    }
                    for name in config.detectors:
                        fitted = _factory(name, reference_boot, config).fit(pool)
                        # The fit is identical under both protocols; only calibration changes.
                        thresholds = {
                            "A": float(
                                np.quantile(fitted.score(calib_boot), config.quantile)
                            ),
                            "B": float(
                                np.quantile(fitted.score(pool), config.quantile)
                            ),
                        }
                        if name == "spectral":
                            indicator = labels.astype(float) - actual
                            denom = np.linalg.norm(indicator) * np.linalg.norm(
                                fitted.fiedler_
                            )
                            cosine = (
                                float(abs(indicator @ fitted.fiedler_) / denom)
                                if denom
                                else None
                            )
                            signed_p = fitted.score_signed(data["oracle"])
                            signed_q = fitted.score_signed(oracle_q)
                            spectral_records.append(
                                {
                                    **common,
                                    "fiedler_cosine": cosine,
                                    "fiedler_gap": fitted.fiedler_gap_,
                                    "lambda2": fitted.lambda2_,
                                    "bandwidth": fitted.bandwidth_,
                                    "auroc_abs": _auroc(
                                        np.abs(signed_p), np.abs(signed_q)
                                    ),
                                    "auroc_signed": _auroc(signed_p, signed_q),
                                }
                            )
                        for protocol, threshold in thresholds.items():
                            for probe, direction, x, v in path_specs:
                                alpha, censored, recross = first_rejection(
                                    fitted.score,
                                    threshold,
                                    x,
                                    v,
                                    alphas,
                                    refine_steps=config.refine_steps,
                                )
                                exact = (
                                    alpha_star_maha(
                                        x, v, fitted.mu, fitted.Sigma, threshold
                                    )
                                    if name == "maha"
                                    else None
                                )
                                target = targets[
                                    n, pi, name, protocol, probe, direction
                                ]
                                prediction = (
                                    target["oracle_alpha"] if name == "maha" else None
                                )
                                records.append(
                                    {
                                        **common,
                                        "detector": name,
                                        "protocol": protocol,
                                        "probe": probe,
                                        "direction": direction,
                                        "alpha_star": alpha,
                                        "censored": censored,
                                        "n_recross": recross,
                                        "threshold": threshold,
                                        "exact_fitted_alpha": exact,
                                        "analytic_alpha": prediction,
                                        **target,
                                    }
                                )
            print(f"world {r + 1}/{config.R}, bootstrap {b + 1}/{config.B}", flush=True)
    _write_csv(output / "paths.csv", records)
    _write_csv(output / "spectral.csv", spectral_records)
    np.savez_compressed(output / "worlds.npz", **world_arrays)
    metadata = {
        "config": asdict(config),
        "coverage_target": "Maha: exact population boundary. Other detectors: independent finite-reference fit on oracle split, conditional on each probe/direction",
        "resampling": "nonparametric bootstrap of dev, calibration, independent Q reference and Q injection reservoir",
        "sigma": "SD across B replicates; undefined when any replicate is censored",
        "interval": "90% percentile interval using empirical inverse CDF (no interpolation)",
        "spectral": "exact insertion delta lambda2 of unnormalized RBF Laplacian, frozen per-fit bandwidth",
        "toward": "unit vector from each probe toward the Q mean",
        "pool_size": "n clean points + round(n*pi/(1-pi)) Q points; predictions use actual fraction",
        "calibration": "A: independent clean empirical quantile; B: literal in-pool empirical quantile, including self for kNN",
        "oracle_split": "P oracle split supplies finite-reference fits and held-out P evaluation for bootstrap spectral AUROC; oracle calibration, Q reference and Q injection are independent draws. Reference fitting may round the target fraction; oracle_pi_actual records it.",
        "censoring": "blank alpha is right-censored at alpha_max; grid may miss between-node excursions",
        "numpy_version": np.__version__,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    from analysis_synth import analyze

    analyze(records, spectral_records, config, output, plots=plots)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    for name in (
        "d",
        "n",
        "n_calib",
        "n_probe",
        "n_contaminant",
        "n_oracle",
        "B",
        "R",
        "m",
        "alpha_steps",
        "refine_steps",
        "k",
        "seed",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(defaults, name)
        )
    for name in ("sep", "quantile", "alpha_max"):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=float, default=getattr(defaults, name)
        )
    parser.add_argument("--pis", nargs="+", type=float, default=defaults.pis)
    parser.add_argument(
        "--detectors", nargs="+", choices=defaults.detectors, default=defaults.detectors
    )
    parser.add_argument("--output", type=Path, default=Path("results/synth"))
    parser.add_argument("--no-plots", action="store_true")
    args = vars(parser.parse_args())
    output, no_plots = args.pop("output"), args.pop("no_plots")
    run(Config(**args), output, plots=not no_plots)


if __name__ == "__main__":
    main()
