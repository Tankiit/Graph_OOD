"""Analysis tables and static figures for the paired synthetic experiments."""

import json
from collections import defaultdict

import numpy as np


def _group(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def _percentile_bound(rows, probability):
    # Censored observations are known to exceed every observed alpha.
    ordered = sorted(row["alpha_star"] for row in rows if not row["censored"])
    index = max(0, int(np.ceil(probability * len(rows))) - 1)
    return ordered[index] if index < len(ordered) else None


def _coverage(lower, upper, target, horizon, target_censored=False):
    """Infer coverage when possible even if an interval endpoint is censored."""
    if target is None:
        return False if target_censored and upper is not None else None
    if (lower is not None and target < lower) or (upper is not None and target > upper):
        return False
    if lower is None and target <= horizon:
        return False
    if lower is not None and (upper is not None or target <= horizon):
        return True
    return None


def analyze(records, spectral_records, config, output, *, plots=True):
    from run_synth import _write_csv

    keys = ("world", "n_clean", "pi", "detector", "protocol", "probe", "direction")
    grouped = _group(records, keys)
    variability, intervals, deltas = [], [], []
    for key, rows in grouped.items():
        tags = dict(zip(keys, key))
        observed = [row["alpha_star"] for row in rows if not row["censored"]]
        variability.append(
            {
                **tags,
                "pi_actual": rows[0]["pi_actual"],
                "B": len(rows),
                "n_observed": len(observed),
                "censor_fraction": 1 - len(observed) / len(rows),
                "sigma_b": float(np.std(observed, ddof=1))
                if len(observed) == len(rows)
                else None,
            }
        )
        lower, upper = _percentile_bound(rows, 0.05), _percentile_bound(rows, 0.95)
        target = rows[0]["oracle_alpha"]
        intervals.append(
            {
                **tags,
                "pi_actual": rows[0]["pi_actual"],
                "lower": lower,
                "upper": upper,
                "lower_censored": lower is None,
                "upper_censored": upper is None,
                **{
                    key: rows[0][key]
                    for key in (
                        "oracle_alpha",
                        "oracle_kind",
                        "oracle_censored",
                        "oracle_pi_actual",
                        "oracle_n_pool",
                    )
                },
                "covered": _coverage(
                    lower, upper, target, config.alpha_max, rows[0]["oracle_censored"]
                ),
                "interval_identified": lower is not None and upper is not None,
            }
        )
    baseline_keys = (
        "world",
        "b",
        "n_clean",
        "detector",
        "protocol",
        "probe",
        "direction",
    )
    baseline = {
        tuple(row[k] for k in baseline_keys): row for row in records if row["pi"] == 0
    }
    for row in records:
        zero = baseline[tuple(row[k] for k in baseline_keys)]
        usable = not row["censored"] and not zero["censored"]
        predicted = (
            row["analytic_alpha"] - zero["analytic_alpha"]
            if row["analytic_alpha"] is not None and zero["analytic_alpha"] is not None
            else None
        )
        deltas.append(
            {
                **{k: row[k] for k in baseline_keys},
                "pi": row["pi"],
                "pi_actual": row["pi_actual"],
                "delta_alpha": row["alpha_star"] - zero["alpha_star"]
                if usable
                else None,
                "analytic_delta": predicted,
                "pair_censored": not usable,
            }
        )
    # For each probe/direction index, there is one interval from each independent world.
    coverage_keys = (
        "detector",
        "oracle_kind",
        "n_clean",
        "pi",
        "protocol",
        "probe",
        "direction",
    )
    coverage = []
    for key, rows in _group(intervals, coverage_keys).items():
        identified = [
            row
            for row in rows
            if row["interval_identified"] and row["covered"] is not None
        ]
        known = [row for row in rows if row["covered"] is not None]
        successes = sum(bool(row["covered"]) for row in known)
        coverage.append(
            {
                **dict(zip(coverage_keys, key)),
                "R": len(rows),
                "n_identified_intervals": len(identified),
                "n_resolved_coverage": len(known),
                "n_covered": successes,
                "n_unresolved": len(rows) - len(known),
                "coverage": successes / len(rows) if len(known) == len(rows) else None,
                "coverage_lower_bound": successes / len(rows),
                "coverage_upper_bound": (successes + len(rows) - len(known))
                / len(rows),
                "coverage_identified_only": float(
                    np.mean([row["covered"] for row in identified])
                )
                if identified
                else None,
            }
        )
    _write_csv(output / "variability.csv", variability)
    _write_csv(output / "paired_deltas.csv", deltas)
    _write_csv(output / "intervals.csv", intervals)
    _write_csv(output / "coverage.csv", coverage)
    checked = [
        row
        for row in records
        if row["detector"] == "maha" and row["pi"] == 0 and not row["censored"]
    ]
    errors = [abs(row["alpha_star"] - row["exact_fitted_alpha"]) for row in checked]
    summary = {
        "n_paths": len(records),
        "n_spectral_fits": len(spectral_records),
        "n_closed_form_checks": len(checked),
        "closed_form_max_abs_error": max(errors) if errors else None,
        "censored_fraction": float(np.mean([row["censored"] for row in records])),
        "coverage_note": "Maha uses a population oracle; other detectors use an independent finite-reference oracle, not a population truth. Each row uses R independent worlds; paths within a world are dependent.",
        "interval_note": "90% bootstrap percentile intervals; no nominal-coverage guarantee. Unidentified coverage has explicit bounds.",
        "sigma_note": "Reported only when all B path values are observed. Missing sigma is not zero.",
        "delta_note": "Paired delta is missing when either path is censored; plots use complete groups only.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if plots:
        _plots(checked, variability, deltas, coverage, spectral_records, config, output)


def _plots(checked, variability, deltas, coverage, spectral_records, config, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "pdf.fonttype": 42,
        }
    )

    def save(fig, name):
        if name != "01_closed_form":
            for ax in fig.axes:
                ax.set_xticks(sorted(config.pis))
                padding = max(0.01, (max(config.pis) - min(config.pis)) * 0.05)
                ax.set_xlim(min(config.pis) - padding, max(config.pis) + padding)
        fig.tight_layout()
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    if checked:
        exact = [row["exact_fitted_alpha"] for row in checked]
        ax.scatter(exact, [row["alpha_star"] for row in checked], s=12, alpha=0.4)
        limit = max(max(exact), 1)
        ax.plot([0, limit], [0, limit], color="0.5", linewidth=1)
    ax.set(
        xlabel="Closed-form α* (fitted parameters)",
        ylabel="Numerical α*",
        title="Mahalanobis at π = 0; uncensored paths",
    )
    save(fig, "01_closed_form")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, protocol in zip(axes, ("A", "B")):
        for color_index, detector in enumerate(config.detectors):
            for n, style in ((config.n, "-"), (config.n // 2, "--")):
                xs, ys = [], []
                for pi in sorted(config.pis):
                    rows = [
                        r
                        for r in variability
                        if r["protocol"] == protocol
                        and r["detector"] == detector
                        and r["n_clean"] == n
                        and r["pi"] == pi
                        and r["direction"] == "toward"
                    ]
                    xs.append(pi)
                    ys.append(
                        float(np.mean([r["sigma_b"] for r in rows]))
                        if rows and all(r["sigma_b"] is not None for r in rows)
                        else np.nan
                    )
                ax.plot(
                    xs,
                    ys,
                    style,
                    marker="o",
                    color=f"C{color_index}",
                    label=f"{detector}, n={n}",
                )
        ax.set(xlabel="Requested contamination π", title=f"Protocol {protocol}")
    axes[0].set_ylabel("Mean σ_b across worlds/probes (toward)")
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    save(fig, "02_variability")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, protocol in zip(axes, ("A", "B")):
        for n, marker in ((config.n, "o"), (config.n // 2, "s")):
            empirical, predicted = [], []
            for pi in sorted(config.pis):
                rows = [
                    r
                    for r in deltas
                    if r["detector"] == "maha"
                    and r["protocol"] == protocol
                    and r["n_clean"] == n
                    and r["pi"] == pi
                    and r["direction"] == "toward"
                ]
                empirical.append(
                    float(np.mean([r["delta_alpha"] for r in rows]))
                    if rows and all(r["delta_alpha"] is not None for r in rows)
                    else np.nan
                )
                predicted.append(
                    float(np.mean([r["analytic_delta"] for r in rows]))
                    if rows
                    else np.nan
                )
            (line,) = ax.plot(
                sorted(config.pis), empirical, marker=marker, label=f"resampled, n={n}"
            )
            ax.plot(
                sorted(config.pis),
                predicted,
                "--",
                color=line.get_color(),
                label=f"population, n={n}",
            )
        ax.axhline(0, color="0.8", linewidth=0.7)
        ax.set(xlabel="Requested contamination π", title=f"Protocol {protocol}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean paired Δα*; Mahalanobis (toward)")
    save(fig, "03_contamination_shift")

    for detector in config.detectors:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, protocol in zip(axes, ("A", "B")):
            for n in (config.n, config.n // 2):
                for probe in range(config.n_probe):
                    rows = sorted(
                        [
                            r
                            for r in coverage
                            if r["detector"] == detector
                            and r["protocol"] == protocol
                            and r["n_clean"] == n
                            and r["probe"] == probe
                            and r["direction"] == "toward"
                        ],
                        key=lambda r: r["pi"],
                    )
                    if rows:
                        xs = [r["pi"] for r in rows]
                        lower = np.array([r["coverage_lower_bound"] for r in rows])
                        upper = np.array([r["coverage_upper_bound"] for r in rows])
                        (line,) = ax.plot(
                            xs,
                            [
                                r["coverage"] if r["coverage"] is not None else np.nan
                                for r in rows
                            ],
                            marker="o",
                            label=f"n={n}, probe {probe}",
                        )
                        if np.any(upper > lower):
                            ax.fill_between(
                                xs,
                                lower,
                                upper,
                                color=line.get_color(),
                                alpha=0.15,
                                label="censoring bounds",
                            )
            ax.axhline(0.9, color="0.4", linestyle="--", linewidth=1)
            ax.set(
                xlabel="Requested contamination π",
                title=f"{detector}: protocol {protocol}",
                ylim=(-0.03, 1.03),
            )
            if ax.lines:
                ax.legend(fontsize=7)
        axes[0].set_ylabel(f"90% interval coverage; R={config.R} worlds")
        kind = "population" if detector == "maha" else "finite reference"
        fig.suptitle(f"Oracle: {kind}")
        save(fig, "04_coverage" if detector == "maha" else f"04_coverage_{detector}")

    if spectral_records:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for n in (config.n, config.n // 2):
            for ax, metric in zip(axes, ("fiedler_cosine", "auroc_abs")):
                means = []
                for pi in sorted(config.pis):
                    values = [
                        r[metric]
                        for r in spectral_records
                        if r["n_clean"] == n and r["pi"] == pi and r[metric] is not None
                    ]
                    means.append(float(np.mean(values)) if values else np.nan)
                ax.plot(sorted(config.pis), means, marker="o", label=f"n={n}")
                ax.set(xlabel="Requested contamination π", ylim=(-0.03, 1.03))
                ax.legend()
        axes[0].set_ylabel("|cos(Fiedler, centered P/Q indicator)|")
        axes[0].set_title("Undefined for a pure pool")
        axes[1].set_ylabel("Held-out AUROC of |Δλ₂|")
        axes[1].axhline(0.5, color="0.6", linestyle="--", linewidth=1)
        save(fig, "05_spectral")
