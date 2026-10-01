# Steering

Representation steering utilities and synthetic OOD detection experiments.

## Synthetic experiment

The data loaders in `src/data/synth.py` and the modules `detectors.py`, `paths.py`,
`analytic.py`, `run_synth.py`, and `analysis_synth.py` implement the Gaussian experiment.
Install the project's base dependencies; rendering figures additionally requires Matplotlib.
The plotting dependency is available through `pip install -e '.[synthetic]'`.

```python
import numpy as np
from src.data.synth import make_world, draw, splits
from detectors import Maha, KNN, Spectral, Linear, calibrate, calibrate_on_pool
from paths import first_rejection, random_dirs
from analytic import alpha_star_maha, alpha_star_maha_contaminated

rng = np.random.default_rng(42)
world = make_world(d=4, sep=3.0, seed=0)
X, labels = draw(world, n=200, pi=0.1, rng=rng)
data = splits(world, dict(dev=100, calib=200, probe=10,
                          contaminant=100, oracle=1000), rng)
model, threshold = calibrate(Maha(), data['dev'], data['calib'])
x = data['probe'][0]
v = random_dirs(x, 1, rng)[0]
alpha, censored, recrossings = first_rejection(
    model.score, threshold, x, v, np.linspace(0, 20, 101))
exact = alpha_star_maha(x, v, model.mu, model.Sigma, threshold)
prediction = alpha_star_maha_contaminated(x, v, world, pi=0.1, protocol='A')
linear = Linear(data['contaminant']).fit(data['dev'])
spectral = Spectral().fit(data['dev'])
scores = spectral.score_components(data['probe'])  # dlambda2 and abs_dlambda2
```

Run a small end-to-end check, including all detectors and all five figures:

```sh
python run_synth.py --n 16 --d 2 --B 4 --R 2 --n-calib 20 \
  --n-probe 1 --n-contaminant 20 --n-oracle 12 --m 1 \
  --pis 0 0.2 --alpha-steps 13 --output results/synth_smoke
python -m pytest tests/test_synthetic.py
```

`python run_synth.py --help` lists all settings. Increase `--B` (e.g. 200) and
`--R` (e.g. 100) for coverage studies. The defaults are exploratory, not precise
coverage estimates. Exact spectral scoring performs a dense eigenvalue solve
for every query; use `--detectors maha knn linear` for larger runs and run the
spectral study separately at smaller pool sizes. `--no-plots` skips Matplotlib.

### Definitions and pairing

- P is N(μ,Σ), Q is N(μ+Δ,Σ), and ΔᵀΣ⁻¹Δ = sep². `World` exposes named
  fields and can be unpacked into `(mu, Sigma, Delta)`.
- `splits` draws independent arrays: dev, calib, probe, and oracle from P,
  contaminant from Q. Mixture labels returned by `draw` are for analysis only.
  Linear explicitly uses the separate known-Q reference split.
- Each outer replication draws a new world and datasets. Each of B bootstrap
  replicates resamples dev, calibration, the Q reference, and a separate Q
  injection reservoir. Probes and directions are fixed within a world.
- At every requested π, the pool retains **all the same clean points** and adds
  a nested prefix of Q points. With n clean points, it adds round(nπ/(1−π));
  π is a fraction of the resulting pool, not a fraction of n. Both requested
  and actual fractions are recorded; analytic predictions use the latter.
  The n/2 condition uses prefixes of the same bootstrap samples.
- Protocol A thresholds independent clean calibration scores at the specified
  quantile (default .95). Protocol B thresholds scores of the fitting pool.
  This is an empirical quantile, not a finite-sample conformal correction.
  In-pool kNN scores include the point itself among its neighbours.
- Maha uses squared Mahalanobis distance and sample covariance with an absolute
  1e-8 diagonal ridge. Linear uses Σ̂⁻¹(mean(Q-reference)−mean(pool)).
- Spectral uses the **unnormalized** RBF Laplacian with zero diagonal weights.
  Its bandwidth is the median nonzero pairwise distance of the fitting pool,
  fixed for candidate insertions. Each query is separately appended to the
  pool. `score_signed` returns λ₂(augmented)−λ₂(pool); `score` returns its absolute
  value. The Fiedler cosine uses a centered P/Q indicator and an absolute cosine
  because eigenvector sign is arbitrary. It is undefined for a pure pool;
  eigenvalue gaps are saved to flag non-unique or unstable Fiedler directions.
- Paths are x+αv, with Euclidean unit directions and no point normalization or
  projection. “Toward” points from the probe to μ+Δ. Random directions are
  isotropic. Rejection is strictly score > threshold. Initially rejected
  probes have α*=0. The first detected crossing is refined by bisection;
  recrossings count subsequent grid state changes. A finite grid can miss
  excursions between grid nodes for nonmonotone scores.

### Analytic benchmark

The population mixture mean is μ+πΔ and its covariance is
Σ+π(1−π)ΔΔᵀ. In whitened coordinates, the score distribution is

```
U + (Z + offset)^2 / lambda,
U ~ chi-square(d-1), Z ~ N(0,1), independently,
lambda = 1 + pi*(1-pi)*sep^2,
offset = -pi*sep for P, or (1-pi)*sep for Q.
```

The code integrates this distribution and numerically solves its quantile:
protocol A uses the P distribution; protocol B uses their weighted mixture.
The path boundary then follows from a quadratic equation. At π=0 the population
threshold is the chi-square(d) quantile. These population predictions include
mean offset, covariance inflation, and threshold movement. They are not exact
finite-sample predictions. The separate closed-form implementation check uses
the *fitted* mean, covariance (including ridge), and empirical threshold, so it
should agree with the numerical path to the bisection tolerance.

### Outputs

The output directory contains raw `paths.csv`, `spectral.csv`, `worlds.npz`,
configuration and conventions in `metadata.json`, and checks in `summary.json`.
Analysis tables and PNG/PDF figures cover:

1. Numerical versus exact fitted Mahalanobis α* at π=0 (`01_closed_form`).
2. Bootstrap SD σ_b at n and n/2 (`variability.csv`, `02_variability`).
3. Paired Δα*(π)=α*(π)−α*(0), with population Mahalanobis predictions for both
   protocols (`paired_deltas.csv`, `03_contamination_shift`).
4. Coverage of 90% bootstrap percentile intervals over R independent worlds
   (`intervals.csv`, `coverage.csv`, `04_coverage*`). Mahalanobis uses its exact
   population α*. The other detectors use an independent fit on the oracle
   split, with independent clean calibration, Q reference and Q injection.
   These finite-reference targets are explicitly labeled; increase `n_oracle`
   to study their stability. Oracle pool fractions are separately recorded
   because counts are rounded. Coverage is reported separately by probe/direction index;
   multiple paths from the same world are not counted as independent worlds.
5. Fiedler alignment and held-out spectral AUROC versus π (`spectral.csv`,
   `05_spectral`). The oracle split supplies independent evaluation data;
   population Mahalanobis coverage uses known world parameters directly.

The CSV files include every direction. Summary curves use “toward” paths;
closed-form scatter includes all directions. Blank α* means right censoring at
`alpha_max`, never a substituted horizon value. SD is reported only when all B
values are observed. Paired deltas are blank when either path is censored, and
mean curves require complete groups. Interval endpoints use empirical inverse
CDF order statistics. When censoring prevents a coverage decision, the report
provides lower/upper coverage bounds and the unresolved count. No nominal
coverage guarantee is claimed for bootstrap percentile intervals.

## Learned-direction collaboration

See [the Astrid/Tanmoy experiment handoff](steering_uncertainty/steering_ood/LEARNED_DIRECTIONS_HANDOFF.md) for the agreed CIFAR10
learned-direction extension, proposed ownership, implementation checklist, matched
detector protocol, calibration isolation, source-transfer evaluation and required
artifacts. This is a specification; implementation and benchmark results are pending.
