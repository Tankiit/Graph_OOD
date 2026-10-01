# V1 manuscript audit

Date: 2026-10-01. Source: `steering_ood_v1.tex` (598 lines).

## Scope and status

This is a single-agent, source-based audit informed by the academic-paper-reviewer skill in `~/.claude`, not a completed five-seat review panel or a certified pipeline output. Calibration status: NOT_CALIBRATED. No manuscript edits were made. Mathematical counterexamples were checked numerically with Python; dependency checks and selected implementation comparisons were performed. Experimental numbers were not independently regenerated, citations were not externally verified, and no PDF compilation or visual inspection was completed. A TeX compiler was not found on PATH.

The draft is materially more developed than `steering_ood.tex`: it adds substantive introductory framing, related work, a measurement protocol, six research questions, and implementation settings. Its source still has major mathematical and reproducibility problems. It is not ready for submission. The user clarified that AISTATS is the intended venue, consistent with the source's AISTATS 2027 label, and wants a general theory framing. Venue-specific criteria binding remains unavailable; no current venue policy or formatting compliance is asserted.

## Findings

### 1. Major: the cone theorem's necessity claim is false (lines 256–259)

The theorem states that energy eventually rejects iff `Wv < 0` componentwise. This condition characterizes divergence to positive infinity, not all eventual rejection at a finite threshold.

Counterexample: two classes, `Wv=(-1,0)`, `Wz+b=(0,0)`, threshold `t=-0.5`. Along the ray, `E(alpha)=-log(1+exp(-alpha))`. The starting point is accepted (`E(0)=-0.6931`), but `E(1)=-0.3133 > t`, despite the second directional slope being zero. Thus eventual rejection occurs outside the strict cone.

Correction: let `d=Wv`, `q=Wz+b` and `d_max=max(d)`. If `d_max<0`, energy diverges to positive infinity. If `d_max>0`, it diverges to negative infinity (finite rejection intervals remain possible). If `d_max=0`, it approaches `L=-log(sum(exp(q_c) for d_c=0))` from below, or is constant if all slopes are zero. Eventual strict rejection in this boundary case holds exactly when `L>t`. Retain Gordan's alternative for the existence of directions producing positive-infinite energy, with that narrower statement.

### 2. Major: shrinkage equivalence fails at equality (line 232)

“Rejected near the origin iff `E(0)>t`” is false when `E(0)=t`: the origin itself can be accepted while all sufficiently small positive radial parameters are rejected.

Counterexample: `a=(-3,3)`, `b=(2,0)`, and `t=E(0)=-2.1269280110`. The starting probe at `s=1` is accepted (`E(1)=-3.0181499279`), but `E(0.001)=-2.1246451213 > t`. There is rejection arbitrarily near the accepted origin.

Correction: distinguish the decision exactly at the origin from the decision in a punctured neighborhood. State the equivalence under a strict nonzero origin margin, or explicitly analyze equality. Also complete the omitted `max(a)=0` growth case in part (a).

### 3. Major: estimated first crossing is presented as an exact infimum (lines 170–175)

The manuscript says uniform-grid evaluation followed by bisection gives the continuous first rejection. It can miss an entire rejection interval between grid points; bisection cannot recover intervals that were never sampled. Even a sampled bracket need not contain the first continuous crossing.

The implementation already acknowledges this limitation in `steering_ood/paths.py`, in the `trace_path` docstring: grids can miss excursions and bisection does not exclude earlier excursions. The appendix specifies only 101 grid points and 15 bisection steps.

Correction: denote an estimated first observed crossing, preserve the censoring indicator, and report grid sensitivity or score-specific certified crossing calculations. Separate measurement of `R(alpha)` from a claim that every continuously rejected path has been discovered.

### 4. Major: the source lacks required local build dependencies

Neither `aistats2027.sty` nor `aistats_fallback.sty` is present beside the source. `references.bib` is also absent there; no `.bib` or `.sty` was found under the project during the file search.

Five direct figure includes are missing: `figs/fig_polar.pdf`, `figs/fig_teaser.pdf`, `figs/fig_failure_map.pdf`, `figs/fig_theory.pdf`, and `figs/fig_realise.pdf`. The existing polar PDF is instead under `figures/fig_polar.pdf`. Five optional figure calls also have no corresponding file under their expected `figures/` directory: gauge, curves_seeds, support, mechanism, and real_ood. These optional calls render placeholders, so successful compilation alone would not establish a complete figure package.

Correction: supply the style, bibliography and actual figure assets; unify figure directories; then compile and visually inspect. The single-column theory figure at line 330 also uses `textwidth`, which should be checked against the column width.

### 5. Major: reference coverage cannot be validated (line 85 and throughout)

The energy reference is an empty `citep{}`. Keys for apparently identical baseline methods differ between Related Work and Measurement, for example `sun2022out` versus `sun2022knn`, `lee2018simple` versus `lee2018mahalanobis`, and baseline keys with 2017 versus 2018. These differences may be aliases, but the missing bibliography prevents resolution. Gordan's alternative still has a citation placeholder.

Correction: supply the bibliography, resolve every key, and verify that each source supports its associated claim. Do not infer fabricated references from missing keys; existence and attribution remain unverified here.

### 6. Major: Mahalanobis definition omits the actual covariance estimator (lines 127–132)

The text describes the inverse pooled within-class covariance, emphasizing double precision. The `ShrinkageMahalanobis.fit` implementation instead fits `LedoitWolf(assume_centered=True)` to within-class residuals and uses its shrinkage covariance. `THEORY.md` explicitly assumes a positive-definite float64 Ledoit–Wolf matrix. Shrinkage changes the estimator and potentially the rejection geometry; it is not merely a precision correction. With 1,000 vision references and 2,048 ResNet-50 dimensions, the unregularized residual covariance cannot be full rank.

Correction: state the shrinkage estimator, target and fitted coefficient, and identify this as the implemented baseline. Separate numerical precision effects from regularization effects before attributing differences to another implementation's float32 inversion. The numerical appendix is still a placeholder, so the order-of-magnitude allegation is not documented in this manuscript.

### 7. Major: natural transformations have not established path realization (RQ5, lines 369–382)

Decomposing natural feature displacement into radial and angular components shows shared geometric tendencies, not realization of a specific declared path. The reduction lemma requires pointwise equality along the path, or the appendix's uniform epsilon-tube bound and adequate score margins. Neither is reported for the natural transformations.

Correction: present these results as qualitative consistency with the component analysis. To claim realization, estimate a monotone path correspondence, uniform residual and score-margin guarantees. Also verify the universal “every transformation shrinks” statement against per-encoder trajectories; aggregate figure means alone cannot support it.

### 8. Moderate: the scale-invariance remark overstates its conclusion (lines 247–249)

`S(sz)=S(z)` implies an unchanged decision for positive `s`. It does not imply the score “rejects no radial path”: an initially rejected probe remains rejected throughout.

Correction: say that positive radial scaling induces no new rejection for initially accepted probes, and leave the undefined normalized origin outside the claim.

### 9. Moderate: calibration and uncertainty claims need narrower interpretation (lines 135–139, 180–189)

A common conformal target bounds marginal rejection under exchangeability; it does not make every detector's achieved false-positive rate identical, especially with ties. The generic order statistic also needs the `k=m+1` convention (positive infinity), although the stated pool sizes and target avoid that case.

Hierarchical resampling of four selected encoders and three seeds describes uncertainty under that resampling scheme, not automatic generalization to arbitrary encoders. Probe bootstrap inference should specify how shared reference fits, directions and signs are aggregated and which randomness is conditioned on. The one-sided test statistic, null construction, bootstrap count and treatment of multiple comparisons are absent.

Correction: use “common target false-alarm budget,” report achieved ID rates per detector, document the statistical algorithm and estimand, and label exploratory significance counts accordingly.

### 10. Major: evidence and manuscript completion remain outstanding

Limitations, conclusion, AI-use statement, proofs, numerical appendix, head accuracies and some endpoint checks contain explicit placeholders. The support analysis still requests verification at the current matched horizon. `THEORY.md` contains older horizons (about 1.03 text / 0.81 vision) and says the gauge sweep is outstanding, whereas v1 reports 1.04 / 0.75 and describes a completed sweep. This is provenance drift to resolve, not proof the new results are incorrect.

Correction: bind each figure and numerical claim to exact run IDs, seeds, aggregation scripts and outputs. Recheck the support analysis at the stated horizon and trace new gauge and transformation results to saved artifacts. The abstract's broad radial-energy wording should retain the theorem's conditions and distinguish finite-horizon observations from universal eventual behavior.

## Author's clarified framing

Frame the contribution as a general theory of calibrated rejection geometry along declared representation paths. Separate three layers: properties shared by calibrated scores (order-statistic invariance, pathwise decisions and feature-to-input transfer); consequences under explicit structural assumptions (concavity of affine-logit energy, coercivity of distance scores, positive-scale invariance); and detector-specific formulas and empirical checks.

A possible title is **A Theory of Calibrated Rejection Along Representation Paths**. A defensible central question is: which structural properties determine whether a calibrated detector eventually rejects a path, rejects only over a bounded interval, or remains accepting?

The current results establish several useful cases, rather than a general classification for arbitrary scores and paths. Generalizing the framing requires clearly stating the covered score classes, path classes, nondegeneracy assumptions and boundary cases. Present realization guarantees as conditional; do not turn geometric similarity of natural transformations into a theorem hypothesis already verified.

The existing counterexamples and measurement limitations remain relevant under this framing. The correction to the cone theorem is particularly central because it separates divergence of the score from eventual rejection at a calibrated finite threshold.

## What “structural properties” means here

A structural property is a mathematical feature of a score that constrains its behavior along a declared path. More precise wording for the paper is **mathematical properties of the score along a path**. These properties constrain possible rejection behavior; the calibrated threshold determines which parts of the path are actually rejected.

- **Concavity:** For a linear head, energy along a straight ray is concave. Its strict rejection set along that ray is therefore a single interval, possibly empty or unbounded. Concavity alone does not guarantee that any rejection occurs.
- **Growth at large distances:** With a fixed finite reference set, Euclidean kNN grows without bound under radial growth from a nonzero probe. Class-conditional Mahalanobis with positive-definite precision does likewise. Each must eventually exceed any finite threshold on an unbounded growth path; a finite experimental horizon need not reach that crossing.
- **Scale invariance:** A score depending only on feature direction is unchanged by positive radial scaling. An initially accepted probe remains accepted on that radial path, while an initially rejected probe remains rejected. Normalization is undefined at the origin unless separately specified.
- **Dependence on a shared logit component:** Adding the same feature-dependent linear term to every class logit preserves softmax probabilities and predictions but changes energy. Recalibrating the resulting energy score can therefore change rejection along a path even though the predictor remains identical.

### Suggested paragraph for the manuscript

We study how mathematical properties of a detector's score constrain calibrated rejection along declared representation paths. Concavity of linear-head energy restricts its rejection set on a straight ray to a single interval, whereas divergence of Euclidean kNN and positive-definite Mahalanobis scores guarantees eventual rejection along unbounded radial growth from a nonzero probe. Scores that depend only on feature direction retain their decisions under positive radial scaling. Energy additionally depends on a shared component of the class logits that leaves predictions unchanged but can alter calibrated rejection. These properties characterize possible responses to an intervention; the calibrated threshold determines where rejection occurs, and a finite measurement horizon determines which responses are observed.

## Checks that passed

- The source's referenced labels all have definitions.
- Gauge algebra `E_r(z)=E(z)-r^T z` is consistent with prediction and softmax invariance.
- The energy derivatives and concavity calculation are correct.
- The origin formulas for energy, MSP, kNN and Mahalanobis are structurally correct with the stated detector assumptions.
- The general kNN cosine condition uses the correct non-strict inequality for acceptance. Later text should also use `c_k <= 1/2` in the equal-norm special case, rather than `<`.

These checks establish local mathematical consistency only, not independent replication of the reported 24/24 empirical matches. This report supplies corrections for author consideration; it does not authorize manuscript revisions or certify venue compliance.
