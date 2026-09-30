# Validation of this handoff

Executed in an isolated Python 3.12 CPU environment. The original
`Graph_OOD_steering` checkout remained clean and unchanged.

- **11 tests passed**, including actual PyTorch-OOD and Skorch integration,
  Sentence Transformer local-model save/load/encoding, split-leakage rejection,
  PCA fitting isolation, exact crossings, censoring/re-entry and crossed ANOVA.
- One deprecation warning comes from the Sentence Transformers import alias used
  by the tiny local encoding test. It did not affect the test result.
- **E0 passed** with a maximum analytical crossing error of approximately
  4.61e-11 and doubled-grid difference of approximately 2.33e-11 in the Gaussian
  control. This precision does not transfer to arbitrary NLP or graph paths.
- The independent ID rejection in that one Gaussian calibration realization was
  0.0595 for target 0.05. This is reported rather than hidden; the calibration
  guarantee is marginal and does not fix each realized calibration threshold.
- `scripts/smoke.sh` completed: synthetic cache, Skorch linear-head training,
  four real PyTorch-OOD adapters, static evaluation, 2x2 crossed resampling,
  separate calibration resampling, and a 3x3 fresh-draw Gaussian E2 pilot.
- The official CLINC150 file was retrieved and its partitioner executed.
  Ten duplicate normalized-text records were excluded; the exact removals,
  source SHA-256 and final counts are in `examples/validation/clinc_split_audit.json`.
- PyTorch-OOD 0.3.3 kNN scores agreed with the sklearn kth-neighbor reference;
  energy agreed with negative temperature-scaled log-sum-exp of raw head logits.
  Frozen-head energy/MSP had zero reference contribution in the tested crossed design.

## Not performed

The full MPNet model was not downloaded or evaluated on CLINC150. No NLP
benchmark result, semantic-direction validity, detection improvement, scientific
confidence interval, or paper-level conclusion is claimed. Included example
AUROCs and crossing measurements are synthetic integration outputs.

The code provides the first working E0/E2 implementation. E4 text-edit evaluation,
near-tangent stress sweeps, matched-cost local-derivative comparisons, and a full
statistical inference procedure remain future experiments, as described in README.

`requirements-tested.txt` records exercised versions. Raw test output and E0 JSON
are included in `examples/validation/`.
