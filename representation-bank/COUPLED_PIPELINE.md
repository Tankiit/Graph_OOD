# Coupled generation and extraction

This pipeline freezes generated text and Qwen3.5-family token IDs once, then
teacher-forces those IDs through every base/adapter model. Extraction never
samples. Hidden states and answer-token log probabilities therefore share one
forward pass and one row order.

## Reproduce

```bash
python scripts/freeze_generation_set.py \
  --input artifacts/manifests/truthfulqa_base_labeled.jsonl \
  --output artifacts/frozen/truthfulqa_qwen35_family.json \
  --adapter-output artifacts/frozen/truthfulqa_adapter_data.jsonl

python scripts/run_adapter_bank.py                 # inspect plan
python scripts/run_adapter_bank.py --only M_true M_hal --execute
python scripts/export_adapter_bank.py --only M_true M_hal --raw-only

modal run modal_coupled_extract.py --target primary
modal run modal_coupled_extract.py --target ladder-1
modal run modal_coupled_extract.py --target M_true
modal run modal_coupled_extract.py --target M_hal

python scripts/analyze_gates.py --bank artifacts/coupled/primary.npz
python scripts/analyze_gates.py --base artifacts/coupled/primary.npz \
  --adapter artifacts/coupled/ladder-1.npz
python scripts/analyze_gates.py --m-true artifacts/coupled/M_true.npz \
  --m-hal artifacts/coupled/M_hal.npz

# Gradient-enabled pass: fitted readout from primary.npz, VJPs/JVPs on frozen IDs.
modal run modal_jacobian_probe.py --max-pairs 8 --jvp-examples 3
```

Every frozen set and NPZ bank has a SHA-256 checksum. Loading verifies it.
Extraction rejects tokenizer drift, row-order drift, and missing adapter keys.
Right padding uses explicit sequence lengths; log probabilities are gathered
only over `answer_start:` with the causal one-token shift.

The Jacobian probe is intentionally separate from inference-mode extraction. It
fits a standardized low-rank paired density-ratio head from extracted `h_last`,
then replays the same rows with frozen parameters and activation gradients. It
stores the sample-dependent field `J(h)^T u(h)`, the fixed-readout field
`J(h)^T mean(u)`, and finite-difference JVPs over an epsilon ladder.

## Current pilot

The immutable TruthfulQA pilot contains 154 labeled generations from 44
questions (89 correct, 65 wrong), including 17 questions with both roles. Its
generation checksum is
`c076738b27178297bf8b8514ca9cfdb13197ea0a5939984cea739f9da772bfad`.

Mode-A/B remain disabled until their mode-specific frozen training sets exist.
Llama/Mistral caches remain an explicitly uncontrolled, separate arm.

The expanded TruthfulQA v2 set is a new immutable version rather than an
overwrite of the pilot. Run its ladder with:

```bash
python scripts/run_adapter_bank.py --config configs/adapter_bank_v2.yaml \
  --manifest artifacts/adapter_bank_v2.json --execute
modal run modal_coupled_extract.py --target primary-v2
```

The current v2 run contains 743 labeled generations from 181 questions, with
61 paired questions. At depth 0.8 the primary model reaches question-grouped
held-out EU AUC 0.725. A strict nested evaluation (outer grouped five-fold,
inner grouped four-fold for the EU meta-feature) adds 0.076 AUC over answer
confidence. The rank 1/8/16/32 ladder is deliberately
reported as a flat result: held-out EU AUC is 0.719--0.723 and strictly nested
incremental AUC is 0.0534--0.0539 across ranks. Nominal LoRA rank did not control effective
representation dimensionality under this one-epoch protocol.

Rank-curve cosines are descriptive quantities in the 64-dimensional primary
PCA coordinate system, fitted on the full bank. Their isotropic null SD is
`1/sqrt(64) = 0.125`; values around -0.08 do not establish anti-alignment.
