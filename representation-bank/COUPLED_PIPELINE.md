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
```

Every frozen set and NPZ bank has a SHA-256 checksum. Loading verifies it.
Extraction rejects tokenizer drift, row-order drift, and missing adapter keys.
Right padding uses explicit sequence lengths; log probabilities are gathered
only over `answer_start:` with the causal one-token shift.

## Current pilot

The immutable TruthfulQA pilot contains 154 labeled generations from 44
questions (89 correct, 65 wrong), including 17 questions with both roles. Its
generation checksum is
`c076738b27178297bf8b8514ca9cfdb13197ea0a5939984cea739f9da772bfad`.

Mode-A/B remain disabled until their mode-specific frozen training sets exist.
Llama/Mistral caches remain an explicitly uncontrolled, separate arm.
