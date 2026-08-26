# Representation bank + intervention harness

This repository implements the shared artefact in the spec: Tinker LoRA adapters,
local Hugging Face residual extraction, an append-only Zarr bank, registered
direction quantiles, and forward-pass interventions (`add`, `project_out`,
`clamp`, `scale`).

## Important design choices

- Tinker is used for LoRA training and prefill scoring; hidden states are always
  extracted locally from the matching Hugging Face model or exported adapter.
- Depth is addressed by fractions. The cache stores the fractions, resolved
  layer count, hidden width, and the exact model id.
- `h_last` contains the embedding output plus every transformer-block output,
  hence shape `(L+1,d)`. Span states use the resolved block outputs at
  fractions `0.2/0.5/0.8` and are padded to 32 tokens with a mask.
- Rows are never collapsed into pairwise differences. Pairing is metadata.
- Interventions are hooks on block outputs during the forward pass and therefore
  propagate into every downstream layer.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[tinker,test]'
export TINKER_API_KEY='...'
export TINKER_PROJECT_ID='...'  # optional billing/project attribution
export HF_TOKEN='...'  # if the Qwen weights require authenticated access
```

## Execute the staged plan

### 1. Kill switch: prompt log-probabilities

Use the base model and inspect the returned array: position zero may be `null`,
but later positions must be finite and its length must equal the prompt token
count.

```bash
repbank smoke-logprobs --model Qwen/Qwen3.5-9B-Base
```

### 2. Hugging Face availability

```bash
huggingface-cli download Qwen/Qwen3.5-9B-Base --dry-run
huggingface-cli download Qwen/Qwen3.5-9B --dry-run
```

### 3. Base prompting pilot

Create a JSONL manifest with `pair_id`, `role`, `label`, `label_protocol`, and
`prompt`. Put the few-shot prompt itself in `prompt`; this keeps base and
instruct protocols explicit rather than silently applying a chat template.
Run 20--50 rows first and compare answer parsing, length, and BLEURT label rates.

### 4. Throwaway adapter round-trip

```bash
repbank train-adapter --model Qwen/Qwen3.5-9B-Base --rank 8 \
  --role true --data examples/adapter_data.jsonl --epochs 1 --batch-size 32

repbank export-adapter --tinker-path 'tinker://...' \
  --base-model Qwen/Qwen3.5-9B-Base --output artifacts/roundtrip
```

The default export is a lightweight PEFT adapter. Add `--merge` only if a
standalone merged HF model is actually required. Load the default output with
`AutoPeftModelForCausalLM.from_pretrained(...)`; the merged output loads with
`AutoModelForCausalLM`.

### 5. Qwen3.5-9B × TruthfulQA extraction

Edit `configs/pilot.yaml`, then:

```bash
repbank extract --config configs/pilot.yaml --manifest data/truthfulqa.jsonl
```

Generation is followed by one local teacher-forced forward pass. This is
deliberate: it captures all residual states and exact generated-token
log-probabilities consistently. The first completion token is scored from the
last prompt logit.

### 6. Direction quantiles and intervention

Store a unit-compatible direction as NumPy shape `(d,)` and register its
empirical coordinate distribution:

```bash
repbank register-direction --cache artifacts/qwen35-9b_truthfulqa.zarr \
  --name truth-caa --direction directions/truth-caa.npy
```

Use interventions in generation code:

```python
from repbank.interventions import Intervention, InterventionHarness

spec = Intervention("add", direction, depth_fraction=0.8,
                    token_policy="all", strength=1.5)
with InterventionHarness(model, [spec]):
    output = model.generate(**inputs, max_new_tokens=200)
```

For generation-time `last`, the hook edits the newest token at every decode
step. For a prompt-only span, pass explicit token indices and perform the
prefill/generation call under the harness.

## Adapter ladder

Run `{true,hal} × {8,16,32}` per model. The trainer accepts JSONL rows with a
`role` and a standard `messages` array. Each final sampler checkpoint receives a
seven-day TTL by default. The code intentionally does not assume that the
likelihood ratio is meaningful for instruct models; use the Base checkpoints
for the paper's ratio.

The Qwen renderer is selected by model generation. If a cookbook release uses a
different registered name, pass it explicitly with `--renderer`; this is kept as
a CLI choice because chat rendering changes the actual training data.

`--data` also accepts a Hugging Face dataset ID. The selected split must have a
standard `messages` column containing `{role, content}` objects. A `role` column
is optional: when present it is filtered using `--role`; when absent, the whole
dataset is treated as the already-selected corpus for that role.

```bash
pip install -e '.[tinker,hf,test]'
repbank train-adapter --model Qwen/Qwen3.5-9B-Base --rank 8 --role true \
  --data organization/chat-dataset --split train --epochs 1 --batch-size 32
```

Use `--dataset-config`, `--revision`, `--messages-column`, or `--role-column`
for datasets with subsets, pinned revisions, or different column names.
Add `--max-samples 32 --validate-only` to stream and validate a small sample
without creating a Tinker job.

## What remains protocol-specific

The repository does not invent HaloScope preprocessing or a BLEURT decision
rule. Convert the exact frozen HaloScope examples into the documented manifest
and write the resulting score/threshold into `label`/`label_protocol`. This is
the correct boundary: model mechanics are shared, benchmark authorship and
labeling remain auditable inputs.
