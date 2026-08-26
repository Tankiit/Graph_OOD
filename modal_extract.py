"""
modal_extract.py
────────────────
Hidden-state extraction for the gap-deferral paper.
4 models × 5 datasets on Modal cloud GPUs.

── GPU ceiling: max 2 GPUs at any time ──────────────────────────────────────
  llama3_8b  / mistral_7b / gemma2_9b  →  1× A100-40GB each (run one at a time)
  llama3_70b                            →  2× A100-80GB (single job, at ceiling)
  Models run SEQUENTIALLY. Never more than 1 job active at once.

── Design ────────────────────────────────────────────────────────────────────
  Each model is one Modal function at global scope (Modal requirement).
  All four functions share _run_model() for the extraction loop.
  datasets_loader.py is bundled into the image — identical loaders to local.
  Checkpoints every SAVE_EVERY questions for resumable runs.
  Partial step2-ready saves every PARTIAL_EVERY questions (mid-run inspection).

── Volume / path ─────────────────────────────────────────────────────────────
  Volume: deferral-llm-extract-v1
  Path:   /extract/{model_key}/{dataset}/hidden_states.pt

── Usage ─────────────────────────────────────────────────────────────────────
  modal run modal_extract.py --dry-run
  modal run modal_extract.py --model llama3_8b
  modal run modal_extract.py --small-models        # 8b → mistral → gemma, ~$12
  modal run modal_extract.py --all                 # all 4 sequential, ~$52
  modal run modal_extract.py --list-files
"""

import os
import time
import modal

# ── App ───────────────────────────────────────────────────────────────────────
app = modal.App("gap-demo-extraction")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.44.0",
        "accelerate>=0.28.0",
        "datasets>=2.18.0",
        "tqdm>=4.66.0",
        "numpy>=1.26.0",
        "huggingface_hub>=0.23.0",
        "scikit-learn>=1.4.0",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
    )
    .add_local_file("datasets_loader.py", "/root/datasets_loader.py")
)

hf_cache   = modal.Volume.from_name("hf-model-cache",         create_if_missing=True)
output_vol = modal.Volume.from_name("deferral-llm-extract-v1", create_if_missing=True)

SAVE_EVERY    = 100   # raw checkpoint every N questions (resume)
PARTIAL_EVERY = 500   # step2-ready partial save every N questions

# ── Model configs ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "llama3_8b": {
        "hf_name":        "meta-llama/Llama-3.1-8B-Instruct",
        "gpu":            "A100-40GB",
        "n_gpu":          1,
        "dtype":          "float16",
        "n_probe_layers": 8,
        "token_mode":     "first",
        "description":    "LLaMA-3.1-8B — 32 layers, 1× A100-40GB, ~$4",
    },
    "mistral_7b": {
        "hf_name":        "mistralai/Mistral-7B-Instruct-v0.3",
        "gpu":            "A100-40GB",
        "n_gpu":          1,
        "dtype":          "float16",
        "n_probe_layers": 8,
        "token_mode":     "first",
        "description":    "Mistral-7B-v0.3 — 32 layers, 1× A100-40GB, ~$4",
    },
    "gemma2_9b": {
        "hf_name":        "google/gemma-2-9b-it",
        "gpu":            "A100-40GB",
        "n_gpu":          1,
        "dtype":          "bfloat16",
        "n_probe_layers": 10,
        "token_mode":     "first",
        "description":    "Gemma-2-9B — 42 layers, 1× A100-40GB, ~$4",
    },
    "llama3_70b": {
        "hf_name":        "meta-llama/Llama-3.1-70B-Instruct",
        "gpu":            "A100-80GB",
        "n_gpu":          2,
        "dtype":          "float16",
        "n_probe_layers": 12,
        "token_mode":     "first",
        "description":    "LLaMA-3.1-70B — 80 layers, 2× A100-80GB, ~$40",
    },
}

ALL_DATASETS     = ["truthfulqa", "halueval_qa", "triviaqa", "popqa", "bioasq"]
EXPERT_THRESHOLD = 0.80


# ── Shared extraction utilities ───────────────────────────────────────────────

def _pool(hs_layers, a_s, a_e, token_mode):
    import torch
    ans_len = a_e - a_s
    vecs = []
    for hs in hs_layers:
        h = hs[0, a_s:a_e, :]
        if token_mode == "first" or ans_len == 1:
            v = h[0]
        elif token_mode == "early":
            v = h[:min(3, ans_len)].mean(0)
        else:
            v = h.mean(0)
        vecs.append(v)
    return torch.stack(vecs).mean(0).cpu().float()


def _extract_one(model, tokenizer, device, question, answer, n_layers, token_mode):
    import torch
    pfx   = f"Question: {question}\nAnswer: "
    fids  = tokenizer(pfx + answer, return_tensors="pt").input_ids
    pids  = tokenizer(pfx,          return_tensors="pt").input_ids
    a_s, a_e = pids.shape[1], fids.shape[1]
    if a_e <= a_s:
        return None, None
    fids = fids.to(device)
    with torch.no_grad():
        out = model(fids, output_hidden_states=True)
    h     = _pool(list(out.hidden_states[-n_layers:]), a_s, a_e, token_mode)
    lp    = torch.nn.functional.log_softmax(out.logits[0], dim=-1)
    aids  = fids[0, a_s:a_e]
    gen_lp = lp[a_s-1:a_e-1][torch.arange(len(aids)), aids].mean().item()
    return h, gen_lp


def _run_model(model_key: str, datasets: list):
    """
    Core extraction loop — called inside each global Modal function.
    Shared across all four models; model_key selects the config.
    """
    import sys
    sys.path.insert(0, "/root")
    from datasets_loader import load_dataset_records, DATASET_REGISTRY
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    cfg     = MODEL_CONFIGS[model_key]
    n_probe = cfg["n_probe_layers"]

    print(f"\n{'='*64}")
    print(f"  {model_key}  |  {cfg['hf_name']}")
    print(f"  GPU: {cfg['n_gpu']}× {cfg['gpu']}  |  "
          f"probe: last {n_probe} layers  |  token: {cfg['token_mode']}")
    print(f"  Datasets: {datasets}")
    print(f"{'='*64}\n")

    # Load model once — amortised across all datasets
    t0        = time.time()
    dtype     = torch.float16 if cfg["dtype"] == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_name"], torch_dtype=dtype, device_map="auto")
    model.eval()
    n_total = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    device  = next(model.parameters()).device
    print(f"  Loaded in {time.time()-t0:.0f}s | "
          f"{n_total} layers | d={d_model} | {device}")
    print(f"  Extracting layers {n_total-n_probe+1}–{n_total} "
          f"({(n_total-n_probe)/n_total*100:.0f}–100% depth)\n")

    for ds_name in datasets:
        out_dir  = f"/extract/{model_key}/{ds_name}"
        out_path = f"{out_dir}/hidden_states.pt"
        ckpt_dir = f"{out_dir}/checkpoints"
        os.makedirs(out_dir,  exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        # Skip completed datasets
        if os.path.exists(out_path):
            saved = torch.load(out_path, map_location="cpu", weights_only=False)
            print(f"\n  ✓ {ds_name}: already done "
                  f"({saved['n_questions']} questions) — skipping")
            del saved
            continue

        desc = DATASET_REGISTRY.get(ds_name, {}).get("description", ds_name)
        print(f"\n── {ds_name}  ({desc})")

        records = load_dataset_records(ds_name)
        if not records:
            print(f"  ✗ No records — skipping")
            continue

        # Resume from checkpoint
        ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
        if ckpts:
            ckpt      = torch.load(f"{ckpt_dir}/{ckpts[-1]}",
                                   map_location="cpu", weights_only=False)
            buf       = ckpt["results"]
            start_idx = ckpt["processed"]
            print(f"  Resuming from checkpoint at {start_idx} questions")
        else:
            buf       = {k: [] for k in [
                "h_correct", "h_wrong", "lp_correct", "lp_wrong",
                "questions", "correct_ans", "wrong_ans",
                "categories", "human_accs",
                "y_expert_correct", "y_expert_wrong",
            ]}
            start_idx = 0

        skipped   = 0
        remaining = records[start_idx:]

        for i, rec in enumerate(tqdm(remaining, desc=f"{model_key}/{ds_name}",
                                      initial=start_idx, total=len(records))):
            h_c, lp_c = _extract_one(model, tokenizer, device,
                                      rec["question"], rec["best_answer"],
                                      n_probe, cfg["token_mode"])
            h_w, lp_w = _extract_one(model, tokenizer, device,
                                      rec["question"], rec["wrong_answer"],
                                      n_probe, cfg["token_mode"])
            if h_c is None or h_w is None:
                skipped += 1
                continue

            buf["h_correct"].append(h_c)
            buf["h_wrong"].append(h_w)
            buf["lp_correct"].append(lp_c)
            buf["lp_wrong"].append(lp_w)
            buf["questions"].append(rec["question"])
            buf["correct_ans"].append(rec["best_answer"])
            buf["wrong_ans"].append(rec["wrong_answer"])
            buf["categories"].append(rec.get("category", "Unknown"))
            buf["human_accs"].append(float(rec.get("human_acc", 1.0)))
            buf["y_expert_correct"].append(int(rec["y_expert_correct"]))
            buf["y_expert_wrong"].append(int(rec["y_expert_wrong"]))

            processed = start_idx + i + 1

            # Raw checkpoint (for resume after crash)
            if (i + 1) % SAVE_EVERY == 0:
                torch.save({"results": buf, "processed": processed},
                           f"{ckpt_dir}/ckpt_{processed:05d}.pt")
                output_vol.commit()
                tqdm.write(f"  Checkpoint: {processed} questions")

            # Partial step2-ready save (downloadable mid-run)
            if (i + 1) % PARTIAL_EVERY == 0:
                n_so_far = len(buf["h_correct"])
                if n_so_far > 0:
                    ppath = f"{out_dir}/hidden_states_partial_{n_so_far:05d}.pt"
                    pdict = _build_save_dict(buf, cfg, model_key, ds_name,
                                             n_so_far, n_probe, n_total, d_model)
                    pdict["partial"] = True
                    torch.save(pdict, ppath)
                    output_vol.commit()
                    tqdm.write(f"  Partial save → {ppath} ({n_so_far}q, step2-ready)")

        N = len(buf["h_correct"])
        print(f"  Processed: {N}  Skipped: {skipped}")
        if N == 0:
            continue

        # Final save
        save_dict = _build_save_dict(buf, cfg, model_key, ds_name,
                                      N, n_probe, n_total, d_model)
        torch.save(save_dict, out_path)
        output_vol.commit()

        lp_c = save_dict["lp_correct"]
        lp_w = save_dict["lp_wrong"]
        pct  = (lp_c > lp_w).float().mean() * 100
        print(f"  ✓ Saved → {out_path}")
        print(f"    Shape:              {save_dict['h_correct'].shape}")
        print(f"    lp_correct > lp_wrong: {pct:.1f}%")

        # Remove checkpoints now that final is saved
        for f in os.listdir(ckpt_dir):
            os.remove(f"{ckpt_dir}/{f}")

    print(f"\n✓  All datasets complete for {model_key}")


def _build_save_dict(buf, cfg, model_key, ds_name, N, n_probe, n_total, d_model):
    """Build the canonical .pt dict that train_probe.py expects."""
    import torch
    return {
        "h_correct":        torch.stack(buf["h_correct"]),
        "h_wrong":          torch.stack(buf["h_wrong"]),
        "lp_correct":       torch.tensor(buf["lp_correct"]),
        "lp_wrong":         torch.tensor(buf["lp_wrong"]),
        "questions":        buf["questions"],
        "correct_ans":      buf["correct_ans"],
        "wrong_ans":        buf["wrong_ans"],
        "categories":       buf["categories"],
        "human_accs":       buf["human_accs"],
        "y_expert_correct": torch.tensor(buf["y_expert_correct"], dtype=torch.long),
        "y_expert_wrong":   torch.tensor(buf["y_expert_wrong"],   dtype=torch.long),
        "model":            cfg["hf_name"],
        "model_key":        model_key,
        "dataset":          ds_name,
        "n_questions":      N,
        "n_probe_layers":   n_probe,
        "n_total_layers":   n_total,
        "d_model":          d_model,
        "token_mode":       cfg["token_mode"],
        "expert_threshold": EXPERT_THRESHOLD,
    }


# ── Four Modal functions at global scope (Modal requirement) ──────────────────
# Each has its GPU spec hardcoded as a string — no factory, no class.
# modal.gpu.A100() is deprecated; use the "A100-40GB" / "A100-80GB" string directly.

_VOLUMES = {"/root/.cache/huggingface": hf_cache, "/extract": output_vol}
_SECRETS = [modal.Secret.from_name("huggingface")]

@app.function(image=image, gpu="A100-40GB", volumes=_VOLUMES,
              timeout=28800, memory=65536, secrets=_SECRETS)
def extract_llama3_8b(datasets: list = None):
    _run_model("llama3_8b", datasets or ALL_DATASETS)

@app.function(image=image, gpu="A100-40GB", volumes=_VOLUMES,
              timeout=28800, memory=65536, secrets=_SECRETS)
def extract_mistral_7b(datasets: list = None):
    _run_model("mistral_7b", datasets or ALL_DATASETS)

@app.function(image=image, gpu="A100-40GB", volumes=_VOLUMES,
              timeout=28800, memory=65536, secrets=_SECRETS)
def extract_gemma2_9b(datasets: list = None):
    _run_model("gemma2_9b", datasets or ALL_DATASETS)

@app.function(image=image, gpu="A100-80GB:2", volumes=_VOLUMES,
              timeout=28800, memory=131072, secrets=_SECRETS)
def extract_llama3_70b(datasets: list = None):
    _run_model("llama3_70b", datasets or ALL_DATASETS)


# Map model key → its Modal function
_FN_MAP = {
    "llama3_8b":  extract_llama3_8b,
    "mistral_7b": extract_mistral_7b,
    "gemma2_9b":  extract_gemma2_9b,
    "llama3_70b": extract_llama3_70b,
}


# ── Utility: list volume ──────────────────────────────────────────────────────

@app.function(image=image, volumes={"/extract": output_vol}, timeout=120)
def list_files():
    rows = []
    for root, dirs, files in os.walk("/extract"):
        dirs.sort()
        for fname in sorted(files):
            if not fname.endswith(".pt"):
                continue
            path = os.path.join(root, fname)
            mb   = os.path.getsize(path) / 1e6
            rows.append((path.replace("/extract/", ""), round(mb, 1)))
    return rows


# ── Entrypoint ────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    model:        str  = None,
    datasets:     str  = None,
    all:          bool = False,
    small_models: bool = False,
    dry_run:      bool = False,
    list_files_:  bool = False,
    gate_70b:     bool = True,
):
    """
    GPU ceiling: max 2 GPUs at any time. All models run sequentially.

    --model llama3_8b               single model
    --model llama3_8b,mistral_7b    two models sequentially
    --small-models                  llama3_8b → mistral_7b → gemma2_9b
    --all                           all 4, gate before 70B
    --datasets truthfulqa,halueval_qa   dataset subset
    --dry-run                       show run order and cost
    --list-files                    show volume contents
    --no-gate-70b                   skip key_claim check before 70B
    """
    ds_list = ([d.strip() for d in datasets.split(",")]
               if datasets else ALL_DATASETS)

    bad_ds = [d for d in ds_list if d not in ALL_DATASETS]
    if bad_ds:
        print(f"Unknown datasets: {bad_ds}. Valid: {ALL_DATASETS}")
        return

    # ── List files ────────────────────────────────────────────────────────────
    if list_files_:
        rows = list_files.remote()
        if not rows:
            print("Volume is empty.")
            return
        print(f"\n{'File':<65}  {'MB':>7}")
        print("  " + "─" * 74)
        for rel, mb in rows:
            print(f"  {rel:<63}  {mb:>7.1f}")
        print(f"\n  Total: {sum(mb for _,mb in rows):.0f} MB  "
              f"({len(rows)} files)")
        return

    # ── Determine models ──────────────────────────────────────────────────────
    if all:
        models_to_run = list(MODEL_CONFIGS.keys())
    elif small_models:
        models_to_run = ["llama3_8b", "mistral_7b", "gemma2_9b"]
    elif model:
        models_to_run = [m.strip() for m in model.split(",")]
        bad = [m for m in models_to_run if m not in MODEL_CONFIGS]
        if bad:
            print(f"Unknown models: {bad}. Valid: {list(MODEL_CONFIGS.keys())}")
            return
    else:
        print("\nOptions:")
        print("  modal run modal_extract.py --dry-run")
        print("  modal run modal_extract.py --model llama3_8b")
        print("  modal run modal_extract.py --small-models")
        print("  modal run modal_extract.py --all")
        print("  modal run modal_extract.py --list-files")
        return

    # ── Dry run ───────────────────────────────────────────────────────────────
    if dry_run:
        RATES = {"A100-40GB": 2.10, "A100-80GB": 2.50}
        n_ds  = len(ds_list)
        print(f"\nDRY RUN | datasets ({n_ds}): {ds_list}")
        print(f"Run order (sequential, max 2 GPUs at any time):")
        total = 0.0
        for i, m in enumerate(models_to_run):
            cfg  = MODEL_CONFIGS[m]
            rate = RATES[cfg["gpu"]] * cfg["n_gpu"]
            hrs  = (4.0 if "70b" in m else 2.0) * (n_ds / 5)
            cost = rate * hrs
            total += cost
            print(f"\n  [{i+1}/{len(models_to_run)}] {m}")
            print(f"    {cfg['description']}")
            print(f"    {cfg['n_gpu']}× {cfg['gpu']}  "
                  f"(probe: last {cfg['n_probe_layers']} layers)")
            print(f"    Est: {hrs:.1f}h × ${rate:.2f}/h = ~${cost:.0f}")
        print(f"\n  TOTAL: ~${total:.0f}  (sequential, never >2 GPUs)")
        print(f"\n  Volume: deferral-llm-extract-v1")
        print(f"  Path:   /extract/{{model_key}}/{{dataset}}/hidden_states.pt")
        return

    # ── Sequential execution ──────────────────────────────────────────────────
    small  = [m for m in models_to_run if m != "llama3_70b"]
    run70b = "llama3_70b" in models_to_run
    done   = []

    for m in small:
        cfg = MODEL_CONFIGS[m]
        print(f"\n{'─'*58}")
        print(f"  [{len(done)+1}/{len(models_to_run)}]  {m}")
        print(f"  {cfg['description']}")
        print(f"  Datasets: {ds_list}")
        print(f"{'─'*58}")
        _FN_MAP[m].remote(datasets=ds_list)
        done.append(m)
        print(f"  ✓ {m} complete  ({len(done)}/{len(models_to_run)})")

    if run70b:
        if gate_70b and small:
            print(f"\n{'='*58}")
            print(f"  Small models done. Verify key claim before llama3_70b (~$40).")
            print(f"{'='*58}")
            print(f"\n  Download 8B results:")
            for ds in ds_list:
                print(f"    modal volume get deferral-llm-extract-v1 "
                      f"/extract/llama3_8b/{ds}/hidden_states.pt "
                      f"./llama3_8b_{ds}.pt")
            print(f"\n  Run train_probe.py, check key_claim_holds, then:")
            print(f"    modal run modal_extract.py --model llama3_70b "
                  f"--no-gate-70b --datasets {','.join(ds_list)}")
        else:
            cfg = MODEL_CONFIGS["llama3_70b"]
            print(f"\n{'─'*58}")
            print(f"  [{len(done)+1}/{len(models_to_run)}]  llama3_70b")
            print(f"  {cfg['description']}  ← 2-GPU ceiling")
            print(f"  Datasets: {ds_list}")
            print(f"{'─'*58}")
            _FN_MAP["llama3_70b"].remote(datasets=ds_list)
            done.append("llama3_70b")
            print(f"  ✓ llama3_70b complete  ({len(done)}/{len(models_to_run)})")

    print(f"\n{'='*58}")
    print(f"  ✓  Done — {len(done)} model(s): {done}")
    print(f"  Datasets: {ds_list}")
    print(f"{'='*58}")
    print(f"\n  List files:  modal run modal_extract.py --list-files")
    print(f"  Download:    modal volume get deferral-llm-extract-v1 \\")
    print(f"    /extract/llama3_8b/truthfulqa/hidden_states.pt ./out.pt")