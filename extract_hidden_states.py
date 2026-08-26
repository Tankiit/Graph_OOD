"""
extract_hidden_states.py
────────────────────────
Local hidden-state extraction for the gap-deferral paper.
Runs on MPS (Apple Silicon), CUDA, or CPU.

Shares datasets_loader.py with modal_extract.py — identical loaders,
OVA labels, and output key names across all five datasets.

── Architecture ──────────────────────────────────────────────────────────────
  main()
    └── loads model + records
    └── run_layer_sweep()        optional, finds best probe layer
    └── run_extraction()         main loop → saves .pt file

── Output keys (identical to modal_extract.py) ───────────────────────────────
  h_correct, h_wrong       (N, d)  float32
  lp_correct, lp_wrong     (N,)    float32
  questions, correct_ans, wrong_ans, categories, human_accs
  y_expert_correct, y_expert_wrong  (N,) int64
  model, model_key, dataset         str
  n_questions, n_probe_layers,
  n_total_layers, d_model           int
  token_mode, expert_threshold      metadata
  sweep                             dict (layer→AUROC) if run

── Default probe layers by architecture depth ────────────────────────────────
  ≥ 80 layers (LLaMA-70B)  → last 12
  ≥ 42 layers (Gemma-2-9B) → last 10
  otherwise                 → last 8

── Usage ─────────────────────────────────────────────────────────────────────
  python extract_hidden_states.py
  python extract_hidden_states.py --dataset halueval_qa
  python extract_hidden_states.py --dataset truthfulqa --model meta-llama/Llama-3.1-8B-Instruct
  python extract_hidden_states.py --max-questions 100  # smoke test
  python extract_hidden_states.py --skip-sweep         # faster
  python extract_hidden_states.py --resume             # after interrupt
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Locate datasets_loader.py ─────────────────────────────────────────────────
_HERE = Path(__file__).parent
for _candidate in [_HERE, _HERE.parent / "gap_demo_modal"]:
    if (_candidate / "datasets_loader.py").exists():
        sys.path.insert(0, str(_candidate))
        break

try:
    from datasets_loader import load_dataset_records, DATASET_REGISTRY
except ImportError:
    raise ImportError(
        "Cannot find datasets_loader.py. "
        "Place it alongside this script or in gap_demo_modal/."
    )

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATASET = "truthfulqa"
OUTPUT_DIR      = Path("outputs")
SAVE_EVERY      = 50


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract hidden states for gap-deferral paper")
    p.add_argument("--model",   default=DEFAULT_MODEL,
                   help="HuggingFace model name")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   choices=sorted(DATASET_REGISTRY.keys()),
                   help=f"Dataset. Options: {sorted(DATASET_REGISTRY.keys())}")
    p.add_argument("--layers",  type=int, default=None,
                   help="Last N transformer layers to probe. "
                        "Default: 8 (or 10/12 for deeper models)")
    p.add_argument("--token-mode", default="first",
                   choices=["first", "mean", "early"],
                   help="Token pooling strategy (first = Orgad et al. ICLR 2025)")
    p.add_argument("--output",  default=None,
                   help="Output .pt path. Default: outputs/{dataset}_hidden_states.pt")
    p.add_argument("--max-questions", type=int, default=None,
                   help="Cap questions for a quick smoke test")
    p.add_argument("--skip-sweep",    action="store_true",
                   help="Skip layer sweep (~5 min)")
    p.add_argument("--resume",        action="store_true",
                   help="Resume from last checkpoint")
    return p.parse_args()


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        print("  Using MPS (Apple Silicon)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("  Using CUDA")
        return torch.device("cuda")
    print("  Using CPU")
    return torch.device("cpu")


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_name, device):
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True)
    if device.type == "mps":
        kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model = model.to(device).eval()

    n_layers = model.config.num_hidden_layers
    d_model  = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.0f}M params | {n_layers} layers | d={d_model}")
    return tokenizer, model, n_layers, d_model


def default_probe_layers(n_total_layers: int, layers_arg) -> int:
    """Principled default: last 25% of depth, min 8, max 12."""
    if layers_arg is not None:
        return layers_arg
    if n_total_layers >= 80:   # LLaMA-70B (80 layers)
        return 12
    if n_total_layers >= 42:   # Gemma-2-9B (42 layers)
        return 10
    return 8                   # LLaMA-8B, Mistral-7B (32 layers)


# ── Hidden state pooling ──────────────────────────────────────────────────────

def pool_hidden_state(hs_layers, ans_start, ans_end, token_mode):
    """Mean across last-N layers, pooled at answer tokens."""
    ans_len = ans_end - ans_start
    vecs = []
    for hs in hs_layers:
        h = hs[0, ans_start:ans_end, :]
        if token_mode == "first" or ans_len == 1:
            v = h[0]
        elif token_mode == "early":
            v = h[:min(3, ans_len)].mean(0)
        else:
            v = h.mean(0)
        vecs.append(v)
    return torch.stack(vecs).mean(0).cpu().float()


def extract_one(model, tokenizer, device, question, answer, n_layers, token_mode):
    """Forward pass → (h [d,], gen_logprob float) or (None, None)."""
    q_prefix  = f"Question: {question}\nAnswer: "
    full_text = q_prefix + answer
    full_ids  = tokenizer(full_text, return_tensors="pt").input_ids
    pfx_ids   = tokenizer(q_prefix,  return_tensors="pt").input_ids
    a_s, a_e  = pfx_ids.shape[1], full_ids.shape[1]
    if a_e <= a_s:
        return None, None

    full_ids = full_ids.to(device)
    with torch.no_grad():
        out = model(full_ids, output_hidden_states=True)

    h = pool_hidden_state(list(out.hidden_states[-n_layers:]), a_s, a_e, token_mode)

    logits    = out.logits[0]
    lp        = torch.nn.functional.log_softmax(logits, dim=-1)
    ans_ids   = full_ids[0, a_s:a_e]
    gen_lp    = lp[a_s-1:a_e-1][torch.arange(len(ans_ids)), ans_ids].mean().item()

    return h, gen_lp


# ── Layer sweep ───────────────────────────────────────────────────────────────

def run_layer_sweep(model, tokenizer, device, records, n_total_layers,
                    token_mode, n_sweep=50):
    """One forward pass per question, probe each layer separately."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    print(f"\nRunning layer sweep on {n_sweep} questions ({n_total_layers} layers)...")
    sub = records[:n_sweep]

    all_hs = {li: [] for li in range(n_total_layers + 1)}
    y      = []

    for rec in tqdm(sub, desc="Layer sweep"):
        q     = rec["question"]
        c_ans = rec.get("best_answer", "")
        w_ans = rec.get("wrong_answer", "")
        if not c_ans or not w_ans:
            continue
        for ans, label in [(c_ans, 1), (w_ans, 0)]:
            pfx   = f"Question: {q}\nAnswer: "
            fids  = tokenizer(pfx + ans, return_tensors="pt").input_ids.to(device)
            pids  = tokenizer(pfx, return_tensors="pt").input_ids
            a_s, a_e = pids.shape[1], fids.shape[1]
            if a_e <= a_s:
                continue
            with torch.no_grad():
                out = model(fids, output_hidden_states=True)
            for li, hs in enumerate(out.hidden_states):
                v = pool_hidden_state([hs], a_s, a_e, token_mode)
                all_hs[li].append(v.numpy())
            y.append(label)

    y_arr = np.array(y)
    layer_aurocs = {}
    for li in tqdm(range(n_total_layers + 1), desc="Probing layers"):
        vecs = all_hs[li]
        if len(vecs) < 10 or len(vecs) != len(y_arr):
            continue
        X = np.stack(vecs)
        sc = StandardScaler()
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        scores = cross_val_score(clf, sc.fit_transform(X), y_arr,
                                 cv=5, scoring="roc_auc")
        layer_aurocs[li] = float(scores.mean())

    if not layer_aurocs:
        print("  Sweep failed — no valid layers")
        return {}, 0

    best = max(layer_aurocs, key=layer_aurocs.get)
    print(f"\n  Best layer: {best}/{n_total_layers} "
          f"({best/n_total_layers*100:.0f}% depth) — AUROC={layer_aurocs[best]:.4f}")
    print(f"  Bottom 5: {sorted(layer_aurocs, key=layer_aurocs.get)[:5]}")
    return layer_aurocs, best


# ── Checkpointing ─────────────────────────────────────────────────────────────

def _empty():
    return {k: [] for k in [
        "h_correct", "h_wrong", "lp_correct", "lp_wrong",
        "questions", "correct_ans", "wrong_ans",
        "categories", "human_accs", "y_expert_correct", "y_expert_wrong",
    ]}


def save_checkpoint(results, idx, ckpt_dir):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"results": results, "processed": idx},
               ckpt_dir / f"checkpoint_{idx:05d}.pt")


def load_latest_checkpoint(ckpt_dir):
    cps = sorted(ckpt_dir.glob("checkpoint_*.pt")) if ckpt_dir.exists() else []
    if not cps:
        return None, 0
    data = torch.load(cps[-1], map_location="cpu", weights_only=False)
    print(f"  Resuming from checkpoint at {data['processed']} questions")
    return data["results"], data["processed"]


# ── Extraction ────────────────────────────────────────────────────────────────

def run_extraction(args, records, *, device, tokenizer, model,
                   n_total_layers, d_model, n_probe_layers, sweep_results):
    """
    Main extraction loop.  Reads canonical records from datasets_loader
    (fields: question, best_answer, wrong_answer, category, human_acc,
     y_expert_correct, y_expert_wrong) and writes a .pt file whose key
    schema is identical to modal_extract.py.
    """
    out_path = (Path(args.output) if args.output
                else OUTPUT_DIR / f"{args.dataset}_hidden_states.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUTPUT_DIR / "checkpoints" / args.dataset

    if args.max_questions:
        records = records[:args.max_questions]
        print(f"  Capped at {len(records)} questions")

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        results, start_idx = load_latest_checkpoint(ckpt_dir)
        if results is None:
            start_idx, results = 0, _empty()
    else:
        start_idx, results = 0, _empty()

    remaining = records[start_idx:]
    print(f"\nExtracting ({len(remaining)} questions remaining)...")

    skipped = processed = 0

    for i, rec in enumerate(tqdm(remaining, desc="Questions",
                                  initial=start_idx, total=len(records))):
        q     = rec["question"]
        c_ans = rec.get("best_answer", "")
        w_ans = rec.get("wrong_answer", "")   # canonical string field
        if not c_ans or not w_ans:
            skipped += 1
            continue

        h_c, lp_c = extract_one(model, tokenizer, device, q, c_ans,
                                  n_probe_layers, args.token_mode)
        h_w, lp_w = extract_one(model, tokenizer, device, q, w_ans,
                                  n_probe_layers, args.token_mode)
        if h_c is None or h_w is None:
            skipped += 1
            continue

        results["h_correct"].append(h_c)
        results["h_wrong"].append(h_w)
        results["lp_correct"].append(lp_c)
        results["lp_wrong"].append(lp_w)
        results["questions"].append(q)
        results["correct_ans"].append(c_ans)
        results["wrong_ans"].append(w_ans)
        results["categories"].append(rec.get("category", "Unknown"))
        results["human_accs"].append(float(rec.get("human_acc", 1.0)))
        results["y_expert_correct"].append(int(rec.get("y_expert_correct", 1)))
        results["y_expert_wrong"].append(int(rec.get("y_expert_wrong", 1)))
        processed += 1

        if (i + 1) % SAVE_EVERY == 0:
            save_checkpoint(results, start_idx + i + 1, ckpt_dir)
            tqdm.write(f"  Checkpoint at {start_idx + i + 1} questions")

    N = len(results["h_correct"])
    print(f"\n  Done — processed {N}, skipped {skipped}")
    if N == 0:
        print("Nothing to save.")
        return

    # ── Save — canonical key schema ───────────────────────────────────────────
    save_dict = {
        "h_correct":        torch.stack(results["h_correct"]),
        "h_wrong":          torch.stack(results["h_wrong"]),
        "lp_correct":       torch.tensor(results["lp_correct"]),
        "lp_wrong":         torch.tensor(results["lp_wrong"]),
        "questions":        results["questions"],
        "correct_ans":      results["correct_ans"],
        "wrong_ans":        results["wrong_ans"],
        "categories":       results["categories"],
        "human_accs":       results["human_accs"],
        "y_expert_correct": torch.tensor(results["y_expert_correct"],
                                         dtype=torch.long),
        "y_expert_wrong":   torch.tensor(results["y_expert_wrong"],
                                         dtype=torch.long),
        # ── Metadata — all keys step2 / modal_extract expect ──────────────────
        "model":            args.model,
        "model_key":        args.model.split("/")[-1].replace("-","_").lower(),
        "dataset":          args.dataset,        # always present (HaluEval OVA fix)
        "n_questions":      N,
        "n_probe_layers":   n_probe_layers,      # matches modal_extract.py key
        "n_total_layers":   n_total_layers,
        "d_model":          d_model,
        "token_mode":       args.token_mode,
        "expert_threshold": 0.80,
        "sweep":            sweep_results,
    }
    torch.save(save_dict, out_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    lp_c  = save_dict["lp_correct"]
    lp_w  = save_dict["lp_wrong"]
    n_exp = int(save_dict["y_expert_correct"].sum())
    pct   = (lp_c > lp_w).float().mean() * 100

    print(f"\nSaved → {out_path}")
    print(f"  Shape:                {save_dict['h_correct'].shape}")
    print(f"  lp_correct > lp_wrong: {pct:.1f}%")
    print(f"  Gap mean ± std:        "
          f"{(lp_c-lp_w).mean():.3f} ± {(lp_c-lp_w).std():.3f}")
    print(f"  y_expert=1:            {n_exp}/{N}")
    if sweep_results:
        bl = sweep_results["best_layer"]
        print(f"  Best probe layer:      {bl}/{n_total_layers} "
              f"(AUROC={sweep_results['layer_aurocs'][bl]:.4f})")
    print(f"\n✓  Done. Run:")
    print(f"   python train_probe.py --input {out_path} "
          f"--output outputs/results_{args.dataset}.json")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Gap-deferral — hidden state extraction")
    print(f"  Model:    {args.model}")
    print(f"  Dataset:  {args.dataset}  "
          f"({DATASET_REGISTRY[args.dataset]['description']})")
    print(f"  Token:    {args.token_mode}")
    print(f"{'='*60}")

    tokenizer, model, n_total_layers, d_model = load_model(args.model, device)
    n_probe_layers = default_probe_layers(n_total_layers, args.layers)
    print(f"  Probing last {n_probe_layers} of {n_total_layers} layers "
          f"({(n_total_layers-n_probe_layers)/n_total_layers*100:.0f}–100% depth)")

    print(f"\nLoading {args.dataset} via datasets_loader...")
    records = load_dataset_records(args.dataset)

    # ── Layer sweep ───────────────────────────────────────────────────────────
    sweep_results = {}
    if not args.skip_sweep:
        layer_aurocs, best_layer = run_layer_sweep(
            model, tokenizer, device, records,
            n_total_layers, args.token_mode,
            n_sweep=min(50, len(records)))
        sweep_results = {"layer_aurocs": layer_aurocs,
                         "best_layer":   best_layer,
                         "n_total_layers": n_total_layers}
        sweep_path = OUTPUT_DIR / f"{args.dataset}_layer_sweep.pt"
        torch.save(sweep_results, sweep_path)
        print(f"  Sweep saved → {sweep_path}")
    else:
        print("  Layer sweep skipped (--skip-sweep)")

    run_extraction(
        args, records,
        device=device, tokenizer=tokenizer, model=model,
        n_total_layers=n_total_layers, d_model=d_model,
        n_probe_layers=n_probe_layers,
        sweep_results=sweep_results,
    )


if __name__ == "__main__":
    main()