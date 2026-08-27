from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .cache import CacheRow, RepresentationBank
from .config import RunConfig


def _dtype(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def token_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    # log p(x_t | x_<t); first token has no probability.
    shifted = logits[..., :-1, :].float().log_softmax(-1)
    targets = token_ids[..., 1:].unsqueeze(-1)
    values = shifted.gather(-1, targets).squeeze(-1)
    return torch.cat([torch.full_like(values[..., :1], float("nan")), values], dim=-1)


@torch.inference_mode()
def extract(config_path: str, manifest_path: str) -> None:
    cfg = RunConfig.load(config_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_id, trust_remote_code=cfg.model.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.hf_id, torch_dtype=_dtype(cfg.model.dtype), device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
    ).eval()
    n_layers = int(model.config.num_hidden_layers)
    d_model = int(model.config.hidden_size)
    layer_indices = cfg.capture.model_layer_indices(n_layers)
    bank = RepresentationBank(cfg.cache.path, n_layers=n_layers, d_model=d_model,
                              span_cap=cfg.capture.span_cap, depth_fractions=cfg.capture.depth_fractions,
                              model_id=cfg.model.hf_id, chunk_rows=cfg.capture.chunk_rows,
                              overwrite=cfg.cache.overwrite)
    rows = [json.loads(line) for line in Path(manifest_path).read_text().splitlines() if line.strip()]
    device = model.get_input_embeddings().weight.device
    for record in rows:
        prompt = record["prompt"]
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = prompt_inputs.input_ids.shape[1]
        if "generation_tokens" in record:
            completion = torch.tensor([record["generation_tokens"]], device=device)
            generated = torch.cat([prompt_inputs.input_ids, completion], dim=1)
        elif "generation" in record:
            # Teacher-force the exact externally generated completion. Tokenize
            # jointly so boundary-sensitive tokenizers see the original text.
            generated = tokenizer(prompt + record["generation"], return_tensors="pt").input_ids.to(device)
        else:
            generation_kwargs = {"max_new_tokens": cfg.generation.max_new_tokens,
                                 "do_sample": cfg.generation.do_sample}
            if cfg.generation.do_sample:
                generation_kwargs["temperature"] = cfg.generation.temperature
            generated = model.generate(**prompt_inputs, **generation_kwargs)
        full = model(generated, output_hidden_states=True, use_cache=False)
        states = full.hidden_states  # embedding output plus every block output
        h_last = torch.stack([h[0, -1] for h in states]).cpu().to(torch.float16).numpy()
        completion_len = min(generated.shape[1] - prompt_len, cfg.capture.span_cap)
        span = np.zeros((len(layer_indices), cfg.capture.span_cap, d_model), dtype=np.float16)
        # block i output lives at hidden_states[i+1]
        for j, block_i in enumerate(layer_indices):
            if completion_len:
                span[j, :completion_len] = states[block_i + 1][0, prompt_len:prompt_len + completion_len].cpu().to(torch.float16).numpy()
        probs = token_logprobs(full.logits, generated)[0, prompt_len:prompt_len + completion_len].cpu().numpy()
        padded_probs = np.full(cfg.capture.span_cap, np.nan, dtype=np.float32)
        padded_probs[:completion_len] = probs
        mask = np.zeros(cfg.capture.span_cap, dtype=bool)
        mask[:completion_len] = True
        text = record.get("generation") or tokenizer.decode(
            generated[0, prompt_len:], skip_special_tokens=True
        )
        bank.append(CacheRow(h_last, span, mask, padded_probs, str(record["pair_id"]),
                             record["role"], float(record.get("label", np.nan)),
                             record.get("label_protocol", "BLEURT-0.5"), prompt, text))
