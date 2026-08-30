from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from .generation_set import FrozenGenerationSet, tokenizer_digest


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: str | Path) -> None:
    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    expected = sidecar.read_text().strip().split()[0]
    actual = file_sha256(target)
    if expected != actual:
        raise ValueError(f"bank checksum mismatch: expected {expected}, got {actual}")


def fraction_indices(fractions: list[float], n_layers: int) -> list[int]:
    return [min(n_layers - 1, max(0, round(value * (n_layers - 1)))) for value in fractions]


def model_revision(model: torch.nn.Module, model_path: str, adapter_path: str | None) -> str:
    commit = getattr(model.config, "_commit_hash", None)
    payload = {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "commit": commit,
        "config": model.config.to_dict(),
    }
    if adapter_path and Path(adapter_path).exists():
        payload["adapter_files"] = {
            str(path.relative_to(adapter_path)): file_sha256(path)
            for path in sorted(Path(adapter_path).rglob("*")) if path.is_file()
        }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def load_model(model_path: str, adapter_path: str | None, dtype: torch.dtype, device_map: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, device_map=device_map, trust_remote_code=True
    )
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("adapter extraction requires `pip install -e '.[extract]'`") from exc
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = PeftModel.from_pretrained(model, adapter_path)
        missing = [str(item.message) for item in caught if "missing adapter keys" in str(item.message)]
        if missing:
            raise RuntimeError(f"adapter export is incompatible with model modules: {missing[0]}")
    return model.eval()


@torch.inference_mode()
def extract_frozen(
    *, model_path: str, generation_set_path: str | Path, output_path: str | Path,
    adapter_path: str | None = None, adapter_id: str = "base", rank: int = 0,
    base_model: bool = False, depth_fractions: list[float] | None = None,
    span_cap: int = 32, include_span: bool = True, batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16, device_map: str = "cuda",
) -> dict[str, Any]:
    frozen = FrozenGenerationSet.read(generation_set_path)
    tokenizer = AutoTokenizer.from_pretrained(
        frozen.tokenizer_id, trust_remote_code=True, revision=None
    )
    if tokenizer_digest(tokenizer) != frozen.tokenizer_sha256:
        raise ValueError("tokenizer checksum differs from frozen generation set")
    model_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer_digest(model_tokenizer) != frozen.tokenizer_sha256:
        raise ValueError("model tokenizer is outside the frozen tokenizer family")
    tokenizer.padding_side = "right"
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")

    model = load_model(model_path, adapter_path, dtype, device_map)
    device = model.get_input_embeddings().weight.device
    n_layers = int(model.config.num_hidden_layers)
    d_model = int(model.config.hidden_size)
    fractions = depth_fractions or [0.2, 0.5, 0.8]
    block_indices = fraction_indices(fractions, n_layers)
    count = len(frozen.records)
    h_last = np.empty((count, n_layers + 1, d_model), dtype=np.float16)
    h_span = (
        np.zeros((count, len(fractions), span_cap, d_model), dtype=np.float16)
        if include_span else None
    )
    span_mask = np.zeros((count, span_cap), dtype=np.bool_)
    logprobs = np.full((count, span_cap), np.nan, dtype=np.float32)

    for start in range(0, count, batch_size):
        records = frozen.records[start:start + batch_size]
        sequences = [torch.tensor(record.token_ids, dtype=torch.long) for record in records]
        lengths = torch.tensor([len(sequence) for sequence in sequences], device=device)
        input_ids = pad_sequence(sequences, batch_first=True, padding_value=pad_token_id).to(device)
        positions = torch.arange(input_ids.shape[1], device=device)[None, :]
        attention_mask = positions < lengths[:, None]
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, use_cache=False,
        )
        row_indices = torch.arange(len(records), device=device)
        last_indices = lengths - 1
        for layer, states in enumerate(outputs.hidden_states):
            h_last[start:start + len(records), layer] = (
                states[row_indices, last_indices].to(torch.float16).cpu().numpy()
            )
        token_lp = outputs.logits[:, :-1].float().log_softmax(-1).gather(
            -1, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        for local_index, record in enumerate(records):
            answer_length = len(record.token_ids) - record.answer_start
            captured = min(answer_length, span_cap)
            destination = start + local_index
            span_mask[destination, :captured] = True
            source_start = record.answer_start
            logprobs[destination, :captured] = (
                token_lp[local_index, source_start - 1:source_start - 1 + captured].cpu().numpy()
            )
            if h_span is not None:
                for depth_index, block_index in enumerate(block_indices):
                    h_span[destination, depth_index, :captured] = (
                        outputs.hidden_states[block_index + 1][
                            local_index, source_start:source_start + captured
                        ].to(torch.float16).cpu().numpy()
                    )
        del outputs, token_lp

    revision = model_revision(model, model_path, adapter_path)
    metadata = {
        "schema_version": 1,
        "generation_set_checksum": frozen.checksum_sha256,
        "tokenizer_id": frozen.tokenizer_id,
        "tokenizer_sha256": frozen.tokenizer_sha256,
        "model_path": model_path,
        "model_hash": revision,
        "adapter_id": adapter_id,
        "adapter_path": adapter_path,
        "rank": rank,
        "base_model": base_model,
        "depth_fractions": fractions,
        "block_indices": block_indices,
        "n_layers": n_layers,
        "d_model": d_model,
        "span_cap": span_cap,
        "include_span": include_span,
        "padding_side": "right",
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    arrays: dict[str, Any] = {
        "h_last": h_last,
        "span_mask": span_mask,
        "logprobs": logprobs,
        "pair_id": np.array([record.pair_id for record in frozen.records]),
        "question_id": np.array([record.question_id for record in frozen.records]),
        "role": np.array([record.role for record in frozen.records], dtype=np.int8),
        "label": np.array([record.label for record in frozen.records], dtype=np.int8),
        "answer_start": np.array([record.answer_start for record in frozen.records], dtype=np.int32),
        "sequence_length": np.array([len(record.token_ids) for record in frozen.records], dtype=np.int32),
        "metadata": np.array(json.dumps(metadata, sort_keys=True)),
    }
    if h_span is not None:
        arrays["h_span"] = h_span
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(target)
    checksum = file_sha256(target)
    target.with_suffix(target.suffix + ".sha256").write_text(f"{checksum}  {target.name}\n")
    return {**metadata, "rows": count, "output": str(target), "checksum_sha256": checksum}
