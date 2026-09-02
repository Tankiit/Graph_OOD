from __future__ import annotations

import asyncio
import itertools
import json
import math
import os
import random
from pathlib import Path
from typing import Any


def _load_local_env() -> None:
    """Load a local .env without overriding an already configured environment."""
    env_path = Path.cwd() / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"TINKER_API_KEY", "TINKER_PROJECT_ID"}:
            os.environ.setdefault(key, value.strip().strip("'\""))


def _require_key() -> None:
    _load_local_env()
    api_key = os.environ.get("TINKER_API_KEY")
    if api_key and not api_key.startswith("tml-"):
        raise RuntimeError("TINKER_API_KEY is malformed: Tinker API keys start with 'tml-'.")


def load_training_records(data: str, *, split: str = "train",
                          dataset_config: str | None = None,
                          revision: str | None = None,
                          messages_column: str = "messages",
                          role_column: str = "role",
                          max_samples: int | None = None) -> list[dict[str, Any]]:
    """Load chat records from JSON/JSONL or a Hugging Face dataset ID.

    A role column is optional for single-role datasets. If present, it is used by
    ``train_adapter`` to select the requested adapter role.
    """
    path = Path(data).expanduser()
    if path.is_file():
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text())
            records = payload if isinstance(payload, list) else [payload]
        else:
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.exists():
        raise ValueError(f"--data must be a JSON/JSONL file or HF dataset ID, got: {data}")
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face dataset loading requires `pip install -e '.[tinker,hf]'`."
            ) from exc
        kwargs: dict[str, Any] = {"split": split}
        if max_samples is not None:
            kwargs["streaming"] = True
        if revision is not None:
            kwargs["revision"] = revision
        dataset = load_dataset(data, dataset_config, **kwargs)
        rows = itertools.islice(dataset, max_samples) if max_samples is not None else dataset
        records = [dict(row) for row in rows]

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        records = records[:max_samples]

    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if messages_column not in record:
            raise ValueError(
                f"row {index} has no {messages_column!r} column; HF datasets must contain "
                "a standard chat messages column (or use --messages-column)"
            )
        messages = record[messages_column]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"row {index} has an empty or invalid messages value")
        for message in messages:
            if not isinstance(message, dict) or not {"role", "content"} <= message.keys():
                raise ValueError(f"row {index} messages must contain role/content objects")
        item = {"messages": messages}
        if role_column in record:
            item["role"] = record[role_column]
        normalized.append(item)
    return normalized


def select_role_records(records: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    """Select a role when labeled; unlabeled datasets are single-role corpora."""
    if any("role" in record for record in records):
        records = [record for record in records if record.get("role") == role]
    return records


def scheduled_learning_rate(step: int, total_steps: int, peak_lr: float,
                            schedule: str = "constant", warmup_ratio: float = 0.0,
                            min_lr_ratio: float = 0.1) -> float:
    """Return a constant or warmup-cosine learning rate for a zero-based step."""
    if schedule == "constant":
        return peak_lr
    if schedule != "cosine":
        raise ValueError(f"unknown learning-rate schedule: {schedule}")
    warmup_steps = round(total_steps * warmup_ratio)
    if warmup_steps and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    multiplier = min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    return peak_lr * multiplier


async def smoke_prompt_logprobs(model_id: str, text: str) -> list[float | None]:
    """Kill switch 1: verify prefill scoring on an unmodified base model."""
    _require_key()
    import tinker
    from tinker import types

    service = tinker.ServiceClient()
    sampler = await service.create_sampling_client_async(base_model=model_id)
    tokenizer = sampler.get_tokenizer()
    token_ids = tokenizer.encode(text)
    prompt = types.ModelInput.from_ints(token_ids)
    result = await sampler.sample_async(
        prompt=prompt, num_samples=1, sampling_params=types.SamplingParams(max_tokens=1),
        include_prompt_logprobs=True,
    )
    values = result.prompt_logprobs
    if values is None or len(values) != len(token_ids):
        raise RuntimeError("prompt_logprobs missing or length-mismatched")
    if all(v is None for v in values[1:]):
        raise RuntimeError("prompt_logprobs contains no scored token after position zero")
    return values


async def train_adapter(model_id: str, rank: int, role: str, data_path: str,
                        epochs: int = 3, lr: float = 1e-4, max_length: int = 512,
                        ttl_seconds: int = 7 * 24 * 3600, seed: int = 0,
                        batch_size: int = 32, renderer_name: str | None = None,
                        split: str = "train", dataset_config: str | None = None,
                        revision: str | None = None, messages_column: str = "messages",
                        role_column: str = "role", max_samples: int | None = None,
                        save_name: str | None = None, lr_schedule: str = "constant",
                        warmup_ratio: float = 0.0, min_lr_ratio: float = 0.1) -> str:
    """Train M_true or M_hal from local chat data or a Hugging Face dataset."""
    _require_key()
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised.data import conversation_to_datum
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    records = load_training_records(
        data_path, split=split, dataset_config=dataset_config, revision=revision,
        messages_column=messages_column, role_column=role_column, max_samples=max_samples,
    )
    # A dataset with a role column may hold both adapter corpora. A dataset
    # without one is treated as an already-selected, single-role corpus.
    records = select_role_records(records, role)
    if not records:
        raise ValueError(f"no role={role!r} rows in {data_path}")
    service = tinker.ServiceClient()
    client = await service.create_lora_training_client_async(
        base_model=model_id, rank=rank, seed=seed
    )
    tokenizer = get_tokenizer(model_id)
    renderer_name = renderer_name or ("qwen3_5_disable_thinking" if "Qwen3.5" in model_id else "qwen3")
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    rng = random.Random(seed)
    total_steps = epochs * math.ceil(len(records) / batch_size)
    step = 0
    for _epoch in range(epochs):
        rng.shuffle(records)
        for start in range(0, len(records), batch_size):
            batch = [conversation_to_datum(
                record["messages"], renderer, max_length=max_length,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
                     for record in records[start:start + batch_size]]
            fwd = await client.forward_backward_async(batch, loss_fn="cross_entropy")
            step_lr = scheduled_learning_rate(
                step, total_steps, lr, lr_schedule, warmup_ratio, min_lr_ratio
            )
            opt = await client.optim_step_async(tinker.AdamParams(learning_rate=step_lr))
            await fwd.result_async()
            await opt.result_async()
            step += 1
    save_future = await client.save_weights_for_sampler_async(
        name=save_name or f"{role}-r{rank}", ttl_seconds=ttl_seconds)
    saved = await save_future.result_async()
    return saved.path


def export_adapter(tinker_path: str, base_model: str, output_root: str, merge: bool = False) -> Path:
    """Kill switch 4: download, convert/merge, and leave a loadable HF directory."""
    _load_local_env()
    from tinker_cookbook import weights

    root = Path(output_root)
    raw = root / "tinker_adapter"
    output = root / ("merged_model" if merge else "peft_adapter")
    adapter_dir = weights.download(tinker_path=tinker_path, output_dir=str(raw))
    normalize_tinker_adapter_config(Path(adapter_dir))
    if merge:
        weights.build_hf_model(base_model=base_model, adapter_path=adapter_dir, output_path=str(output))
    else:
        weights.build_lora_adapter(base_model=base_model, adapter_path=adapter_dir, output_path=str(output))
    return output


def download_raw_adapter(tinker_path: str, output_root: str) -> Path:
    """Download and normalize a Tinker checkpoint without converting its keys."""
    _load_local_env()
    from tinker_cookbook import weights

    result = Path(weights.download(tinker_path=tinker_path, output_dir=output_root))
    normalize_tinker_adapter_config(result)
    return result


def normalize_tinker_adapter_config(adapter_dir: Path) -> list[str]:
    """Replace ``all-linear`` with the exact modules present in a Tinker checkpoint.

    Qwen3.5 contains fused linear-attention projections that PEFT discovers under
    ``all-linear`` but Tinker's exported checkpoint does not train. Declaring the
    observed leaf modules prevents PEFT from silently initializing absent LoRA
    matrices at inference time.
    """
    from safetensors import safe_open

    weights_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    with safe_open(weights_path, framework="pt") as checkpoint:
        # Some safetensors releases expose keys but do not make safe_open iterable.
        targets = sorted({
            key.split(".lora_")[0].rsplit(".", 1)[-1]
            for key in checkpoint.keys()  # noqa: SIM118 - safe_open is not iterable
        })
    config = json.loads(config_path.read_text())
    config["target_modules"] = targets
    temporary = config_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    temporary.replace(config_path)
    return targets


def run(coro):
    return asyncio.run(coro)
