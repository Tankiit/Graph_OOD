from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .coupled_extract import load_model
from .gates import load_bank
from .generation_set import FrozenGenerationSet
from .layers import fraction_to_index, transformer_blocks


class RatioHead(nn.Module):
    def __init__(self, d: int, hidden: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value).squeeze(-1)


@dataclass
class FittedReadout:
    center: np.ndarray
    scale: np.ndarray
    basis: np.ndarray
    head: RatioHead
    heldout_pair_accuracy: float
    train_pairs: int
    heldout_pairs: int

    def score(self, state: Tensor) -> Tensor:
        center = torch.as_tensor(self.center, device=state.device, dtype=state.dtype)
        scale = torch.as_tensor(self.scale, device=state.device, dtype=state.dtype)
        basis = torch.as_tensor(self.basis, device=state.device, dtype=state.dtype)
        return self.head(((state - center) / scale) @ basis)


def paired_indices(pair_ids: np.ndarray, roles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    correct, wrong = [], []
    for pair_id in np.unique(pair_ids):
        rows = np.flatnonzero(pair_ids == pair_id)
        c = rows[roles[rows] == 0]
        w = rows[roles[rows] == 1]
        if len(c) and len(w):
            correct.append(c[0])
            wrong.append(w[0])
    return np.asarray(correct, dtype=np.int64), np.asarray(wrong, dtype=np.int64)


def fit_readout(states: np.ndarray, pair_ids: np.ndarray, roles: np.ndarray, *,
                max_dim: int = 32, epochs: int = 500, seed: int = 0) -> FittedReadout:
    ci, wi = paired_indices(pair_ids, roles)
    if len(ci) < 4:
        raise ValueError("Jacobian readout needs at least four correct/wrong pairs")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ci))
    heldout = max(1, len(ci) // 5)
    test, train = order[:heldout], order[heldout:]
    fit_rows = np.concatenate([ci[train], wi[train]])
    center = states[fit_rows].mean(0)
    scale = states[fit_rows].std(0).clip(1e-4)
    standardized = (states[fit_rows] - center) / scale
    _, _, vt = np.linalg.svd(standardized, full_matrices=False)
    basis = vt[:min(max_dim, len(fit_rows) - 1)].T.astype(np.float32)

    def transform(rows: np.ndarray) -> Tensor:
        values = ((states[rows] - center) / scale) @ basis
        return torch.tensor(values, dtype=torch.float32)

    a, b = transform(ci[train]), transform(wi[train])
    torch.manual_seed(seed)
    head = RatioHead(basis.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(epochs):
        optimizer.zero_grad()
        difference = head(b) - head(a)
        torch.nn.functional.softplus(-difference).mean().backward()
        optimizer.step()
    with torch.no_grad():
        accuracy = float((head(transform(wi[test])) > head(transform(ci[test]))).float().mean())
    return FittedReadout(
        center.astype(np.float32), scale.astype(np.float32), basis, head.eval(), accuracy,
        len(train), len(test),
    )


def dispersion(vectors: np.ndarray) -> dict[str, float]:
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-12)
    mean = normalized.mean(0)
    resultant = float(np.linalg.norm(mean))
    mean /= max(resultant, 1e-12)
    angles = np.degrees(np.arccos(np.clip(normalized @ mean, -1, 1)))
    return {
        "median_degrees": float(np.median(angles)),
        "p90_degrees": float(np.percentile(angles, 90)),
        "resultant_length": resultant,
    }


def jvp_diagnostics(jvp: np.ndarray, roles: np.ndarray) -> dict[str, Any]:
    """Report epsilon agreement and truthful/wrong propagation difference."""
    middle = jvp.shape[1] // 2
    reference = jvp[:, middle]
    reference_norm = np.linalg.norm(reference, axis=1).clip(1e-12)
    relative = np.linalg.norm(jvp - reference[:, None], axis=2) / reference_norm[:, None]
    cosine_values = np.sum(jvp * reference[:, None], axis=2) / (
        np.linalg.norm(jvp, axis=2).clip(1e-12) * reference_norm[:, None]
    )
    role_means = [jvp[roles == role, middle].mean(0) for role in (0, 1)]
    denominator = 0.5 * sum(np.linalg.norm(value) for value in role_means)
    return {
        "reference_epsilon_index": middle,
        "median_relative_error_by_epsilon": np.median(relative, axis=0).tolist(),
        "median_cosine_by_epsilon": np.median(cosine_values, axis=0).tolist(),
        "truth_wrong_mean_relative_gap": float(
            np.linalg.norm(role_means[0] - role_means[1]) / max(denominator, 1e-12)
        ),
    }


def _forward_captured(model: nn.Module, input_ids: Tensor, layer_l: int, layer_L: int,
                      shift: Tensor | None = None) -> tuple[Tensor, Tensor]:
    blocks = transformer_blocks(model)
    captured: dict[str, Tensor] = {}

    def early(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Frozen parameters and integer token IDs provide no gradient root.
        # Reintroduce the intervention residual as the leaf whose VJP we want.
        hidden = hidden.detach()
        if shift is not None:
            hidden = hidden.clone()
            hidden[:, -1] += shift
        if torch.is_grad_enabled():
            hidden.requires_grad_(True)
        output = (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        captured["early"] = hidden
        return output

    def late(_module, _inputs, output):
        captured["late"] = output[0] if isinstance(output, tuple) else output

    handles = [blocks[layer_l].register_forward_hook(early),
               blocks[layer_L].register_forward_hook(late)]
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    # Return the exact early tensor consumed downstream, not a post-hoc view.
    return captured["early"], captured["late"][:, -1]


def run_jacobian_probe(*, model_path: str, bank_path: str | Path,
                       generation_set_path: str | Path, output_path: str | Path,
                       adapter_path: str | None = None, layer_fraction: float = 0.5,
                       readout_fraction: float = 0.8, max_pairs: int = 12,
                       jvp_examples: int = 4, epsilons: tuple[float, ...] = (0.1, 0.3, 1.0),
                       dtype: torch.dtype = torch.bfloat16) -> dict[str, Any]:
    bank, metadata = load_bank(bank_path)
    frozen = FrozenGenerationSet.read(generation_set_path)
    if metadata["generation_set_checksum"] != frozen.checksum_sha256:
        raise ValueError("Jacobian bank and frozen generation set checksums differ")
    n_layers = int(metadata["n_layers"])
    layer_l = fraction_to_index(layer_fraction, n_layers)
    layer_L = fraction_to_index(readout_fraction, n_layers)
    if layer_l >= layer_L:
        raise ValueError("intervention depth must precede readout depth")
    readout = fit_readout(
        bank["h_last"][:, layer_L + 1].astype(np.float32), bank["pair_id"], bank["role"]
    )
    ci, wi = paired_indices(bank["pair_id"], bank["role"])
    selected = np.column_stack([ci[:max_pairs], wi[:max_pairs]]).reshape(-1)

    model = load_model(model_path, adapter_path, dtype, "cuda")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    readout.head.to(device=device, dtype=torch.float32)

    # The fixed statistical direction is estimated at the extracted readout states.
    reference = torch.tensor(
        bank["h_last"][selected, layer_L + 1].astype(np.float32), requires_grad=True
    )
    readout.head.to(dtype=torch.float32, device="cpu")
    fixed_u, = torch.autograd.grad(readout.score(reference).sum(), reference)
    fixed_u = fixed_u.mean(0).to(device=device, dtype=dtype)
    readout.head.to(device=device, dtype=torch.float32)

    v_star, v_fixed, used = [], [], []
    for row in selected:
        ids = torch.tensor(frozen.records[int(row)].token_ids, device=device)[None]
        with torch.enable_grad():
            early, late = _forward_captured(model, ids, layer_l, layer_L)
            score = readout.score(late.float()).sum()
            star, = torch.autograd.grad(score, early, retain_graph=True)
            fixed, = torch.autograd.grad((late * fixed_u).sum(), early)
        v_star.append(star[0, -1].float().cpu().numpy())
        v_fixed.append(fixed[0, -1].float().cpu().numpy())
        used.append(int(row))
    v_star_array = np.stack(v_star)
    v_fixed_array = np.stack(v_fixed)
    steering = v_star_array.mean(0)
    steering /= max(np.linalg.norm(steering), 1e-12)

    jvp = np.zeros((min(jvp_examples * 2, len(selected)), len(epsilons), metadata["d_model"]),
                   dtype=np.float32)
    jvp_rows = selected[:len(jvp)]
    for out_index, row in enumerate(jvp_rows):
        ids = torch.tensor(frozen.records[int(row)].token_ids, device=device)[None]
        with torch.no_grad():
            _, baseline = _forward_captured(model, ids, layer_l, layer_L)
            for eps_index, epsilon in enumerate(epsilons):
                shift = torch.tensor(epsilon * steering, device=device, dtype=dtype)
                _, perturbed = _forward_captured(model, ids, layer_l, layer_L, shift)
                jvp[out_index, eps_index] = (
                    (perturbed - baseline) / epsilon
                )[0].float().cpu().numpy()

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    probe_metadata = {
        "schema_version": 1, "model_path": model_path, "adapter_path": adapter_path,
        "source_bank": str(bank_path), "source_bank_checksum": target_checksum(bank_path),
        "generation_set_checksum": frozen.checksum_sha256, "layer_fraction": layer_fraction,
        "readout_fraction": readout_fraction, "layer_index": layer_l,
        "readout_index": layer_L, "heldout_pair_accuracy": readout.heldout_pair_accuracy,
        "train_pairs": readout.train_pairs, "heldout_pairs": readout.heldout_pairs,
        "v_star_dispersion": dispersion(v_star_array),
        "v_fixed_u_dispersion": dispersion(v_fixed_array), "epsilons": list(epsilons),
        "jvp_diagnostics": jvp_diagnostics(jvp, bank["role"][jvp_rows]),
    }
    with target.open("wb") as handle:
        np.savez(
            handle, v_star=v_star_array.astype(np.float16),
            v_fixed_u=v_fixed_array.astype(np.float16), steering=steering.astype(np.float32),
            jvp=jvp.astype(np.float16), row_indices=np.asarray(used), jvp_rows=jvp_rows,
            roles=bank["role"][selected], metadata=np.array(json.dumps(probe_metadata, sort_keys=True)),
        )
    checksum = target_checksum(target)
    target.with_suffix(target.suffix + ".sha256").write_text(f"{checksum}  {target.name}\n")
    return {**probe_metadata, "rows": len(selected), "output": str(target), "checksum": checksum}


def target_checksum(path: str | Path) -> str:
    from .coupled_extract import file_sha256

    return file_sha256(path)
