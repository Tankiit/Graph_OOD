#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def summarize(path: Path) -> dict[str, float | int]:
    effective, participation, top_mass, frobenius = [], [], [], []
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        for a_key in sorted(keys):
            if ".lora_A." not in a_key:
                continue
            b_key = a_key.replace(".lora_A.", ".lora_B.")
            if b_key not in keys:
                continue
            a = checkpoint.get_tensor(a_key).float()
            b = checkpoint.get_tensor(b_key).float()
            _, rb = torch.linalg.qr(b, mode="reduced")
            _, ra = torch.linalg.qr(a.T, mode="reduced")
            singular = torch.linalg.svdvals(rb @ ra.T).numpy()
            probability = singular / max(float(singular.sum()), 1e-30)
            effective.append(float(np.exp(-(probability * np.log(probability + 1e-30)).sum())))
            participation.append(float(singular.sum() ** 2 / max(float(singular @ singular), 1e-30)))
            top_mass.append(float(probability[0]))
            frobenius.append(float(np.sqrt(singular @ singular)))
    return {
        "modules": len(effective),
        "entropy_effective_rank_median": float(np.median(effective)),
        "participation_rank_median": float(np.median(participation)),
        "top_singular_mass_median": float(np.median(top_mass)),
        "update_frobenius_median": float(np.median(frobenius)),
        "update_frobenius_mean": float(np.mean(frobenius)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapters", nargs="+", help="NAME=adapter_model.safetensors")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {}
    for item in args.adapters:
        name, raw_path = item.split("=", 1)
        result[name] = summarize(Path(raw_path))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
