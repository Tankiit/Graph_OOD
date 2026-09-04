from __future__ import annotations

import json

import modal

app = modal.App("repbank-holdout-analysis")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy>=1.26", "pydantic>=2.7", "torch>=2.6", "transformers>=5.5,<5.6"
    )
    .add_local_dir("src/repbank", remote_path="/root/repbank")
    .add_local_file(
        "artifacts/frozen/truthfulqa_holdout_difficulty_split.json",
        remote_path="/root/difficulty_split.json",
    )
)
outputs = modal.Volume.from_name("repbank-results", create_if_missing=True)


@app.function(image=image, cpu=16, memory=32768, timeout=20 * 60,
              volumes={"/outputs": outputs})
def analyze(repeats: int = 10_000) -> dict:
    import numpy as np

    from repbank.gates import auc, heldout_scores, load_bank

    paths = {
        "primary": "/outputs/coupled/holdout-primary.npz",
        "rank32_50ep_cosine": "/outputs/coupled/holdout-rank32-50ep-cosine.npz",
    }
    with open("/root/difficulty_split.json") as handle:
        split = json.load(handle)
    selected_indices = np.asarray(sorted(
        index for item in split["questions"].values()
        for index in item["evaluation_row_indices"]
    ))
    bin_by_group = {
        group: item["difficulty_bin"] for group, item in split["questions"].items()
    }
    predictions = {}
    labels = groups = None
    for name, path in paths.items():
        bank, metadata = load_bank(path)
        block = round(0.8 * (metadata["n_layers"] - 1))
        states = bank["h_last"][selected_indices, block + 1].astype(np.float32)
        current_labels = bank["label"][selected_indices].astype(np.int8)
        current_groups = bank["question_id"][selected_indices]
        confidence = np.nanmean(bank["logprobs"][selected_indices], axis=1)
        eu, raw = heldout_scores(states, current_labels, current_groups)
        predictions[name] = {"eu": eu, "raw": raw, "confidence": confidence}
        if labels is None:
            labels, groups = current_labels, current_groups
        elif not np.array_equal(labels, current_labels) or not np.array_equal(groups, current_groups):
            raise ValueError("holdout banks have different row order")

    unique_groups = np.unique(groups)
    bins = {
        "all": np.ones(len(labels), dtype=bool),
        **{name: np.array([bin_by_group[group] == name for group in groups])
           for name in ("easy", "medium", "hard")},
    }
    result = {
        "difficulty_definition": split["protocol"],
        "rows": len(labels), "questions": len(unique_groups), "models": {}, "paired_bootstrap": {},
    }
    for model, scores in predictions.items():
        result["models"][model] = {}
        for bin_name, mask in bins.items():
            result["models"][model][bin_name] = {
                "rows": int(mask.sum()), "questions": len(np.unique(groups[mask])),
                "correct": int(labels[mask].sum()), "wrong": int(mask.sum() - labels[mask].sum()),
                **{f"{metric}_auc": auc(labels[mask], values[mask])
                   for metric, values in scores.items()},
            }

    rng = np.random.default_rng(501_693)
    for bin_name, mask in bins.items():
        eligible = np.unique(groups[mask])
        result["paired_bootstrap"][bin_name] = {}
        for metric in ("eu", "confidence"):
            differences = []
            for _ in range(repeats):
                draw = rng.choice(eligible, len(eligible), replace=True)
                indices = np.concatenate([np.flatnonzero(mask & (groups == group)) for group in draw])
                if len(np.unique(labels[indices])) < 2:
                    continue
                differences.append(
                    auc(labels[indices], predictions["rank32_50ep_cosine"][metric][indices])
                    - auc(labels[indices], predictions["primary"][metric][indices])
                )
            values = np.asarray(differences)
            point = (result["models"]["rank32_50ep_cosine"][bin_name][f"{metric}_auc"]
                     - result["models"]["primary"][bin_name][f"{metric}_auc"])
            result["paired_bootstrap"][bin_name][metric] = {
                "difference": point, "q025": float(np.quantile(values, 0.025)),
                "q975": float(np.quantile(values, 0.975)),
                "p_difference_le_zero": float((1 + np.sum(values <= 0)) / (len(values) + 1)),
            }
    target = "/outputs/coupled/holdout-difficulty-results.json"
    with open(target, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    outputs.commit()
    return result


@app.local_entrypoint()
def main(repeats: int = 10_000):
    print(json.dumps(analyze.remote(repeats), indent=2, sort_keys=True))
