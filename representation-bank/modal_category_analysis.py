from __future__ import annotations

import json

import modal

app = modal.App("repbank-category-analysis")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy>=1.26", "pydantic>=2.7", "torch>=2.6", "transformers>=5.5,<5.6"
    )
    .add_local_dir("src/repbank", remote_path="/root/repbank")
    .add_local_file(
        "data/program_datasets/protocol_qa/truthfulqa/validation.jsonl",
        remote_path="/root/truthfulqa.jsonl",
    )
)
outputs = modal.Volume.from_name("repbank-results", create_if_missing=True)


def rank_correlation(left, right) -> float:
    import numpy as np

    def ranks(values):
        order = np.argsort(values)
        output = np.empty(len(values), dtype=float)
        output[order] = np.arange(len(values))
        return output

    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])


@app.function(image=image, cpu=16, memory=49152, timeout=45 * 60,
              volumes={"/outputs": outputs})
def analyze(repeats: int = 5000) -> dict:
    import numpy as np

    from repbank.gates import (
        auc,
        load_bank,
        same_class_covariance_control,
        stratified_heldout_scores,
    )
    from repbank.geometry import reduced_basis

    with open("/root/truthfulqa.jsonl") as handle:
        category_rows = [json.loads(line) for line in handle]
    category_by_group = {
        f"truthfulqa-{index:04d}": row["category"] for index, row in enumerate(category_rows)
    }
    banks = [load_bank(path) for path in (
        "/outputs/coupled/primary-v2.npz", "/outputs/coupled/holdout-primary.npz"
    )]
    for _, metadata in banks[1:]:
        for field in ("model_hash", "tokenizer_sha256", "n_layers", "d_model"):
            if metadata[field] != banks[0][1][field]:
                raise ValueError(f"primary banks differ on {field}")
    metadata = banks[0][1]
    block = round(0.8 * (metadata["n_layers"] - 1))
    states = np.concatenate([
        bank["h_last"][:, block + 1].astype(np.float32) for bank, _ in banks
    ])
    labels = np.concatenate([bank["label"].astype(np.int8) for bank, _ in banks])
    groups = np.concatenate([bank["question_id"] for bank, _ in banks])
    confidence = np.concatenate([np.nanmean(bank["logprobs"], axis=1) for bank, _ in banks])
    categories = np.array([category_by_group[group] for group in groups])
    eu, raw, leverage = stratified_heldout_scores(
        states, labels, groups, categories, return_leverage=True
    )

    center, basis = reduced_basis(states, max_dim=64)
    reduced = (states - center) @ basis
    covariance_control = same_class_covariance_control(
        reduced[labels == 1], reduced[labels == 0], repeats=10_000
    )
    rng = np.random.default_rng(501_693)
    category_results = {}
    for category in sorted(np.unique(categories).tolist()):
        mask = categories == category
        category_groups = np.unique(groups[mask])
        result = {
            "questions": len(category_groups), "rows": int(mask.sum()),
            "correct": int(labels[mask].sum()), "wrong": int(mask.sum() - labels[mask].sum()),
            "eu_auc": auc(labels[mask], eu[mask]),
            "confidence_auc": auc(labels[mask], confidence[mask]),
            "mean_correct_manifold_leverage": float(leverage[mask].mean()),
        }
        draws = []
        if result["correct"] and result["wrong"]:
            for _ in range(repeats):
                sampled = rng.choice(category_groups, len(category_groups), replace=True)
                indices = np.concatenate([np.flatnonzero(mask & (groups == group)) for group in sampled])
                if len(np.unique(labels[indices])) == 2:
                    draws.append(auc(labels[indices], eu[indices]))
        if draws:
            result["eu_auc_q025"] = float(np.quantile(draws, 0.025))
            result["eu_auc_q975"] = float(np.quantile(draws, 0.975))
        category_results[category] = result

    eligible = [item for item in category_results.values()
                if item["questions"] >= 10 and "eu_auc_q025" in item]
    mechanism = {
        "eligible_category_min_questions": 10,
        "eligible_categories": len(eligible),
        "spearman_category_mean_leverage_vs_eu_auc": rank_correlation(
            np.array([item["mean_correct_manifold_leverage"] for item in eligible]),
            np.array([item["eu_auc"] for item in eligible]),
        ),
    }
    result = {
        "protocol": "category-stratified question-grouped five-fold; fold-local PCA/directions",
        "rows": len(labels), "questions": len(np.unique(groups)),
        "correct": int(labels.sum()), "wrong": int(len(labels) - labels.sum()),
        "headline": {
            "eu_auc": auc(labels, eu), "raw_auc": auc(labels, raw),
            "confidence_auc": auc(labels, confidence),
        },
        "same_class_covariance_control": covariance_control,
        "category_mechanism": mechanism,
        "categories": category_results,
    }
    target = "/outputs/coupled/full-category-stratified-results.json"
    with open(target, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    outputs.commit()
    return result


@app.local_entrypoint()
def main(repeats: int = 5000):
    print(json.dumps(analyze.remote(repeats), indent=2, sort_keys=True))
