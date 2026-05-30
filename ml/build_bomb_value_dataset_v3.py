from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SAMPLE_TYPES = {
    "positive": 1,
    "easy_negative": 2,
    "rollout_hard_negative": 3,
    "counterfactual_negative": 4,
}


def _sample_indices(rng, indices, max_count):
    indices = np.asarray(indices, dtype=np.int64)
    if max_count <= 0 or len(indices) <= max_count:
        return indices
    return rng.choice(indices, size=max_count, replace=False)


def build(args):
    rng = np.random.default_rng(args.seed)
    base = np.load(args.base_dataset)
    labels = base["labels"].astype(np.float32)
    source = base["source"].astype(np.int8) if "source" in base.files else np.zeros(len(labels), dtype=np.int8)

    pos_idx = np.flatnonzero(labels == 1)
    easy_idx = np.flatnonzero(labels == 0)
    easy_target = int(max(args.min_easy_negatives, len(pos_idx) * args.easy_negative_ratio))
    easy_idx = _sample_indices(rng, easy_idx, easy_target)

    obs_parts = [base["observations"][pos_idx], base["observations"][easy_idx]]
    scalar_parts = [base["scalar_features"][pos_idx], base["scalar_features"][easy_idx]]
    label_parts = [np.ones(len(pos_idx), dtype=np.float32), np.zeros(len(easy_idx), dtype=np.float32)]
    type_parts = [
        np.full(len(pos_idx), SAMPLE_TYPES["positive"], dtype=np.int8),
        np.full(len(easy_idx), SAMPLE_TYPES["easy_negative"], dtype=np.int8),
    ]
    weight_parts = [
        np.full(len(pos_idx), args.positive_weight, dtype=np.float32),
        np.full(len(easy_idx), args.easy_negative_weight, dtype=np.float32),
    ]

    rollout_count = 0
    if args.rollout_hard_negatives and Path(args.rollout_hard_negatives).exists():
        rollout = np.load(args.rollout_hard_negatives)
        idx = _sample_indices(rng, np.arange(len(rollout["labels"])), args.max_rollout_hard_negatives)
        rollout_count = len(idx)
        obs_parts.append(rollout["observations"][idx])
        scalar_parts.append(rollout["scalar_features"][idx])
        label_parts.append(np.zeros(len(idx), dtype=np.float32))
        type_parts.append(np.full(len(idx), SAMPLE_TYPES["rollout_hard_negative"], dtype=np.int8))
        weight_parts.append(np.full(len(idx), args.rollout_hard_negative_weight, dtype=np.float32))

    counter_count = 0
    counter_reason_distribution = {}
    if args.counterfactual_negatives and Path(args.counterfactual_negatives).exists():
        counter = np.load(args.counterfactual_negatives)
        idx = _sample_indices(rng, np.arange(len(counter["labels"])), args.max_counterfactual_negatives)
        counter_count = len(idx)
        obs_parts.append(counter["observations"][idx])
        scalar_parts.append(counter["scalar_features"][idx])
        label_parts.append(np.zeros(len(idx), dtype=np.float32))
        type_parts.append(np.full(len(idx), SAMPLE_TYPES["counterfactual_negative"], dtype=np.int8))
        weight_parts.append(np.full(len(idx), args.counterfactual_negative_weight, dtype=np.float32))
        if "negative_reason" in counter.files:
            counter_reason_distribution = {str(k): int(v) for k, v in zip(*np.unique(counter["negative_reason"][idx], return_counts=True))}

    observations = np.concatenate(obs_parts, axis=0).astype(np.float32)
    scalar_features = np.concatenate(scalar_parts, axis=0).astype(np.float32)
    out_labels = np.concatenate(label_parts, axis=0).astype(np.float32)
    sample_type = np.concatenate(type_parts, axis=0).astype(np.int8)
    sample_weight = np.concatenate(weight_parts, axis=0).astype(np.float32)

    order = rng.permutation(len(out_labels))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=observations[order],
        scalar_features=scalar_features[order],
        labels=out_labels[order],
        source=sample_type[order],
        sample_type=sample_type[order],
        sample_weight=sample_weight[order],
    )
    stats = {
        "base_dataset": args.base_dataset,
        "rollout_hard_negatives": args.rollout_hard_negatives,
        "counterfactual_negatives": args.counterfactual_negatives,
        "output": str(output),
        "samples": int(len(out_labels)),
        "positive_count": int(len(pos_idx)),
        "easy_negative_count": int(len(easy_idx)),
        "rollout_hard_negative_count": int(rollout_count),
        "counterfactual_negative_count": int(counter_count),
        "positive_fraction": float(out_labels.mean()),
        "sample_type_distribution": {str(k): int(v) for k, v in zip(*np.unique(sample_type, return_counts=True))},
        "counterfactual_reason_distribution": counter_reason_distribution,
        "weights": {
            "positive": args.positive_weight,
            "easy_negative": args.easy_negative_weight,
            "rollout_hard_negative": args.rollout_hard_negative_weight,
            "counterfactual_negative": args.counterfactual_negative_weight,
        },
    }
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build selector dataset v3 with weighted hard negatives.")
    parser.add_argument("--base_dataset", default="ml/datasets/bomb_value_dataset.npz")
    parser.add_argument("--rollout_hard_negatives", default="ml/datasets/bomb_hard_negatives.npz")
    parser.add_argument("--counterfactual_negatives", default="ml/datasets/bomb_counterfactual_negatives.npz")
    parser.add_argument("--output", default="ml/datasets/bomb_value_dataset_v3.npz")
    parser.add_argument("--easy_negative_ratio", type=float, default=2.0)
    parser.add_argument("--min_easy_negatives", type=int, default=500)
    parser.add_argument("--max_rollout_hard_negatives", type=int, default=5000)
    parser.add_argument("--max_counterfactual_negatives", type=int, default=5000)
    parser.add_argument("--positive_weight", type=float, default=1.0)
    parser.add_argument("--easy_negative_weight", type=float, default=1.0)
    parser.add_argument("--rollout_hard_negative_weight", type=float, default=4.0)
    parser.add_argument("--counterfactual_negative_weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
