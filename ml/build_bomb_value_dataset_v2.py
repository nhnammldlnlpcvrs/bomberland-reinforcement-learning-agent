from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.build_bomb_value_dataset import build as build_v1


def _load_or_build_base(args):
    base_path = Path(args.base_dataset)
    if base_path.exists() and not args.rebuild_base:
        return np.load(base_path)

    class BaseArgs:
        input = args.input
        output = args.base_dataset
        escape_window = args.escape_window
        min_current_danger = args.min_current_danger
        min_future_danger = args.min_future_danger
        min_hard_negatives = args.min_hard_negatives
        seed = args.seed

    build_v1(BaseArgs())
    return np.load(base_path)


def build(args):
    base = _load_or_build_base(args)
    hard = np.load(args.hard_negatives)
    rng = np.random.default_rng(args.seed)

    base_obs = base["observations"].astype(np.float32)
    base_scalars = base["scalar_features"].astype(np.float32)
    base_labels = base["labels"].astype(np.float32)
    base_source = base["source"].astype(np.int8) if "source" in base.files else np.zeros(len(base_labels), dtype=np.int8)

    positive_count = int(base_labels.sum())
    easy_negative_mask = base_labels == 0
    hard_count_target = min(len(hard["labels"]), max(args.min_hard_negatives_v2, int(positive_count * args.hard_negative_ratio)))
    hard_idx = rng.choice(np.arange(len(hard["labels"])), size=hard_count_target, replace=False) if hard_count_target else np.zeros(0, dtype=np.int64)

    observations = np.concatenate([base_obs, hard["observations"][hard_idx].astype(np.float32)], axis=0)
    scalar_features = np.concatenate([base_scalars, hard["scalar_features"][hard_idx].astype(np.float32)], axis=0)
    labels = np.concatenate([base_labels, np.zeros(len(hard_idx), dtype=np.float32)], axis=0)
    source = np.concatenate([base_source, np.full(len(hard_idx), 4, dtype=np.int8)], axis=0)

    hard_reason = np.full(len(labels), 0, dtype=np.int16)
    hard_reason[len(base_labels):] = hard["negative_reason"][hard_idx].astype(np.int16)
    original_index = np.concatenate([
        base["original_index"].astype(np.int64) if "original_index" in base.files else np.arange(len(base_labels), dtype=np.int64),
        hard_idx.astype(np.int64),
    ])

    order = rng.permutation(len(labels))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=observations[order],
        scalar_features=scalar_features[order],
        labels=labels[order],
        source=source[order],
        hard_negative_reason=hard_reason[order],
        original_index=original_index[order],
    )

    stats = {
        "input": args.input,
        "base_dataset": args.base_dataset,
        "hard_negatives": args.hard_negatives,
        "output": str(output),
        "samples": int(len(labels)),
        "positive_count": positive_count,
        "easy_negative_count": int(easy_negative_mask.sum()),
        "hard_negative_count": int(len(hard_idx)),
        "positive_fraction": float(labels.mean()),
        "source_distribution": {str(k): int(v) for k, v in zip(*np.unique(source, return_counts=True))},
        "hard_reason_distribution": {str(k): int(v) for k, v in zip(*np.unique(hard["negative_reason"][hard_idx], return_counts=True))} if len(hard_idx) else {},
    }
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build bomb value selector dataset v2 with rollout-derived hard negatives.")
    parser.add_argument("--input", default="ml/datasets/rl_bc_bomb_escape.npz")
    parser.add_argument("--base_dataset", default="ml/datasets/bomb_value_dataset.npz")
    parser.add_argument("--hard_negatives", default="ml/datasets/bomb_hard_negatives.npz")
    parser.add_argument("--output", default="ml/datasets/bomb_value_dataset_v2.npz")
    parser.add_argument("--escape_window", type=int, default=5)
    parser.add_argument("--min_current_danger", type=int, default=3)
    parser.add_argument("--min_future_danger", type=int, default=1)
    parser.add_argument("--min_hard_negatives", type=int, default=1000)
    parser.add_argument("--min_hard_negatives_v2", type=int, default=500)
    parser.add_argument("--hard_negative_ratio", type=float, default=2.0)
    parser.add_argument("--rebuild_base", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
