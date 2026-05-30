from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_KEYS = (
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "dones",
    "mode_ids",
    "death_within_7",
    "in_blast_corridor",
    "escaped_blast",
    "own_bomb_recent",
    "boxes_destroyed_future",
    "post_bomb_survival_steps",
    "reachable_area_delta",
    "discounted_returns",
)

OPTIONAL_LABELS = (
    "has_escape_path_now",
    "has_escape_after_bomb",
    "would_destroy_boxes_if_bomb",
    "in_future_blast",
    "trapped_if_bomb",
    "safe_tiles_after_bomb_count",
    "blast_corridor_distance",
)


def _load(path: str, source_id: int) -> dict:
    data = np.load(path)
    out = {}
    n = len(data["observations"])
    for key in REQUIRED_KEYS:
        out[key] = data[key]
    for key in OPTIONAL_LABELS:
        if key in data:
            out[key] = data[key]
        else:
            out[key] = np.zeros(n, dtype=np.float32)
    out["source_ids"] = np.full(n, source_id, dtype=np.int64)
    return out


def _concat(parts: list[dict]) -> dict:
    keys = list(parts[0].keys())
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def _balanced_indices(data: dict, args) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    n = len(data["observations"])
    base = np.arange(n)
    selected = [base]
    label_specs = [
        ("death_within_7", args.target_positive_per_label),
        ("escaped_blast", args.target_positive_per_label),
        ("own_bomb_recent", args.target_positive_per_label),
        ("has_escape_after_bomb", args.target_counterfactual_per_label),
        ("trapped_if_bomb", args.target_counterfactual_per_label),
    ]
    for key, target in label_specs:
        if key not in data or target <= 0:
            continue
        pos = np.flatnonzero(data[key].astype(np.float32) > 0.5)
        neg = np.flatnonzero(data[key].astype(np.float32) <= 0.5)
        if len(pos):
            selected.append(rng.choice(pos, size=min(target, max(target, len(pos))), replace=len(pos) < target))
        if len(neg):
            selected.append(rng.choice(neg, size=min(args.negative_per_label, len(neg)), replace=False))
    idx = np.concatenate(selected)
    if len(idx) > args.max_samples:
        idx = rng.choice(idx, size=args.max_samples, replace=False)
    rng.shuffle(idx)
    return idx


def build(args):
    paths = [args.original, args.targeted]
    parts = [_load(path, idx) for idx, path in enumerate(paths) if path and Path(path).exists()]
    if not parts:
        raise FileNotFoundError("No input datasets found.")
    data = _concat(parts)
    idx = _balanced_indices(data, args)
    output = {key: value[idx] for key, value in data.items()}
    train_split = np.zeros(len(idx), dtype=np.int8)
    rng = np.random.default_rng(args.seed + 1)
    order = np.arange(len(idx))
    rng.shuffle(order)
    train_n = int(len(order) * (1.0 - args.val_fraction))
    train_split[order[:train_n]] = 1
    output["train_split"] = train_split

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output)
    stats = {
        "samples": int(len(idx)),
        "train_samples": int(train_split.sum()),
        "val_samples": int(len(idx) - train_split.sum()),
        "source_distribution": {str(int(k)): int((output["source_ids"] == k).sum()) for k in np.unique(output["source_ids"])},
        "mode_distribution": {str(int(k)): int((output["mode_ids"] == k).sum()) for k in np.unique(output["mode_ids"])},
    }
    for key in ("death_within_7", "escaped_blast", "own_bomb_recent", "has_escape_after_bomb", "trapped_if_bomb", "in_future_blast"):
        if key in output:
            stats[f"{key}_rate"] = float(output[key].astype(np.float32).mean())
    stats["would_destroy_boxes_mean"] = float(output["would_destroy_boxes_if_bomb"].mean())
    path.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Merge and balance curriculum auxiliary datasets.")
    parser.add_argument("--original", default="ml/datasets/curriculum_rollouts_stochastic.npz")
    parser.add_argument("--targeted", default="ml/datasets/curriculum_aux_targeted.npz")
    parser.add_argument("--output", default="ml/datasets/curriculum_aux_balanced.npz")
    parser.add_argument("--target_positive_per_label", type=int, default=3000)
    parser.add_argument("--target_counterfactual_per_label", type=int, default=3000)
    parser.add_argument("--negative_per_label", type=int, default=2000)
    parser.add_argument("--max_samples", type=int, default=50_000)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7200)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
