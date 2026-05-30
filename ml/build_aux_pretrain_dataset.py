from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_OUTPUT_KEYS = (
    "observations",
    "death_within_7",
    "escaped_blast",
    "has_escape_path_now",
    "has_escape_after_bomb",
    "trapped_if_bomb",
    "in_future_blast",
    "safe_tiles_after_bomb_count",
    "blast_corridor_distance",
    "boxes_destroyed_future",
    "reachable_area_delta",
    "discounted_returns",
    "scenario_types",
)

SCENARIO_NORMAL_CURRICULUM = 10


def _take(data: np.lib.npyio.NpzFile, key: str, n: int, default=0.0, dtype=np.float32) -> np.ndarray:
    if key in data:
        return data[key].astype(dtype)
    return np.full(n, default, dtype=dtype)


def _load_aux_dataset(path: str, source_id: int, max_samples: int, seed: int) -> dict:
    data = np.load(path)
    n = len(data["observations"])
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if max_samples > 0 and len(idx) > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    out = {
        "observations": data["observations"][idx].astype(np.float32),
        "death_within_7": _take(data, "death_within_7", n)[idx],
        "escaped_blast": _take(data, "escaped_blast", n)[idx],
        "has_escape_path_now": _take(data, "has_escape_path_now", n)[idx],
        "has_escape_after_bomb": _take(data, "has_escape_after_bomb", n)[idx],
        "trapped_if_bomb": _take(data, "trapped_if_bomb", n)[idx],
        "in_future_blast": _take(data, "in_future_blast", n)[idx],
        "safe_tiles_after_bomb_count": _take(data, "safe_tiles_after_bomb_count", n)[idx],
        "blast_corridor_distance": _take(data, "blast_corridor_distance", n, default=-1.0)[idx],
        "boxes_destroyed_future": _take(data, "boxes_destroyed_future", n)[idx],
        "reachable_area_delta": _take(data, "reachable_area_delta", n)[idx],
        "discounted_returns": _take(data, "discounted_returns", n)[idx],
        "scenario_types": np.full(len(idx), SCENARIO_NORMAL_CURRICULUM, dtype=np.int64),
        "source_ids": np.full(len(idx), source_id, dtype=np.int64),
    }
    return out


def _load_guided_dataset(path: str, source_id: int, max_samples: int, seed: int) -> dict:
    data = np.load(path)
    n = len(data["observations"])
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if max_samples > 0 and len(idx) > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    boxes = _take(data, "would_destroy_boxes_if_bomb", n)[idx]
    safe_tiles = _take(data, "safe_tiles_after_bomb_count", n)[idx]
    out = {
        "observations": data["observations"][idx].astype(np.float32),
        "death_within_7": (data["death_prob"][idx] >= 0.7).astype(np.float32) if "death_prob" in data else np.zeros(len(idx), dtype=np.float32),
        "escaped_blast": np.zeros(len(idx), dtype=np.float32),
        "has_escape_path_now": _take(data, "has_escape_path_now", n)[idx],
        "has_escape_after_bomb": _take(data, "has_escape_after_bomb", n)[idx],
        "trapped_if_bomb": _take(data, "trapped_if_bomb", n)[idx],
        "in_future_blast": _take(data, "in_future_blast", n)[idx],
        "safe_tiles_after_bomb_count": safe_tiles,
        "blast_corridor_distance": _take(data, "blast_corridor_distance", n, default=-1.0)[idx],
        "boxes_destroyed_future": boxes,
        "reachable_area_delta": np.zeros(len(idx), dtype=np.float32),
        # Guided scenarios do not have observed returns; keep neutral so critic pretraining is dominated by rollout data.
        "discounted_returns": np.zeros(len(idx), dtype=np.float32),
        "scenario_types": data["scenario_types"][idx].astype(np.int64),
        "source_ids": np.full(len(idx), source_id, dtype=np.int64),
    }
    return out


def _concat(parts: list[dict]) -> dict:
    keys = REQUIRED_OUTPUT_KEYS + ("source_ids",)
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def _balanced_indices(data: dict, args) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    indices = [np.arange(len(data["observations"]))]
    for key, target in (
        ("death_within_7", args.target_per_binary),
        ("has_escape_path_now", args.target_per_binary),
        ("has_escape_after_bomb", args.target_per_binary),
        ("trapped_if_bomb", args.target_per_binary),
        ("in_future_blast", args.target_per_binary),
    ):
        labels = data[key].astype(np.float32)
        pos = np.flatnonzero(labels > 0.5)
        neg = np.flatnonzero(labels <= 0.5)
        if len(pos):
            indices.append(rng.choice(pos, size=min(target, max(target, len(pos))), replace=len(pos) < target))
        if len(neg):
            indices.append(rng.choice(neg, size=min(target, len(neg)), replace=False))
    for scenario in np.unique(data["scenario_types"]):
        scenario_idx = np.flatnonzero(data["scenario_types"] == scenario)
        if len(scenario_idx):
            indices.append(rng.choice(scenario_idx, size=min(args.target_per_scenario, len(scenario_idx)), replace=False))
    idx = np.concatenate(indices)
    if args.max_samples > 0 and len(idx) > args.max_samples:
        idx = rng.choice(idx, size=args.max_samples, replace=False)
    rng.shuffle(idx)
    return idx


def build(args):
    parts = []
    if args.curriculum_dataset and Path(args.curriculum_dataset).exists():
        parts.append(_load_aux_dataset(args.curriculum_dataset, source_id=0, max_samples=args.max_curriculum_samples, seed=args.seed))
    if args.scenario_dataset and Path(args.scenario_dataset).exists():
        parts.append(_load_guided_dataset(args.scenario_dataset, source_id=1, max_samples=args.max_scenario_samples, seed=args.seed + 1))
    if not parts:
        raise FileNotFoundError("No input datasets found.")
    data = _concat(parts)
    idx = _balanced_indices(data, args)
    output = {key: value[idx] for key, value in data.items()}

    rng = np.random.default_rng(args.seed + 2)
    order = np.arange(len(idx))
    rng.shuffle(order)
    train_split = np.zeros(len(idx), dtype=np.int8)
    train_split[order[: int(len(order) * (1.0 - args.val_fraction))]] = 1
    output["train_split"] = train_split

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output)
    stats = {
        "samples": int(len(idx)),
        "train_samples": int(train_split.sum()),
        "val_samples": int(len(idx) - train_split.sum()),
        "source_distribution": {str(int(k)): int((output["source_ids"] == k).sum()) for k in np.unique(output["source_ids"])},
        "scenario_distribution": {str(int(k)): int((output["scenario_types"] == k).sum()) for k in np.unique(output["scenario_types"])},
    }
    for key in (
        "death_within_7",
        "escaped_blast",
        "has_escape_path_now",
        "has_escape_after_bomb",
        "trapped_if_bomb",
        "in_future_blast",
    ):
        stats[f"{key}_rate"] = float(output[key].astype(np.float32).mean())
    stats["safe_tiles_after_bomb_mean"] = float(output["safe_tiles_after_bomb_count"].astype(np.float32).mean())
    stats["boxes_destroyed_future_mean"] = float(output["boxes_destroyed_future"].astype(np.float32).mean())
    path.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build aux/critic pretraining dataset from v3 labels and aux-guided scenarios.")
    parser.add_argument("--curriculum_dataset", default="ml/datasets/curriculum_aux_balanced_v3.npz")
    parser.add_argument("--scenario_dataset", default="ml/datasets/aux_guided_scenarios.npz")
    parser.add_argument("--output", default="ml/datasets/aux_pretrain_dataset_v3.npz")
    parser.add_argument("--max_curriculum_samples", type=int, default=30000)
    parser.add_argument("--max_scenario_samples", type=int, default=20000)
    parser.add_argument("--target_per_binary", type=int, default=4000)
    parser.add_argument("--target_per_scenario", type=int, default=3000)
    parser.add_argument("--max_samples", type=int, default=60000)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7700)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
