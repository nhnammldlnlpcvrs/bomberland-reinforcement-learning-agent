from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PLACE_BOMB = 5


def _has(data, key):
    return key in data.files


def filter_dataset(args):
    data = np.load(args.input)
    actions = data["actions"].astype(np.int64)
    is_bomb = data["is_bomb_action"].astype(bool) if _has(data, "is_bomb_action") else actions == PLACE_BOMB
    survived = data["teacher_survived_after_bomb"].astype(bool) if _has(data, "teacher_survived_after_bomb") else np.ones(len(actions), dtype=bool)
    boxes = data["boxes_destroyed_after_bomb"].astype(np.int16) if _has(data, "boxes_destroyed_after_bomb") else np.zeros(len(actions), dtype=np.int16)
    reward_proxy = data["reward_proxy"].astype(np.float32) if _has(data, "reward_proxy") else np.zeros(len(actions), dtype=np.float32)
    mask = is_bomb & survived & ((boxes > 0) | (reward_proxy > args.min_reward_proxy))
    if args.require_current_safe and _has(data, "current_danger"):
        mask &= data["current_danger"].astype(np.int16) >= args.min_current_danger
    if args.max_future_danger > 0 and _has(data, "future_danger"):
        future = data["future_danger"].astype(np.int16)
        mask &= (future >= args.max_future_danger) | (future == 9999)

    if not np.any(mask):
        raise ValueError(f"No useful-safe bomb samples selected from {args.input}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "observations": data["observations"][mask].astype(np.float32),
        "actions": np.full(int(mask.sum()), PLACE_BOMB, dtype=np.int64),
        "is_bomb_action": np.ones(int(mask.sum()), dtype=np.int8),
        "did_place_bomb": np.ones(int(mask.sum()), dtype=np.int8),
        "sample_weight": np.ones(int(mask.sum()), dtype=np.float32),
    }
    for key in (
        "teacher_survived_after_bomb",
        "boxes_destroyed_after_bomb",
        "current_danger",
        "future_danger",
        "step",
        "bomb_context_id",
        "reward_proxy",
    ):
        if _has(data, key):
            arrays[key] = data[key][mask]
    np.savez_compressed(output, **arrays)

    kept_steps = data["step"][mask] if _has(data, "step") else np.zeros(int(mask.sum()), dtype=np.int16)
    stats = {
        "input": args.input,
        "output": str(output),
        "input_samples": int(len(actions)),
        "samples_kept": int(mask.sum()),
        "kept_fraction": float(mask.mean()),
        "boxes_destroyed_mean": float(np.mean(boxes[mask])),
        "boxes_destroyed_distribution": {str(i): int(v) for i, v in enumerate(np.bincount(boxes[mask].astype(np.int64)))},
        "step_mean": float(np.mean(kept_steps)),
        "step_min": int(np.min(kept_steps)),
        "step_max": int(np.max(kept_steps)),
    }
    if _has(data, "current_danger"):
        current = data["current_danger"][mask]
        stats["current_danger_mean"] = float(np.mean(current))
        stats["current_danger_min"] = int(np.min(current))
        stats["current_danger_max"] = int(np.max(current))
    if _has(data, "future_danger"):
        future = data["future_danger"][mask]
        stats["future_danger_mean"] = float(np.mean(future))
        stats["future_danger_min"] = int(np.min(future))
        stats["future_danger_max"] = int(np.max(future))
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Filter teacher bomb samples to useful-safe bomb contexts.")
    parser.add_argument("--input", default="ml/datasets/rl_bc_bomb_escape.npz")
    parser.add_argument("--output", default="ml/datasets/rl_bc_useful_safe_bomb_contexts.npz")
    parser.add_argument("--min_reward_proxy", type=float, default=0.0)
    parser.add_argument("--require_current_safe", action="store_true", default=True)
    parser.add_argument("--allow_current_danger", action="store_false", dest="require_current_safe")
    parser.add_argument("--min_current_danger", type=int, default=3)
    parser.add_argument("--max_future_danger", type=int, default=0)
    args = parser.parse_args()
    filter_dataset(args)


if __name__ == "__main__":
    main()
