from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PLACE_BOMB = 5
STOP = 0
MOVE_ACTIONS = {1, 2, 3, 4}


def _has(data, key):
    return key in data.files


def filter_dataset(args):
    data = np.load(args.input)
    actions = data["actions"].astype(np.int64)
    post_escape = data["is_post_bomb_escape"].astype(bool) if _has(data, "is_post_bomb_escape") else data["post_bomb_escape"].astype(bool)
    survived = data["teacher_survived_after_bomb"].astype(bool) if _has(data, "teacher_survived_after_bomb") else np.ones(len(actions), dtype=bool)
    mask = post_escape & survived & (actions != PLACE_BOMB)
    if args.moves_only:
        mask &= np.isin(actions, list(MOVE_ACTIONS))
    if args.exclude_danger_stop and _has(data, "current_danger"):
        current_danger = data["current_danger"].astype(np.int16)
        mask &= ~((actions == STOP) & (current_danger <= args.danger_threshold))

    if not np.any(mask):
        raise ValueError(f"No successful escape samples selected from {args.input}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key in data.files:
        value = data[key]
        if value.shape[:1] == (len(actions),):
            arrays[key] = value[mask]
        else:
            arrays[key] = value
    arrays["is_post_bomb_escape"] = np.ones(int(mask.sum()), dtype=np.int8)
    arrays["post_bomb_escape"] = np.ones(int(mask.sum()), dtype=np.int8)
    arrays["is_bomb_action"] = np.zeros(int(mask.sum()), dtype=np.int8)
    arrays["did_place_bomb"] = np.zeros(int(mask.sum()), dtype=np.int8)
    arrays["sample_weight"] = np.ones(int(mask.sum()), dtype=np.float32)
    np.savez_compressed(output, **arrays)

    kept_actions = actions[mask]
    stats = {
        "input": args.input,
        "output": str(output),
        "input_samples": int(len(actions)),
        "samples_kept": int(mask.sum()),
        "kept_fraction": float(mask.mean()),
        "action_distribution": {str(i): int(v) for i, v in enumerate(np.bincount(kept_actions, minlength=6))},
    }
    if _has(data, "action_after_bomb_step_delta"):
        stats["avg_action_after_bomb_step_delta"] = float(np.mean(data["action_after_bomb_step_delta"][mask]))
    if _has(data, "current_danger"):
        stats["current_danger_mean"] = float(np.mean(data["current_danger"][mask]))
        stats["current_danger_min"] = int(np.min(data["current_danger"][mask]))
        stats["current_danger_max"] = int(np.max(data["current_danger"][mask]))
    if _has(data, "future_danger"):
        stats["future_danger_mean"] = float(np.mean(data["future_danger"][mask]))
    if _has(data, "boxes_destroyed_after_bomb"):
        boxes = data["boxes_destroyed_after_bomb"][mask]
        stats["boxes_destroyed_after_bomb_mean"] = float(np.mean(boxes))
        stats["boxes_destroyed_after_bomb_positive_fraction"] = float((boxes > 0).mean())
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Keep only successful post-bomb escape BC samples.")
    parser.add_argument("--input", default="ml/datasets/rl_bc_bomb_escape.npz")
    parser.add_argument("--output", default="ml/datasets/rl_bc_successful_escape_only.npz")
    parser.add_argument("--moves_only", action="store_true", default=True)
    parser.add_argument("--include_stop", action="store_false", dest="moves_only")
    parser.add_argument("--exclude_danger_stop", action="store_true")
    parser.add_argument("--danger_threshold", type=int, default=2)
    args = parser.parse_args()
    filter_dataset(args)


if __name__ == "__main__":
    main()
