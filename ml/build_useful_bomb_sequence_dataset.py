from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PLACE_BOMB = 5
MOVE_ACTIONS = {1, 2, 3, 4}


def _has(data, key):
    return key in data.files


def build(args):
    data = np.load(args.input)
    actions = data["actions"].astype(np.int64)
    is_bomb = data["is_bomb_action"].astype(bool) if _has(data, "is_bomb_action") else actions == PLACE_BOMB
    survived = data["teacher_survived_after_bomb"].astype(bool)
    boxes = data["boxes_destroyed_after_bomb"].astype(np.int16)
    current_danger = data["current_danger"].astype(np.int16) if _has(data, "current_danger") else np.full(len(actions), 9999, dtype=np.int16)
    future_danger = data["future_danger"].astype(np.int16) if _has(data, "future_danger") else np.full(len(actions), 9999, dtype=np.int16)
    context_ids = data["bomb_context_id"].astype(np.int32)
    deltas = data["action_after_bomb_step_delta"].astype(np.int16)

    bomb_context_for_sample = np.zeros(len(actions), dtype=np.int32)
    bomb_context_for_sample[is_bomb] = np.arange(1, int(is_bomb.sum()) + 1, dtype=np.int32)

    bomb_mask = is_bomb & survived & (boxes > 0)
    if args.require_current_safe:
        bomb_mask &= current_danger >= args.min_current_danger
    bomb_indices = np.flatnonzero(bomb_mask)

    escape_indices = []
    escape_sequence_ids = []
    bomb_indices_kept = []
    sequence_ids = []
    for bomb_idx in bomb_indices:
        context_id = int(bomb_context_for_sample[bomb_idx])
        seq_mask = (
            (context_ids == context_id)
            & np.isin(actions, list(MOVE_ACTIONS))
            & (deltas >= 1)
            & (deltas <= args.escape_window)
            & survived
        )
        seq_indices = np.flatnonzero(seq_mask)
        if len(seq_indices) < args.min_escape_steps:
            continue
        seq_id = len(sequence_ids) + 1
        sequence_ids.append(seq_id)
        bomb_indices_kept.append(bomb_idx)
        escape_indices.extend(seq_indices.tolist())
        escape_sequence_ids.extend([seq_id] * len(seq_indices))

    if not bomb_indices_kept:
        raise ValueError("No useful bomb sequences selected")

    bomb_indices_kept = np.asarray(bomb_indices_kept, dtype=np.int64)
    escape_indices = np.asarray(escape_indices, dtype=np.int64)
    escape_sequence_ids = np.asarray(escape_sequence_ids, dtype=np.int32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        bomb_obs=data["observations"][bomb_indices_kept].astype(np.float32),
        bomb_action=np.full(len(bomb_indices_kept), PLACE_BOMB, dtype=np.int64),
        escape_obs=data["observations"][escape_indices].astype(np.float32),
        escape_action=actions[escape_indices].astype(np.int64),
        bomb_sequence_id=np.asarray(sequence_ids, dtype=np.int32),
        escape_sequence_id=escape_sequence_ids,
        step_delta=deltas[escape_indices].astype(np.int16),
        boxes_destroyed_after_bomb=boxes[bomb_indices_kept].astype(np.int16),
        bomb_current_danger=current_danger[bomb_indices_kept].astype(np.int16),
        bomb_future_danger=future_danger[bomb_indices_kept].astype(np.int16),
        escape_current_danger=current_danger[escape_indices].astype(np.int16),
        escape_future_danger=future_danger[escape_indices].astype(np.int16),
    )

    stats = {
        "input": args.input,
        "output": str(output),
        "sequences": int(len(sequence_ids)),
        "bomb_context_count": int(len(bomb_indices_kept)),
        "escape_sample_count": int(len(escape_indices)),
        "escape_action_distribution": {str(i): int(v) for i, v in enumerate(np.bincount(actions[escape_indices], minlength=6))},
        "boxes_destroyed_mean": float(np.mean(boxes[bomb_indices_kept])),
        "boxes_destroyed_distribution": {str(i): int(v) for i, v in enumerate(np.bincount(boxes[bomb_indices_kept].astype(np.int64)))},
        "step_delta_distribution": {str(i): int(v) for i, v in enumerate(np.bincount(deltas[escape_indices].astype(np.int64), minlength=args.escape_window + 1))},
        "bomb_current_danger_mean": float(np.mean(current_danger[bomb_indices_kept])),
        "bomb_future_danger_mean": float(np.mean(future_danger[bomb_indices_kept])),
    }
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build useful bomb + post-bomb escape sequence dataset.")
    parser.add_argument("--input", default="ml/datasets/rl_bc_bomb_escape.npz")
    parser.add_argument("--output", default="ml/datasets/rl_bc_useful_bomb_sequences.npz")
    parser.add_argument("--escape_window", type=int, default=5)
    parser.add_argument("--min_escape_steps", type=int, default=1)
    parser.add_argument("--require_current_safe", action="store_true", default=True)
    parser.add_argument("--allow_current_danger", action="store_false", dest="require_current_safe")
    parser.add_argument("--min_current_danger", type=int, default=3)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
