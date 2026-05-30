from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PLACE_BOMB = 5
MOVE_ACTIONS = {1, 2, 3, 4}


def _has(data, key):
    return key in data.files


def _bomb_context_ids(actions):
    is_bomb = actions == PLACE_BOMB
    out = np.zeros(len(actions), dtype=np.int32)
    out[is_bomb] = np.arange(1, int(is_bomb.sum()) + 1, dtype=np.int32)
    return out


def _escape_counts(actions, context_ids, deltas, survived, max_delta):
    counts = {}
    mask = np.isin(actions, list(MOVE_ACTIONS)) & survived & (deltas >= 1) & (deltas <= max_delta)
    for context_id in context_ids[mask]:
        counts[int(context_id)] = counts.get(int(context_id), 0) + 1
    return counts


def build(args):
    data = np.load(args.input)
    actions = data["actions"].astype(np.int64)
    observations = data["observations"].astype(np.float32)
    survived = data["teacher_survived_after_bomb"].astype(bool)
    boxes = data["boxes_destroyed_after_bomb"].astype(np.int16)
    current_danger = data["current_danger"].astype(np.int16) if _has(data, "current_danger") else np.full(len(actions), 9999, dtype=np.int16)
    future_danger = data["future_danger"].astype(np.int16) if _has(data, "future_danger") else np.full(len(actions), 9999, dtype=np.int16)
    deltas = data["action_after_bomb_step_delta"].astype(np.int16)
    sample_context_ids = data["bomb_context_id"].astype(np.int32)
    bomb_context_ids = _bomb_context_ids(actions)
    escape_count_by_context = _escape_counts(actions, sample_context_ids, deltas, survived, args.escape_window)

    bomb_indices = np.flatnonzero(actions == PLACE_BOMB)
    context_escape_counts = np.asarray([escape_count_by_context.get(int(bomb_context_ids[i]), 0) for i in bomb_indices], dtype=np.int16)
    safe_now = current_danger[bomb_indices] >= args.min_current_danger
    future_safe = (future_danger[bomb_indices] >= args.min_future_danger) | (future_danger[bomb_indices] == 9999)
    positive = survived[bomb_indices] & (boxes[bomb_indices] > 0) & safe_now & future_safe & (context_escape_counts > 0)
    negative = (~survived[bomb_indices]) | (boxes[bomb_indices] <= 0) | (~safe_now) | (~future_safe) | (context_escape_counts <= 0)

    pos_idx = bomb_indices[positive]
    neg_idx = bomb_indices[negative]
    rng = np.random.default_rng(args.seed)
    non_bomb_candidates = np.flatnonzero(actions != PLACE_BOMB)
    hard_count = min(len(non_bomb_candidates), max(args.min_hard_negatives, len(pos_idx)))
    hard_idx = rng.choice(non_bomb_candidates, size=hard_count, replace=False) if hard_count else np.zeros(0, dtype=np.int64)

    selected_idx = np.concatenate([pos_idx, neg_idx, hard_idx])
    labels = np.concatenate([
        np.ones(len(pos_idx), dtype=np.float32),
        np.zeros(len(neg_idx), dtype=np.float32),
        np.zeros(len(hard_idx), dtype=np.float32),
    ])
    source = np.concatenate([
        np.full(len(pos_idx), 1, dtype=np.int8),
        np.full(len(neg_idx), 2, dtype=np.int8),
        np.full(len(hard_idx), 3, dtype=np.int8),
    ])
    order = rng.permutation(len(selected_idx))
    selected_idx = selected_idx[order]
    labels = labels[order]
    source = source[order]

    scalar_features = np.stack([
        current_danger[selected_idx].astype(np.float32) / 9999.0,
        future_danger[selected_idx].astype(np.float32) / 9999.0,
        boxes[selected_idx].astype(np.float32) / 7.0,
        data["step"][selected_idx].astype(np.float32) / 500.0 if _has(data, "step") else np.zeros(len(selected_idx), dtype=np.float32),
        np.asarray([escape_count_by_context.get(int(bomb_context_ids[i] if actions[i] == PLACE_BOMB else sample_context_ids[i]), 0) for i in selected_idx], dtype=np.float32) / args.escape_window,
    ], axis=1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=observations[selected_idx],
        scalar_features=scalar_features.astype(np.float32),
        labels=labels.astype(np.float32),
        source=source,
        original_index=selected_idx.astype(np.int64),
        action=actions[selected_idx].astype(np.int64),
        boxes_destroyed_after_bomb=boxes[selected_idx].astype(np.int16),
        current_danger=current_danger[selected_idx].astype(np.int16),
        future_danger=future_danger[selected_idx].astype(np.int16),
    )
    stats = {
        "input": args.input,
        "output": str(output),
        "samples": int(len(selected_idx)),
        "positive_count": int(labels.sum()),
        "negative_count": int((labels == 0).sum()),
        "positive_fraction": float(labels.mean()),
        "bomb_positive_candidates": int(len(pos_idx)),
        "bomb_negative_candidates": int(len(neg_idx)),
        "hard_negative_count": int(len(hard_idx)),
        "positive_boxes_mean": float(np.mean(boxes[pos_idx])) if len(pos_idx) else 0.0,
        "positive_current_danger_mean": float(np.mean(current_danger[pos_idx])) if len(pos_idx) else 0.0,
    }
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build offline bomb context value dataset.")
    parser.add_argument("--input", default="ml/datasets/rl_bc_bomb_escape.npz")
    parser.add_argument("--output", default="ml/datasets/bomb_value_dataset.npz")
    parser.add_argument("--escape_window", type=int, default=5)
    parser.add_argument("--min_current_danger", type=int, default=3)
    parser.add_argument("--min_future_danger", type=int, default=1)
    parser.add_argument("--min_hard_negatives", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
