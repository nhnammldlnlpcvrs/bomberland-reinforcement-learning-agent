from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PLACE_BOMB = 5
STOP = 0
MOVE_ACTIONS = {1, 2, 3, 4}


def _stack_frames(frames: list[np.ndarray], frame_stack: int) -> np.ndarray:
    if not frames:
        raise ValueError("Cannot stack an empty frame list.")
    selected = list(frames[-frame_stack:])
    while len(selected) < frame_stack:
        selected.insert(0, selected[0])
    return np.concatenate(selected, axis=0).astype(np.float32)


def build(args):
    data = np.load(args.input)
    frame_stack = max(1, int(args.frame_stack))
    bomb_obs = data["bomb_obs"].astype(np.float32)
    bomb_action = data["bomb_action"].astype(np.int64) if "bomb_action" in data else np.full(len(bomb_obs), PLACE_BOMB, dtype=np.int64)
    escape_obs = data["escape_obs"].astype(np.float32)
    escape_action = data["escape_action"].astype(np.int64)
    bomb_seq = data["bomb_sequence_id"].astype(np.int64)
    escape_seq = data["escape_sequence_id"].astype(np.int64)
    step_delta = data["step_delta"].astype(np.int64) if "step_delta" in data else np.ones(len(escape_obs), dtype=np.int64)
    boxes = data["boxes_destroyed_after_bomb"].astype(np.float32) if "boxes_destroyed_after_bomb" in data else np.zeros(len(bomb_obs), dtype=np.float32)

    bomb_by_seq = {int(seq): idx for idx, seq in enumerate(bomb_seq)}
    escape_indices_by_seq: dict[int, list[int]] = {}
    for idx, seq in enumerate(escape_seq):
        escape_indices_by_seq.setdefault(int(seq), []).append(idx)
    for seq in escape_indices_by_seq:
        escape_indices_by_seq[seq].sort(key=lambda idx: int(step_delta[idx]))

    observations = []
    actions = []
    sample_weight = []
    is_bomb = []
    is_escape = []
    sequence_ids = []
    deltas = []
    box_values = []

    for seq, bomb_idx in bomb_by_seq.items():
        if bomb_action[bomb_idx] == PLACE_BOMB and boxes[bomb_idx] >= args.min_boxes:
            observations.append(_stack_frames([bomb_obs[bomb_idx]], frame_stack))
            actions.append(PLACE_BOMB)
            sample_weight.append(args.bomb_weight)
            is_bomb.append(1)
            is_escape.append(0)
            sequence_ids.append(seq)
            deltas.append(0)
            box_values.append(float(boxes[bomb_idx]))

        history = [bomb_obs[bomb_idx]]
        for esc_idx in escape_indices_by_seq.get(seq, []):
            action = int(escape_action[esc_idx])
            if args.exclude_stop_escape and action == STOP:
                history.append(escape_obs[esc_idx])
                continue
            if action not in MOVE_ACTIONS:
                history.append(escape_obs[esc_idx])
                continue
            observations.append(_stack_frames(history + [escape_obs[esc_idx]], frame_stack))
            actions.append(action)
            sample_weight.append(args.escape_weight)
            is_bomb.append(0)
            is_escape.append(1)
            sequence_ids.append(seq)
            deltas.append(int(step_delta[esc_idx]))
            box_values.append(float(boxes[bomb_idx]))
            history.append(escape_obs[esc_idx])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "sample_weight": np.asarray(sample_weight, dtype=np.float32),
        "is_bomb_action": np.asarray(is_bomb, dtype=np.int8),
        "is_post_bomb_escape": np.asarray(is_escape, dtype=np.int8),
        "sequence_id": np.asarray(sequence_ids, dtype=np.int64),
        "step_delta": np.asarray(deltas, dtype=np.int64),
        "boxes_destroyed_after_bomb": np.asarray(box_values, dtype=np.float32),
    }
    np.savez_compressed(output, **arrays)
    action_counts = np.bincount(arrays["actions"], minlength=6)
    stats = {
        "input": args.input,
        "output": args.output,
        "frame_stack": frame_stack,
        "samples": int(len(arrays["actions"])),
        "bomb_samples": int(arrays["is_bomb_action"].sum()),
        "escape_samples": int(arrays["is_post_bomb_escape"].sum()),
        "action_distribution": {str(i): int(v) for i, v in enumerate(action_counts)},
        "boxes_mean": float(arrays["boxes_destroyed_after_bomb"].mean()) if len(arrays["actions"]) else 0.0,
        "bomb_weight": float(args.bomb_weight),
        "escape_weight": float(args.escape_weight),
        "source": Path(args.input).name,
    }
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build frame-stacked BC dataset from safe/useful bomb sequence data.")
    parser.add_argument("--input", default="ml/datasets/rl_bc_selected_bomb_sequences_v3.npz")
    parser.add_argument("--output", default="ml/datasets/fs4_selected_bomb_bc.npz")
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--bomb_weight", type=float, default=0.02)
    parser.add_argument("--escape_weight", type=float, default=8.0)
    parser.add_argument("--min_boxes", type=float, default=0.0)
    parser.add_argument("--exclude_stop_escape", action="store_true", default=True)
    parser.add_argument("--include_stop_escape", action="store_false", dest="exclude_stop_escape")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
