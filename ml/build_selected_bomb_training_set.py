from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ml.train_bomb_value_model import BombValueNet

PLACE_BOMB = 5
MOVE_ACTIONS = {1, 2, 3, 4}


def _bomb_context_ids(actions):
    is_bomb = actions == PLACE_BOMB
    out = np.zeros(len(actions), dtype=np.int32)
    out[is_bomb] = np.arange(1, int(is_bomb.sum()) + 1, dtype=np.int32)
    return out


def build(args):
    data = np.load(args.input)
    actions = data["actions"].astype(np.int64)
    observations = data["observations"].astype(np.float32)
    current_danger = data["current_danger"].astype(np.int16)
    future_danger = data["future_danger"].astype(np.int16)
    boxes = data["boxes_destroyed_after_bomb"].astype(np.int16)
    survived = data["teacher_survived_after_bomb"].astype(bool)
    deltas = data["action_after_bomb_step_delta"].astype(np.int16)
    context_ids = data["bomb_context_id"].astype(np.int32)
    bomb_context_ids = _bomb_context_ids(actions)
    bomb_indices = np.flatnonzero(actions == PLACE_BOMB)
    scalar_features = np.stack([
        current_danger[bomb_indices].astype(np.float32) / 9999.0,
        future_danger[bomb_indices].astype(np.float32) / 9999.0,
        boxes[bomb_indices].astype(np.float32) / 7.0,
        data["step"][bomb_indices].astype(np.float32) / 500.0,
        np.ones(len(bomb_indices), dtype=np.float32),
    ], axis=1)

    checkpoint = torch.load(args.model, map_location=args.device)
    model = BombValueNet(checkpoint.get("in_channels", observations.shape[1]), checkpoint.get("scalar_dim", scalar_features.shape[1]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device)
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(bomb_indices), 1024):
            obs = torch.as_tensor(observations[bomb_indices[start:start + 1024]], dtype=torch.float32, device=args.device)
            scalars = torch.as_tensor(scalar_features[start:start + 1024], dtype=torch.float32, device=args.device)
            probs.append(torch.sigmoid(model(obs, scalars)).cpu().numpy())
    probs = np.concatenate(probs)
    threshold = args.threshold if args.threshold is not None else float(checkpoint.get("threshold", 0.8))
    candidate_bombs = bomb_indices[probs >= threshold]
    candidate_probs = probs[probs >= threshold]

    bomb_kept = []
    escape_indices = []
    escape_sequence_ids = []
    for bomb_idx, prob in zip(candidate_bombs, candidate_probs):
        context_id = int(bomb_context_ids[bomb_idx])
        seq_mask = (
            (context_ids == context_id)
            & np.isin(actions, list(MOVE_ACTIONS))
            & survived
            & (deltas >= 1)
            & (deltas <= args.escape_window)
        )
        seq_idx = np.flatnonzero(seq_mask)
        if len(seq_idx) < args.min_escape_steps:
            continue
        if args.require_positive_box and boxes[bomb_idx] <= 0:
            continue
        seq_id = len(bomb_kept) + 1
        bomb_kept.append(bomb_idx)
        escape_indices.extend(seq_idx.tolist())
        escape_sequence_ids.extend([seq_id] * len(seq_idx))

    if not bomb_kept:
        raise ValueError("Selector produced no training sequences")
    bomb_kept = np.asarray(bomb_kept, dtype=np.int64)
    escape_indices = np.asarray(escape_indices, dtype=np.int64)
    escape_sequence_ids = np.asarray(escape_sequence_ids, dtype=np.int32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        bomb_obs=observations[bomb_kept],
        bomb_action=np.full(len(bomb_kept), PLACE_BOMB, dtype=np.int64),
        escape_obs=observations[escape_indices],
        escape_action=actions[escape_indices].astype(np.int64),
        bomb_sequence_id=np.arange(1, len(bomb_kept) + 1, dtype=np.int32),
        escape_sequence_id=escape_sequence_ids,
        step_delta=deltas[escape_indices].astype(np.int16),
        boxes_destroyed_after_bomb=boxes[bomb_kept].astype(np.int16),
        selector_prob=probs[np.searchsorted(bomb_indices, bomb_kept)].astype(np.float32),
    )
    stats = {
        "input": args.input,
        "model": args.model,
        "output": str(output),
        "threshold": float(threshold),
        "selected_bomb_contexts": int(len(bomb_kept)),
        "escape_sample_count": int(len(escape_indices)),
        "escape_action_distribution": {str(i): int(v) for i, v in enumerate(np.bincount(actions[escape_indices], minlength=6))},
        "boxes_destroyed_mean": float(np.mean(boxes[bomb_kept])),
        "selector_prob_mean": float(np.mean(probs[np.searchsorted(bomb_indices, bomb_kept)])),
    }
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="ml/datasets/rl_bc_bomb_escape.npz")
    parser.add_argument("--model", default="ml/checkpoints/rl_agent_pure/bomb_value_model.pt")
    parser.add_argument("--output", default="ml/datasets/rl_bc_selected_bomb_sequences.npz")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--escape_window", type=int, default=5)
    parser.add_argument("--min_escape_steps", type=int, default=1)
    parser.add_argument("--require_positive_box", action="store_true", default=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
