from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TYPE_SAFE_BOMB = 1
TYPE_UNSAFE = 3


def _pad_single_obs(obs: np.ndarray, seq_len: int, channels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq = np.zeros((len(obs), seq_len, channels, 13, 13), dtype=np.float32)
    actions = np.zeros((len(obs), seq_len), dtype=np.int64)
    mask = np.zeros((len(obs), seq_len), dtype=bool)
    seq[:, 0] = obs.astype(np.float32)
    mask[:, 0] = True
    return seq, actions, mask


def _nearest_negative_pairs(pos_boxes: np.ndarray, neg_boxes: np.ndarray, pairs_per_positive: int) -> tuple[np.ndarray, np.ndarray]:
    if len(pos_boxes) == 0 or len(neg_boxes) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pos_ids = []
    neg_ids = []
    for pos_idx, value in enumerate(pos_boxes):
        order = np.argsort(np.abs(neg_boxes.astype(np.float32) - float(value)))
        for neg_idx in order[:pairs_per_positive]:
            pos_ids.append(pos_idx)
            neg_ids.append(int(neg_idx))
    return np.asarray(pos_ids, dtype=np.int64), np.asarray(neg_ids, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pairwise ranking data for modular bomb-head calibration.")
    parser.add_argument("--balanced_dataset", default="ml/datasets/recurrent_bc_balanced_sequences.npz")
    parser.add_argument("--counterfactual_negatives", default="ml/datasets/bomb_counterfactual_negatives.npz")
    parser.add_argument("--output", default="ml/datasets/recurrent_bomb_ranking_dataset.npz")
    parser.add_argument("--pairs_per_positive", type=int, default=8)
    parser.add_argument("--max_counterfactual_negatives", type=int, default=720)
    parser.add_argument("--seed", type=int, default=9980)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    base = np.load(args.balanced_dataset, allow_pickle=True)
    observations = base["observations"].astype(np.float32)
    actions = base["actions"].astype(np.int64)
    valid_mask = base["valid_mask"].astype(bool)
    sequence_type = base["sequence_type"].astype(np.int16)
    boxes = base["boxes_destroyed_after_bomb"].astype(np.float32)
    survived = base["teacher_survived_after_bomb"].astype(np.float32)
    context_id = base["bomb_context_id"].astype(np.int64)

    pos_idx = np.where(sequence_type == TYPE_SAFE_BOMB)[0]
    unsafe_idx = np.where(sequence_type == TYPE_UNSAFE)[0]

    neg_obs_parts = [observations[unsafe_idx]]
    neg_actions_parts = [actions[unsafe_idx]]
    neg_mask_parts = [valid_mask[unsafe_idx]]
    neg_boxes_parts = [boxes[unsafe_idx]]
    neg_source_parts = [np.full(len(unsafe_idx), 1, dtype=np.int16)]

    cf_path = Path(args.counterfactual_negatives)
    cf_count = 0
    if cf_path.exists():
        cf = np.load(cf_path, allow_pickle=True)
        cf_obs = cf["observations"].astype(np.float32)
        cf_boxes = cf.get("would_destroy_boxes", np.zeros(len(cf_obs), dtype=np.float32)).astype(np.float32)
        order = np.arange(len(cf_obs))
        rng.shuffle(order)
        order = order[: args.max_counterfactual_negatives]
        seq, cf_actions, cf_mask = _pad_single_obs(cf_obs[order], observations.shape[1], observations.shape[2])
        neg_obs_parts.append(seq)
        neg_actions_parts.append(cf_actions)
        neg_mask_parts.append(cf_mask)
        neg_boxes_parts.append(cf_boxes[order])
        neg_source_parts.append(np.full(len(order), 2, dtype=np.int16))
        cf_count = len(order)

    pos_obs = observations[pos_idx]
    pos_actions = actions[pos_idx]
    pos_mask = valid_mask[pos_idx]
    pos_boxes = boxes[pos_idx]
    pos_survived = survived[pos_idx]
    pos_context_id = context_id[pos_idx]

    neg_obs = np.concatenate(neg_obs_parts, axis=0)
    neg_actions = np.concatenate(neg_actions_parts, axis=0)
    neg_mask = np.concatenate(neg_mask_parts, axis=0)
    neg_boxes = np.concatenate(neg_boxes_parts, axis=0).astype(np.float32)
    neg_source = np.concatenate(neg_source_parts, axis=0)

    pair_pos_idx, pair_neg_idx = _nearest_negative_pairs(pos_boxes, neg_boxes, args.pairs_per_positive)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        pos_observations=pos_obs,
        pos_actions=pos_actions,
        pos_valid_mask=pos_mask,
        pos_boxes_destroyed_after_bomb=pos_boxes,
        pos_teacher_survived_after_bomb=pos_survived,
        pos_bomb_context_id=pos_context_id,
        neg_observations=neg_obs,
        neg_actions=neg_actions,
        neg_valid_mask=neg_mask,
        neg_boxes_destroyed_after_bomb=neg_boxes,
        neg_source=neg_source,
        pair_pos_idx=pair_pos_idx,
        pair_neg_idx=pair_neg_idx,
    )

    report = {
        "balanced_dataset": args.balanced_dataset,
        "counterfactual_negatives": str(cf_path) if cf_path.exists() else "",
        "positive_count": int(len(pos_idx)),
        "unsafe_negative_count": int(len(unsafe_idx)),
        "counterfactual_negative_count": int(cf_count),
        "total_negative_count": int(len(neg_obs)),
        "pair_count": int(len(pair_pos_idx)),
        "positive_boxes_mean": float(pos_boxes.mean()) if len(pos_boxes) else 0.0,
        "negative_boxes_mean": float(neg_boxes.mean()) if len(neg_boxes) else 0.0,
        "negative_source_counts": {
            "balanced_unsafe": int(np.sum(neg_source == 1)),
            "counterfactual": int(np.sum(neg_source == 2)),
        },
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
