from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TYPE_SAFE_BOMB = 1
TYPE_UNSAFE = 3

SOURCE_NAMES = np.asarray(
    [
        "safe_value_positive",
        "safe_zero_value_negative",
        "unsafe_negative",
        "gameplay_zero_value_negative",
    ],
    dtype=object,
)


def _pad_single_obs(obs: np.ndarray, seq_len: int, channels: int) -> tuple[np.ndarray, np.ndarray]:
    seq = np.zeros((len(obs), seq_len, channels, 13, 13), dtype=np.float32)
    mask = np.zeros((len(obs), seq_len), dtype=bool)
    if len(obs):
        seq[:, 0] = obs.astype(np.float32)
        mask[:, 0] = True
    return seq, mask


def _append(parts: dict, obs, mask, labels, source_type, has_escape, boxes, reachable, enemy, zero_reason):
    count = len(labels)
    if count == 0:
        return
    parts["observations"].append(obs.astype(np.float32))
    parts["valid_mask"].append(mask.astype(bool))
    parts["label_value_now"].append(np.asarray(labels, dtype=np.float32))
    parts["source_type"].append(np.full(count, source_type, dtype=np.int16))
    parts["has_escape_after_bomb"].append(np.asarray(has_escape, dtype=np.float32))
    parts["would_destroy_boxes"].append(np.asarray(boxes, dtype=np.float32))
    parts["expected_boxes"].append(np.asarray(boxes, dtype=np.float32))
    parts["reachable_delta"].append(np.asarray(reachable, dtype=np.float32))
    parts["enemy_pressure_proxy"].append(np.asarray(enemy, dtype=np.float32))
    parts["zero_value_reason"].append(np.asarray(zero_reason, dtype=np.int16))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline bomb_value_now dataset for modular recurrent BC.")
    parser.add_argument("--balanced_dataset", default="ml/datasets/recurrent_bc_balanced_sequences.npz")
    parser.add_argument("--counterfactual_negatives", default="ml/datasets/bomb_counterfactual_negatives.npz")
    parser.add_argument("--ranking_dataset", default="ml/datasets/recurrent_bomb_ranking_dataset.npz")
    parser.add_argument("--activation_log", default="logs/modular_activation_analysis.json")
    parser.add_argument("--output", default="ml/datasets/bomb_value_now_dataset.npz")
    parser.add_argument("--max_safe_zero", type=int, default=1200)
    parser.add_argument("--max_unsafe", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=9990)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    parts = {
        "observations": [],
        "valid_mask": [],
        "label_value_now": [],
        "source_type": [],
        "has_escape_after_bomb": [],
        "would_destroy_boxes": [],
        "expected_boxes": [],
        "reachable_delta": [],
        "enemy_pressure_proxy": [],
        "zero_value_reason": [],
    }

    balanced = np.load(args.balanced_dataset, allow_pickle=True)
    seq_len = int(balanced["observations"].shape[1])
    channels = int(balanced["observations"].shape[2])
    seq_type = balanced["sequence_type"].astype(np.int16)
    boxes = balanced["boxes_destroyed_after_bomb"].astype(np.float32)
    survived = balanced["teacher_survived_after_bomb"].astype(np.float32)
    positive_idx = np.where((seq_type == TYPE_SAFE_BOMB) & (survived > 0.5) & (boxes > 0))[0]
    unsafe_idx = np.where(seq_type == TYPE_UNSAFE)[0]
    _append(
        parts,
        balanced["observations"][positive_idx],
        balanced["valid_mask"][positive_idx],
        np.ones(len(positive_idx), dtype=np.float32),
        0,
        np.ones(len(positive_idx), dtype=np.float32),
        boxes[positive_idx],
        np.maximum(boxes[positive_idx], 0),
        np.zeros(len(positive_idx), dtype=np.float32),
        np.zeros(len(positive_idx), dtype=np.int16),
    )
    _append(
        parts,
        balanced["observations"][unsafe_idx],
        balanced["valid_mask"][unsafe_idx],
        np.zeros(len(unsafe_idx), dtype=np.float32),
        2,
        np.zeros(len(unsafe_idx), dtype=np.float32),
        boxes[unsafe_idx],
        np.zeros(len(unsafe_idx), dtype=np.float32),
        np.zeros(len(unsafe_idx), dtype=np.float32),
        np.full(len(unsafe_idx), 3, dtype=np.int16),
    )

    cf_path = Path(args.counterfactual_negatives)
    safe_zero_count = unsafe_count = 0
    if cf_path.exists():
        cf = np.load(cf_path, allow_pickle=True)
        cf_obs = cf["observations"].astype(np.float32)
        cf_boxes = cf["would_destroy_boxes"].astype(np.float32)
        cf_escape = cf["has_escape_after_bomb"].astype(np.float32)
        cf_reachable = cf["reachable_after_bomb"].astype(np.float32)
        cf_enemy = cf["enemy_pressure"].astype(np.float32) if "enemy_pressure" in cf.files else np.zeros(len(cf_obs), dtype=np.float32)
        danger_current = cf["danger_current"].astype(np.float32) if "danger_current" in cf.files else np.zeros(len(cf_obs), dtype=np.float32)
        danger_future = cf["danger_future"].astype(np.float32) if "danger_future" in cf.files else np.zeros(len(cf_obs), dtype=np.float32)

        safe_escape_cells = cf["safe_escape_cells"].astype(np.float32) if "safe_escape_cells" in cf.files else cf_reachable
        escape_exits = cf["escape_exits"].astype(np.float32) if "escape_exits" in cf.files else np.zeros(len(cf_obs), dtype=np.float32)
        # These are the crucial negatives missing from previous training:
        # placing a bomb is escapable, but it has no immediate value. The
        # counterfactual generator stores danger_current on a 1..7 scale, so do
        # not require it to be zero here.
        safe_zero = np.where(
            (cf_escape > 0.5)
            & (cf_boxes <= 0)
            & (cf_enemy <= 0)
            & (safe_escape_cells > 0)
            & (escape_exits > 0)
        )[0]
        unsafe = np.where((cf_escape <= 0.5) | (danger_current > 0) | (danger_future > 2))[0]
        rng.shuffle(safe_zero)
        rng.shuffle(unsafe)
        safe_zero = safe_zero[: args.max_safe_zero]
        unsafe = unsafe[: args.max_unsafe]
        safe_zero_seq, safe_zero_mask = _pad_single_obs(cf_obs[safe_zero], seq_len, channels)
        unsafe_seq, unsafe_mask = _pad_single_obs(cf_obs[unsafe], seq_len, channels)
        _append(
            parts,
            safe_zero_seq,
            safe_zero_mask,
            np.zeros(len(safe_zero), dtype=np.float32),
            1,
            cf_escape[safe_zero],
            cf_boxes[safe_zero],
            cf_reachable[safe_zero],
            cf_enemy[safe_zero],
            np.ones(len(safe_zero), dtype=np.int16),
        )
        _append(
            parts,
            unsafe_seq,
            unsafe_mask,
            np.zeros(len(unsafe), dtype=np.float32),
            2,
            cf_escape[unsafe],
            cf_boxes[unsafe],
            cf_reachable[unsafe],
            cf_enemy[unsafe],
            np.full(len(unsafe), 2, dtype=np.int16),
        )
        safe_zero_count = len(safe_zero)
        unsafe_count = len(unsafe)

    # Gameplay activation logs currently contain aggregate counters, not states.
    gameplay_zero_value_count = 0
    if Path(args.activation_log).exists():
        try:
            log = json.loads(Path(args.activation_log).read_text(encoding="utf-8"))
            for seed_data in log.get("activation_by_seed", {}).values():
                for threshold_data in seed_data.values():
                    for opponent_data in threshold_data.values():
                        gameplay_zero_value_count += int(opponent_data.get("bombs_with_zero_value", 0))
        except Exception:
            gameplay_zero_value_count = 0

    arrays = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, source_type_names=SOURCE_NAMES, **arrays)

    labels = arrays["label_value_now"]
    source = arrays["source_type"]
    report = {
        "output": str(output),
        "positive_count": int(labels.sum()),
        "safe_zero_value_negative_count": int(np.sum(source == 1)),
        "unsafe_negative_count": int(np.sum(source == 2)),
        "gameplay_zero_value_negative_count_available_only": int(gameplay_zero_value_count),
        "gameplay_zero_value_negative_count_with_obs": 0,
        "total_count": int(len(labels)),
        "positive_fraction": float(labels.mean()) if len(labels) else 0.0,
        "source_distribution": {str(SOURCE_NAMES[i]): int(np.sum(source == i)) for i in range(len(SOURCE_NAMES))},
        "expected_boxes_mean_positive": float(arrays["expected_boxes"][labels > 0.5].mean()) if np.any(labels > 0.5) else 0.0,
        "expected_boxes_mean_negative": float(arrays["expected_boxes"][labels <= 0.5].mean()) if np.any(labels <= 0.5) else 0.0,
        "counterfactual_safe_zero_used": int(safe_zero_count),
        "counterfactual_unsafe_used": int(unsafe_count),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
