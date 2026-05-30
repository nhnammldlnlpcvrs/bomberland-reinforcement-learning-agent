from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.constants import PLACE_BOMB, STOP


TYPE_NAMES = np.asarray(
    ["normal_movement", "safe_bomb", "post_bomb_escape", "unsafe_bomb_negative"],
    dtype=object,
)
TYPE_TO_ID = {name: idx for idx, name in enumerate(TYPE_NAMES.tolist())}


def _pad_sequence(obs_list, action_list, max_len):
    obs_shape = obs_list[0].shape
    obs = np.zeros((max_len, *obs_shape), dtype=np.float32)
    actions = np.zeros((max_len,), dtype=np.int64)
    mask = np.zeros((max_len,), dtype=bool)
    length = min(max_len, len(action_list))
    obs[:length] = np.asarray(obs_list[:length], dtype=np.float32)
    actions[:length] = np.asarray(action_list[:length], dtype=np.int64)
    mask[:length] = True
    return obs, actions, mask, length


def _append_sequence(
    sequences,
    obs_list,
    action_list,
    sequence_type,
    max_len,
    boxes_destroyed=0.0,
    survived=0.0,
    bomb_context_id=-1,
    step_delta=-1,
):
    if not obs_list or not action_list:
        return
    obs, actions, mask, length = _pad_sequence(obs_list, action_list, max_len)
    sequences.append(
        {
            "obs": obs,
            "actions": actions,
            "mask": mask,
            "length": length,
            "sequence_type": TYPE_TO_ID[sequence_type],
            "boxes_destroyed_after_bomb": float(boxes_destroyed),
            "teacher_survived_after_bomb": float(survived),
            "bomb_context_id": int(bomb_context_id),
            "step_delta": int(step_delta),
        }
    )


def build_normal_sequences(path, max_sequences, max_len, stride, rng):
    data = np.load(path, allow_pickle=True)
    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    valid = data["valid_mask"].astype(bool)
    candidates = []
    for ep_idx in range(len(actions)):
        length = int(valid[ep_idx].sum())
        if length <= 0:
            continue
        for start in range(0, length, stride):
            end = min(length, start + max_len)
            if end <= start:
                continue
            window_actions = actions[ep_idx, start:end]
            if np.any(window_actions == PLACE_BOMB):
                continue
            candidates.append((ep_idx, start, end))
    rng.shuffle(candidates)
    sequences = []
    for ep_idx, start, end in candidates[:max_sequences]:
        _append_sequence(
            sequences,
            list(observations[ep_idx, start:end]),
            list(actions[ep_idx, start:end]),
            "normal_movement",
            max_len,
            bomb_context_id=ep_idx,
        )
    return sequences


def _load_sequence_dataset(path):
    if not path or not Path(path).exists():
        return None
    return np.load(path, allow_pickle=True)


def build_bomb_and_escape_sequences(paths, max_bomb, max_escape, max_len, rng):
    bomb_sequences = []
    escape_sequences = []
    seen_bombs = set()
    seen_escape = set()
    for path in paths:
        data = _load_sequence_dataset(path)
        if data is None:
            continue
        bomb_obs = data["bomb_obs"].astype(np.float32)
        bomb_actions = data["bomb_action"].astype(np.int64)
        bomb_ids = data["bomb_sequence_id"].astype(np.int64)
        escape_obs = data["escape_obs"].astype(np.float32)
        escape_actions = data["escape_action"].astype(np.int64)
        escape_ids = data["escape_sequence_id"].astype(np.int64)
        step_delta = data["step_delta"].astype(np.int64)
        boxes = data["boxes_destroyed_after_bomb"].astype(np.float32)
        escape_by_id = defaultdict(list)
        for obs, action, seq_id, delta in zip(escape_obs, escape_actions, escape_ids, step_delta):
            if action == PLACE_BOMB:
                continue
            # Escape sequences should mostly teach moving out; keep STOP only if it is explicitly present.
            escape_by_id[int(seq_id)].append((int(delta), obs, int(action)))
        order = list(range(len(bomb_obs)))
        rng.shuffle(order)
        for idx in order:
            seq_id = int(bomb_ids[idx])
            key = (Path(path).name, seq_id)
            if key in seen_bombs:
                continue
            if int(bomb_actions[idx]) != PLACE_BOMB:
                continue
            box_value = float(boxes[idx]) if idx < len(boxes) else 0.0
            if box_value <= 0:
                continue
            seen_bombs.add(key)
            _append_sequence(
                bomb_sequences,
                [bomb_obs[idx]],
                [PLACE_BOMB],
                "safe_bomb",
                max_len,
                boxes_destroyed=box_value,
                survived=1.0,
                bomb_context_id=seq_id,
            )
            escape_steps = sorted(escape_by_id.get(seq_id, []), key=lambda item: item[0])
            escape_steps = [item for item in escape_steps if 1 <= item[0] <= 5 and item[2] != PLACE_BOMB]
            if escape_steps:
                obs_list = [item[1] for item in escape_steps[:max_len]]
                action_list = [item[2] for item in escape_steps[:max_len]]
                if action_list:
                    seen_escape.add(key)
                    _append_sequence(
                        escape_sequences,
                        obs_list,
                        action_list,
                        "post_bomb_escape",
                        max_len,
                        boxes_destroyed=box_value,
                        survived=1.0,
                        bomb_context_id=seq_id,
                        step_delta=int(max(item[0] for item in escape_steps[:max_len])),
                    )
    rng.shuffle(bomb_sequences)
    rng.shuffle(escape_sequences)
    return bomb_sequences[:max_bomb], escape_sequences[:max_escape]


def build_unsafe_negative_sequences(path, max_sequences, max_len, rng):
    if not path or not Path(path).exists() or max_sequences <= 0:
        return []
    data = np.load(path, allow_pickle=True)
    observations = data["observations"].astype(np.float32)
    labels = data["labels"].astype(np.int64) if "labels" in data.files else np.zeros(len(observations), dtype=np.int64)
    reason = data["negative_reason"] if "negative_reason" in data.files else np.asarray(["unknown"] * len(observations))
    indices = [i for i, label in enumerate(labels) if int(label) == 0]
    rng.shuffle(indices)
    sequences = []
    for idx in indices[:max_sequences]:
        # Counterfactual negatives only say "do not bomb"; STOP is the least committal non-bomb label.
        _append_sequence(
            sequences,
            [observations[idx]],
            [STOP],
            "unsafe_bomb_negative",
            max_len,
            survived=0.0,
            bomb_context_id=idx,
            step_delta=-1,
        )
        sequences[-1]["negative_reason"] = str(reason[idx])
    return sequences


def pack_sequences(sequences, output):
    observations = np.asarray([s["obs"] for s in sequences], dtype=np.float32)
    actions = np.asarray([s["actions"] for s in sequences], dtype=np.int64)
    valid_mask = np.asarray([s["mask"] for s in sequences], dtype=bool)
    lengths = np.asarray([s["length"] for s in sequences], dtype=np.int32)
    sequence_type = np.asarray([s["sequence_type"] for s in sequences], dtype=np.int16)
    metadata = {
        "boxes_destroyed_after_bomb": np.asarray([s["boxes_destroyed_after_bomb"] for s in sequences], dtype=np.float32),
        "teacher_survived_after_bomb": np.asarray([s["teacher_survived_after_bomb"] for s in sequences], dtype=np.float32),
        "bomb_context_id": np.asarray([s["bomb_context_id"] for s in sequences], dtype=np.int64),
        "step_delta": np.asarray([s["step_delta"] for s in sequences], dtype=np.int16),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=observations,
        actions=actions,
        valid_mask=valid_mask,
        lengths=lengths,
        sequence_type=sequence_type,
        sequence_type_names=TYPE_NAMES,
        **metadata,
    )
    return observations, actions, valid_mask, sequence_type, metadata


def summarize(actions, mask, sequence_type, metadata):
    action_values = actions[mask]
    type_counts = Counter(int(v) for v in sequence_type)
    summary = {
        "num_sequences": int(len(sequence_type)),
        "total_steps": int(mask.sum()),
        "action_distribution": {str(i): int(v) for i, v in enumerate(np.bincount(action_values, minlength=6))},
        "bomb_label_count": int(np.sum(action_values == PLACE_BOMB)),
        "sequence_type_distribution": {str(TYPE_NAMES[k]): int(type_counts.get(k, 0)) for k in range(len(TYPE_NAMES))},
        "boxes_destroyed_after_bomb_mean": float(np.mean(metadata["boxes_destroyed_after_bomb"][metadata["boxes_destroyed_after_bomb"] > 0]))
        if np.any(metadata["boxes_destroyed_after_bomb"] > 0)
        else 0.0,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build sequence-balanced recurrent BC dataset.")
    parser.add_argument("--normal_dataset", default="ml/datasets/recurrent_bc_online_robust.npz")
    parser.add_argument("--selected_sequences", default="ml/datasets/rl_bc_selected_bomb_sequences_v3.npz")
    parser.add_argument("--useful_sequences", default="ml/datasets/rl_bc_useful_bomb_sequences.npz")
    parser.add_argument("--unsafe_negatives", default="ml/datasets/bomb_counterfactual_negatives.npz")
    parser.add_argument("--output", default="ml/datasets/recurrent_bc_balanced_sequences.npz")
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--normal_sequences", type=int, default=1200)
    parser.add_argument("--safe_bomb_sequences", type=int, default=180)
    parser.add_argument("--escape_sequences", type=int, default=500)
    parser.add_argument("--unsafe_sequences", type=int, default=120)
    parser.add_argument("--normal_stride", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9960)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    normal = build_normal_sequences(args.normal_dataset, args.normal_sequences, args.max_len, args.normal_stride, rng)
    bomb, escape = build_bomb_and_escape_sequences(
        [args.selected_sequences, args.useful_sequences],
        args.safe_bomb_sequences,
        args.escape_sequences,
        args.max_len,
        rng,
    )
    unsafe = build_unsafe_negative_sequences(args.unsafe_negatives, args.unsafe_sequences, args.max_len, rng)
    sequences = normal + bomb + escape + unsafe
    rng.shuffle(sequences)
    observations, actions, valid_mask, sequence_type, metadata = pack_sequences(sequences, args.output)
    summary = summarize(actions, valid_mask, sequence_type, metadata)
    summary.update(
        {
            "output": args.output,
            "max_len": int(args.max_len),
            "sources": {
                "normal_dataset": args.normal_dataset,
                "selected_sequences": args.selected_sequences,
                "useful_sequences": args.useful_sequences,
                "unsafe_negatives": args.unsafe_negatives,
            },
        }
    )
    Path(args.output).with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
