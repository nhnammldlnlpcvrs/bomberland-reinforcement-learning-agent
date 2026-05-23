"""Build safe-action ranking datasets from Bomberland imitation datasets.

This is research-only. The safe-action mask is a lightweight approximation from
encoded replay features; production safety remains authoritative.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ml.dataset_builder import str2bool
from ml.train_imitation import ACTION_NAMES


ACTION_DELTAS = {
    0: (0, 0),    # STOP
    1: (-1, 0),   # LEFT in the competition coordinate convention
    2: (1, 0),    # RIGHT
    3: (0, -1),   # UP
    4: (0, 1),    # DOWN
}


def _load_metadata(data):
    raw = data.get("metadata_json")
    if raw is None:
        return {}
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {}


def _self_position(obs):
    cells = np.argwhere(obs[6] > 0.5)
    if len(cells) == 0:
        return None
    return int(cells[0, 0]), int(cells[0, 1])


def _in_bounds(row, col):
    return 0 <= row < 13 and 0 <= col < 13


def _adjacent_signal(obs, row, col):
    crates = obs[1]
    enemies = obs[7]
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = row + dr, col + dc
        if _in_bounds(nr, nc) and (crates[nr, nc] > 0.5 or enemies[nr, nc] > 0.5):
            return True
    return False


def compute_safe_action_mask(obs, chosen_action, include_unchosen_bombs=False):
    """Approximate valid/safe actions from encoded feature planes."""
    mask = np.zeros(6, dtype=bool)
    pos = _self_position(obs)
    if pos is None:
        return mask

    row, col = pos
    walkable = obs[8]
    safe = obs[9]
    bombs = obs[2]

    for action, (dr, dc) in ACTION_DELTAS.items():
        nr, nc = row + dr, col + dc
        if _in_bounds(nr, nc) and walkable[nr, nc] > 0.5 and safe[nr, nc] > 0.5:
            mask[action] = True

    current_safe = safe[row, col] > 0.5 and bombs[row, col] <= 0.5
    if int(chosen_action) == 5:
        # The heuristic already chose bomb in replay, so keep it as a positive
        # candidate even though this builder cannot recompute can_escape_after_bomb.
        mask[5] = current_safe
    elif include_unchosen_bombs and current_safe and _adjacent_signal(obs, row, col):
        mask[5] = True

    if 0 <= int(chosen_action) < 6 and not mask[int(chosen_action)]:
        return np.zeros(6, dtype=bool)
    return mask


def _sample_indices(total, max_samples, seed):
    indices = np.arange(total)
    if max_samples is not None and total > max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=int(max_samples), replace=False))
    return indices


def build_action_ranking_dataset(args):
    data = np.load(args.input, allow_pickle=False)
    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    rewards_proxy = data["rewards_proxy"] if "rewards_proxy" in data else np.zeros(len(actions), dtype=np.float32)
    outcomes = data["outcomes"].astype(str) if "outcomes" in data else np.array(["unknown"] * len(actions))
    ranks = data["ranks"] if "ranks" in data else np.full(len(actions), -1, dtype=np.int64)
    steps = data["steps"] if "steps" in data else np.zeros(len(actions), dtype=np.int64)
    metadata = _load_metadata(data)
    source_samples = metadata.get("samples", [])

    selected = _sample_indices(len(actions), args.max_samples, args.seed)
    kept_obs = []
    kept_actions = []
    kept_masks = []
    kept_rewards = []
    kept_outcomes = []
    kept_ranks = []
    kept_steps = []
    kept_meta = []
    skipped = 0
    action_mask_counts = np.zeros(6, dtype=np.int64)

    for idx in selected:
        action = int(actions[idx])
        if action < 0 or action >= 6:
            skipped += 1
            continue
        mask = compute_safe_action_mask(
            observations[idx],
            action,
            include_unchosen_bombs=args.include_unchosen_bombs,
        )
        if not mask[action] or not mask.any():
            skipped += 1
            continue

        kept_obs.append(observations[idx])
        kept_actions.append(action)
        kept_masks.append(mask)
        kept_rewards.append(float(rewards_proxy[idx]))
        kept_outcomes.append(str(outcomes[idx]))
        kept_ranks.append(int(ranks[idx]))
        kept_steps.append(int(steps[idx]))
        action_mask_counts += mask.astype(np.int64)
        if idx < len(source_samples) and isinstance(source_samples[idx], dict):
            item = dict(source_samples[idx])
        else:
            item = {"source_index": int(idx)}
        item["source_index"] = int(idx)
        item["safe_actions"] = [ACTION_NAMES[i] for i, value in enumerate(mask) if value]
        kept_meta.append(item)

    if not kept_obs:
        raise RuntimeError("No valid ranking samples were extracted.")

    observations_out = np.stack(kept_obs, axis=0).astype(np.float32)
    actions_out = np.array(kept_actions, dtype=np.int64)
    masks_out = np.stack(kept_masks, axis=0).astype(bool)
    metadata_out = {
        "source": str(args.input),
        "schema": "action_ranking_v1",
        "action_names": list(ACTION_NAMES),
        "options": vars(args),
        "stats": {
            "source_samples": int(len(actions)),
            "selected_samples": int(len(selected)),
            "kept_samples": int(len(actions_out)),
            "skipped_samples": int(skipped),
            "safe_action_counts": action_mask_counts.tolist(),
        },
        "samples": kept_meta,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=observations_out,
        target_actions=actions_out,
        safe_action_masks=masks_out,
        rewards_proxy=np.array(kept_rewards, dtype=np.float32),
        outcomes=np.array(kept_outcomes),
        ranks=np.array(kept_ranks, dtype=np.int64),
        steps=np.array(kept_steps, dtype=np.int64),
        metadata_json=json.dumps(metadata_out, sort_keys=True),
    )

    target_counts = np.bincount(actions_out, minlength=6)
    print(f"Saved action ranking dataset: {output}")
    print(f"kept_samples={len(actions_out)} skipped={skipped}")
    print("target action distribution:")
    for idx, name in enumerate(ACTION_NAMES):
        pct = 100.0 * target_counts[idx] / max(1, len(actions_out))
        print(f"  {idx} {name}: {int(target_counts[idx])} ({pct:.1f}%)")
    print("safe action candidate counts:")
    for idx, name in enumerate(ACTION_NAMES):
        pct = 100.0 * action_mask_counts[idx] / max(1, len(actions_out))
        print(f"  {idx} {name}: {int(action_mask_counts[idx])} ({pct:.1f}%)")
    return output


def main():
    parser = argparse.ArgumentParser(description="Build safe-action ranking dataset from imitation .npz.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--include_unchosen_bombs", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_action_ranking_dataset(args)


if __name__ == "__main__":
    main()
