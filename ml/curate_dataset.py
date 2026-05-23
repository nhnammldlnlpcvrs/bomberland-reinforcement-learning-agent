"""Curate Bomberland imitation datasets for less passive policy training."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ml.dataset_builder import str2bool
from ml.train_imitation import ACTION_NAMES


TARGET_BOMB_RATIO = 0.07
MAX_STOP_RATIO = 0.30
MIN_DATASET_SIZE_WARNING = 1000


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


def _pct(count, total):
    return 0.0 if total <= 0 else 100.0 * float(count) / float(total)


def _outcome_counts(outcomes):
    return Counter(str(outcome) for outcome in outcomes.tolist())


def _action_counts(actions):
    return np.bincount(actions.astype(np.int64), minlength=len(ACTION_NAMES))


def _episode_ids(metadata, total):
    samples = metadata.get("samples", [])
    ids = []
    for idx in range(total):
        if idx < len(samples) and isinstance(samples[idx], dict):
            ids.append(str(samples[idx].get("episode_id", "unknown")))
        else:
            ids.append("unknown")
    return np.array(ids, dtype=object)


def _enemy_distance_from_obs(observation):
    self_plane = observation[6]
    enemy_plane = observation[7]
    self_cells = np.argwhere(self_plane > 0.5)
    enemy_cells = np.argwhere(enemy_plane > 0.5)
    if len(self_cells) == 0 or len(enemy_cells) == 0:
        return 99
    sr, sc = self_cells[0]
    distances = np.abs(enemy_cells[:, 0] - sr) + np.abs(enemy_cells[:, 1] - sc)
    return int(distances.min())


def _endgame_mask(observations, steps):
    mask = np.zeros(len(steps), dtype=bool)
    for idx, (obs, step) in enumerate(zip(observations, steps)):
        enemy_close = _enemy_distance_from_obs(obs) <= 4
        has_bomb_pressure = bool(obs[2].sum() > 0 or obs[3].max() > 0.0)
        mask[idx] = int(step) >= 350 or enemy_close or has_bomb_pressure
    return mask


def _limit_per_episode(indices, episode_ids, max_per_episode, rng):
    if max_per_episode is None:
        return indices
    grouped = defaultdict(list)
    for idx in indices:
        grouped[episode_ids[idx]].append(int(idx))
    kept = []
    for values in grouped.values():
        if len(values) > max_per_episode:
            kept.extend(rng.choice(values, size=max_per_episode, replace=False).tolist())
        else:
            kept.extend(values)
    return np.array(sorted(kept), dtype=np.int64)


def _undersample_stop(indices, actions, rng):
    stop_indices = indices[actions[indices] == 0]
    other_indices = indices[actions[indices] != 0]
    if len(indices) == 0 or len(stop_indices) == 0:
        return indices
    max_stop = int((MAX_STOP_RATIO * len(other_indices)) / max(1e-9, 1.0 - MAX_STOP_RATIO))
    max_stop = max(1, max_stop)
    if len(stop_indices) <= max_stop:
        return indices
    kept_stop = rng.choice(stop_indices, size=max_stop, replace=False)
    return np.array(sorted(np.concatenate([other_indices, kept_stop])), dtype=np.int64)


def _oversample_bombs(indices, actions, rng):
    bomb_indices = indices[actions[indices] == 5]
    if len(indices) == 0 or len(bomb_indices) == 0:
        return indices, 0
    current_bombs = len(bomb_indices)
    target_bombs = int(np.ceil(TARGET_BOMB_RATIO * len(indices)))
    extra_needed = max(0, target_bombs - current_bombs)
    if extra_needed <= 0:
        return indices, 0
    extra = rng.choice(bomb_indices, size=extra_needed, replace=True)
    return np.concatenate([indices, extra]).astype(np.int64), int(extra_needed)


def _oversample_endgame(indices, endgame_mask, rng):
    endgame_indices = indices[endgame_mask[indices]]
    if len(indices) == 0 or len(endgame_indices) == 0:
        return indices, 0
    target = int(0.35 * len(indices))
    extra_needed = max(0, target - len(endgame_indices))
    extra_needed = min(extra_needed, len(indices) // 4)
    if extra_needed <= 0:
        return indices, 0
    extra = rng.choice(endgame_indices, size=extra_needed, replace=True)
    return np.concatenate([indices, extra]).astype(np.int64), int(extra_needed)


def _metrics(name, actions, outcomes, steps, endgame_mask):
    total = int(len(actions))
    counts = _action_counts(actions)
    outcomes_count = _outcome_counts(outcomes)
    print(f"=== {name} ===")
    print(f"total samples: {total}")
    for idx, action_name in enumerate(ACTION_NAMES):
        print(f"{idx} {action_name}: {int(counts[idx])} ({_pct(counts[idx], total):.1f}%)")
    print(
        "outcomes: "
        f"win={_pct(outcomes_count['win'], total):.1f}% "
        f"draw={_pct(outcomes_count['draw'], total):.1f}% "
        f"loss={_pct(outcomes_count['loss'], total):.1f}%"
    )
    print(f"STOP ratio: {_pct(counts[0], total):.1f}%")
    print(f"PLACE_BOMB ratio: {_pct(counts[5], total):.1f}%")
    print(f"average step: {float(np.mean(steps)) if total else 0.0:.1f}")
    print(f"endgame samples: {int(endgame_mask.sum())} ({_pct(int(endgame_mask.sum()), total):.1f}%)")
    return {
        "total": total,
        "action_counts": counts.tolist(),
        "stop_ratio": _pct(counts[0], total),
        "bomb_ratio": _pct(counts[5], total),
        "average_step": float(np.mean(steps)) if total else 0.0,
        "endgame_count": int(endgame_mask.sum()),
    }


def curate_dataset(args):
    rng = np.random.default_rng(args.seed)
    data = np.load(args.input, allow_pickle=False)
    observations = data["observations"]
    actions = data["actions"].astype(np.int64)
    rewards_proxy = data["rewards_proxy"]
    outcomes = data["outcomes"].astype(str)
    ranks = data["ranks"]
    steps = data["steps"].astype(np.int64)
    metadata = _load_metadata(data)
    episode_ids = _episode_ids(metadata, len(actions))
    endgame = _endgame_mask(observations, steps)

    before = _metrics("Before curation", actions, outcomes, steps, endgame)

    mask = np.ones(len(actions), dtype=bool)
    if args.wins_only:
        mask &= outcomes == "win"
    if args.exclude_draws:
        mask &= outcomes != "draw"
    indices = np.flatnonzero(mask).astype(np.int64)
    indices = _limit_per_episode(indices, episode_ids, args.max_per_episode, rng)

    if args.undersample_stop:
        indices = _undersample_stop(indices, actions, rng)

    duplicated = 0
    if args.oversample_endgame:
        indices, extra = _oversample_endgame(indices, endgame, rng)
        duplicated += extra

    if args.oversample_bomb_actions:
        indices, extra = _oversample_bombs(indices, actions, rng)
        duplicated += extra

    curated_observations = observations[indices].astype(np.float32)
    curated_actions = actions[indices].astype(np.int64)
    curated_rewards = rewards_proxy[indices].astype(np.float32)
    curated_outcomes = outcomes[indices]
    curated_ranks = ranks[indices].astype(np.int64)
    curated_steps = steps[indices].astype(np.int64)
    curated_endgame = endgame[indices]

    source_samples = metadata.get("samples", [])
    curated_samples = []
    duplicate_seen = Counter()
    for source_idx in indices.tolist():
        if source_idx < len(source_samples) and isinstance(source_samples[source_idx], dict):
            item = dict(source_samples[source_idx])
        else:
            item = {"episode_id": "unknown", "step": int(steps[source_idx])}
        duplicate_seen[source_idx] += 1
        if duplicate_seen[source_idx] > 1:
            item["curation_duplicate"] = True
        item["source_index"] = int(source_idx)
        curated_samples.append(item)

    after = _metrics("After curation", curated_actions, curated_outcomes, curated_steps, curated_endgame)

    duplicate_ratio = _pct(duplicated, len(curated_actions))
    print("=== Curation Warnings ===")
    warnings = []
    if after["total"] < MIN_DATASET_SIZE_WARNING:
        warnings.append("dataset too small after curation")
    if after["stop_ratio"] > 35.0:
        warnings.append(f"STOP remains high: {after['stop_ratio']:.1f}%")
    if after["bomb_ratio"] < 4.0:
        warnings.append(f"PLACE_BOMB remains low: {after['bomb_ratio']:.1f}%")
    if duplicate_ratio > 30.0:
        warnings.append(f"many duplicated samples: {duplicate_ratio:.1f}%")
    if not warnings:
        print("none")
    else:
        for warning in warnings:
            print(f"WARNING: {warning}")

    output_metadata = dict(metadata)
    output_metadata["curation"] = {
        "source": str(args.input),
        "options": vars(args),
        "before": before,
        "after": after,
        "duplicated_samples": int(duplicated),
        "duplicated_ratio": duplicate_ratio,
    }
    output_metadata["samples"] = curated_samples

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=curated_observations,
        actions=curated_actions,
        rewards_proxy=curated_rewards,
        outcomes=curated_outcomes,
        ranks=curated_ranks,
        steps=curated_steps,
        metadata_json=json.dumps(output_metadata, sort_keys=True),
    )
    print(f"Saved curated dataset: {output_path}")
    return output_path, before, after


def main():
    parser = argparse.ArgumentParser(description="Curate Bomberland imitation datasets.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wins_only", type=str2bool, default=False)
    parser.add_argument("--exclude_draws", type=str2bool, default=False)
    parser.add_argument("--oversample_bomb_actions", type=str2bool, default=True)
    parser.add_argument("--undersample_stop", type=str2bool, default=True)
    parser.add_argument("--oversample_endgame", type=str2bool, default=True)
    parser.add_argument("--max_per_episode", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    curate_dataset(args)


if __name__ == "__main__":
    main()
