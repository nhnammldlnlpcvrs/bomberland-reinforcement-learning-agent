"""Print summary statistics for Bomberland imitation datasets."""

import argparse
import json
from collections import Counter

import numpy as np


ACTION_NAMES = {
    0: "STOP",
    1: "LEFT",
    2: "RIGHT",
    3: "UP",
    4: "DOWN",
    5: "PLACE_BOMB",
}


def _load_metadata(data):
    raw = data.get("metadata_json")
    if raw is None:
        return {}
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _pct(count, total):
    if total <= 0:
        return 0.0
    return 100.0 * count / total


def print_stats(dataset):
    data = np.load(dataset, allow_pickle=False)
    observations = data["observations"]
    actions = data["actions"]
    outcomes = data["outcomes"]
    ranks = data["ranks"]
    steps = data["steps"]
    metadata = _load_metadata(data)
    sample_meta = metadata.get("samples", [])
    build_stats = metadata.get("stats", {})

    total = int(actions.shape[0])
    episode_ids = {item.get("episode_id") for item in sample_meta if isinstance(item, dict)}
    action_counts = Counter(int(action) for action in actions.tolist())
    outcome_counts = Counter(str(outcome) for outcome in outcomes.tolist())
    episode_lengths = []
    by_episode = {}
    for item in sample_meta:
        if not isinstance(item, dict):
            continue
        episode_id = item.get("episode_id")
        if episode_id is None:
            continue
        by_episode.setdefault(episode_id, []).append(int(item.get("step", 0)))
    for episode_steps in by_episode.values():
        if episode_steps:
            episode_lengths.append(max(episode_steps))

    print("=== Dataset Stats ===")
    print(f"dataset: {dataset}")
    print(f"total samples: {total}")
    print(f"total episodes: {len(episode_ids)}")
    print(f"observations shape: {observations.shape}")
    print(f"actions shape: {actions.shape}")
    print(f"rewards_proxy shape: {data['rewards_proxy'].shape}")
    print(f"outcomes shape: {outcomes.shape}")
    print(f"ranks shape: {ranks.shape}")
    print(f"steps shape: {steps.shape}")

    print("\n=== Action Distribution ===")
    for action in range(6):
        count = action_counts[action]
        print(f"{action} {ACTION_NAMES[action]}: {count} ({_pct(count, total):.1f}%)")

    print("\n=== Outcomes ===")
    for outcome in ("win", "draw", "loss", "unknown"):
        count = outcome_counts[outcome]
        print(f"{outcome}: {count} ({_pct(count, total):.1f}%)")

    print("\n=== Timing ===")
    print(f"average episode length: {float(np.mean(episode_lengths)) if episode_lengths else 0.0:.1f}")
    print(f"average step sampled: {float(np.mean(steps)) if total else 0.0:.1f}")

    print("\n=== Skips ===")
    print(f"skipped malformed frames: {build_stats.get('skipped_malformed_frames', 0)}")
    print(f"skipped missing-action frames: {build_stats.get('skipped_missing_action', 0)}")
    print(f"skipped invalid-action frames: {build_stats.get('skipped_invalid_action', 0)}")
    print(f"skipped dead-agent frames: {build_stats.get('skipped_dead_agent_frames', 0)}")
    print(f"skipped no-team files: {build_stats.get('skipped_no_team', 0)}")
    print(f"skipped filtered episodes: {build_stats.get('skipped_filtered_episodes', 0)}")

    print("\n=== Warnings ===")
    warnings = []
    if total:
        most_common_action, most_common_count = action_counts.most_common(1)[0]
        if most_common_count / total > 0.55:
            warnings.append(
                f"action imbalance: {ACTION_NAMES.get(most_common_action, most_common_action)} "
                f"is {_pct(most_common_count, total):.1f}%"
            )
        if action_counts[0] / total > 0.35:
            warnings.append(f"too many STOP samples: {_pct(action_counts[0], total):.1f}%")
        if outcome_counts["draw"] / total > 0.45:
            warnings.append(f"many draw samples: {_pct(outcome_counts['draw'], total):.1f}%")
    if not warnings:
        print("none")
    else:
        for warning in warnings:
            print(f"WARNING: {warning}")


def main():
    parser = argparse.ArgumentParser(description="Summarize a Bomberland imitation dataset.")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    print_stats(args.dataset)


if __name__ == "__main__":
    main()

