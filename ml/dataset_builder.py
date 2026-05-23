"""Build imitation-learning datasets from Bomberland replay logs."""

import argparse
import json
from pathlib import Path

import numpy as np

from ml.features import CHANNEL_NAMES, encode_observation


VALID_ACTIONS = {0, 1, 2, 3, 4, 5}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def iter_replay_files(log_dir):
    """Yield replay JSON files from a log directory."""
    root = Path(log_dir)
    if not root.exists():
        return
    yield from sorted(root.rglob("*.json"))


def load_replay(path):
    """Load one replay JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _names(payload):
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    for key in ("team_ids", "agents", "agent_names", "teams", "participants"):
        values = _as_list(payload.get(key))
        if values:
            return [str(value) for value in values]
    values = _as_list(meta.get("agent_names"))
    if values:
        return [str(value) for value in values]
    return []


def _history(payload):
    for key in ("history", "frames", "steps", "trajectory", "replay"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _actions(frame):
    actions = frame.get("actions", frame.get("action"))
    return _as_list(actions)


def _players(frame):
    players = frame.get("players", frame.get("player_state", frame.get("agents")))
    return _as_list(players)


def _player_alive(player):
    if isinstance(player, dict):
        return bool(player.get("alive", player.get("is_alive", True)))
    values = _as_list(player)
    if len(values) >= 3:
        return bool(values[2])
    return True


def _step(frame, fallback=0):
    for key in ("step", "_step", "t", "tick"):
        try:
            if frame.get(key) is not None:
                return int(frame.get(key))
        except (TypeError, ValueError):
            continue
    return int(fallback)


def _target_index(payload, team_name):
    wanted = str(team_name).lower()
    names = _names(payload)
    for idx, name in enumerate(names):
        if str(name).lower() == wanted:
            return idx
    return None


def _outcome_and_rank(payload, history, agent_idx):
    ranks = _as_list(payload.get("ranks", payload.get("rank")))
    if ranks and agent_idx < len(ranks):
        rank = int(ranks[agent_idx])
        winners = sum(1 for value in ranks if int(value) == 0)
        if rank == 0 and winners == 1:
            return "win", rank
        if rank == 0:
            return "draw", rank
        return "loss", rank

    if not history:
        return "unknown", -1
    final_players = _players(history[-1])
    if agent_idx >= len(final_players):
        return "unknown", -1
    alive = _player_alive(final_players[agent_idx])
    alive_count = sum(1 for player in final_players if _player_alive(player))
    if alive and alive_count == 1:
        return "win", 0
    if alive and alive_count > 1:
        return "draw", 0
    return "loss", 1


def _reward_proxy(outcome):
    if outcome == "win":
        return 1.0
    if outcome == "loss":
        return -1.0
    return 0.0


def _survival_steps(payload, agent_idx):
    steps = _as_list(payload.get("survival_steps", payload.get("steps_csv")))
    if steps and agent_idx < len(steps):
        try:
            return int(steps[agent_idx])
        except (TypeError, ValueError):
            return -1
    return -1


def _valid_frame_pair(prev_frame, action_frame, agent_idx):
    actions = _actions(action_frame)
    if agent_idx >= len(actions):
        return False, "missing_action"
    try:
        action = int(actions[agent_idx])
    except (TypeError, ValueError):
        return False, "invalid_action"
    if action not in VALID_ACTIONS:
        return False, "invalid_action"

    players = _players(prev_frame)
    if agent_idx >= len(players):
        return False, "malformed_frame"
    if not _player_alive(players[agent_idx]):
        return False, "dead_agent"
    if not (prev_frame.get("map") is not None or prev_frame.get("board") is not None or prev_frame.get("grid") is not None):
        return False, "malformed_frame"
    return True, action


def _metadata_for_sample(path, payload, agent_idx, prev_frame, action, outcome, rank):
    episode_id = payload.get("seed")
    if episode_id is None:
        episode_id = Path(path).stem
    return {
        "episode_id": str(episode_id),
        "file": Path(path).name,
        "step": _step(prev_frame),
        "action": int(action),
        "alive": True,
        "final_rank": int(rank),
        "final_outcome": outcome,
        "survival_steps": _survival_steps(payload, agent_idx),
    }


def extract_obs_action_pairs(path, payload, team_name, options, stats):
    """Extract encoded obs/action pairs for one replay."""
    agent_idx = _target_index(payload, team_name)
    if agent_idx is None:
        stats["skipped_no_team"] += 1
        return []

    history = _history(payload)
    if len(history) < 2:
        stats["skipped_malformed_frames"] += 1
        return []

    outcome, rank = _outcome_and_rank(payload, history, agent_idx)
    if options["wins_only"] and outcome != "win":
        stats["skipped_filtered_episodes"] += 1
        return []
    if options["exclude_draws"] and outcome == "draw":
        stats["skipped_filtered_episodes"] += 1
        return []

    samples = []
    for idx in range(1, len(history)):
        prev_frame = history[idx - 1]
        action_frame = history[idx]
        ok, action_or_reason = _valid_frame_pair(prev_frame, action_frame, agent_idx)
        if not ok:
            reason = action_or_reason
            if reason == "dead_agent":
                stats["skipped_dead_agent_frames"] += 1
            elif reason == "missing_action":
                stats["skipped_missing_action"] += 1
            elif reason == "invalid_action":
                stats["skipped_invalid_action"] += 1
            else:
                stats["skipped_malformed_frames"] += 1
            continue

        action = int(action_or_reason)
        frame_for_encoding = dict(prev_frame)
        frame_for_encoding["_agent_index"] = agent_idx
        try:
            encoded = encode_observation(frame_for_encoding, team_name=team_name)
            obs_tensor = encoded["tensor"]
        except Exception:
            stats["skipped_malformed_frames"] += 1
            continue

        metadata = _metadata_for_sample(
            path, payload, agent_idx, prev_frame, action, outcome, rank
        )
        samples.append(
            {
                "observation": obs_tensor,
                "action": action,
                "reward_proxy": _reward_proxy(outcome),
                "outcome": outcome,
                "rank": int(rank),
                "step": int(metadata["step"]),
                "metadata": metadata,
            }
        )

        max_samples = options.get("max_samples_per_episode")
        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


def build_dataset(log_dir, team_name, output, wins_only=False,
                  max_samples_per_episode=None, exclude_draws=False):
    """Build and save a compressed imitation dataset."""
    stats = {
        "files_scanned": 0,
        "episodes_used": 0,
        "skipped_no_team": 0,
        "skipped_filtered_episodes": 0,
        "skipped_malformed_frames": 0,
        "skipped_missing_action": 0,
        "skipped_invalid_action": 0,
        "skipped_dead_agent_frames": 0,
    }
    options = {
        "wins_only": bool(wins_only),
        "exclude_draws": bool(exclude_draws),
        "max_samples_per_episode": max_samples_per_episode,
    }

    all_samples = []
    for path in iter_replay_files(log_dir) or []:
        stats["files_scanned"] += 1
        try:
            payload = load_replay(path)
        except Exception:
            stats["skipped_malformed_frames"] += 1
            continue
        samples = extract_obs_action_pairs(path, payload, team_name, options, stats)
        if samples:
            stats["episodes_used"] += 1
            all_samples.extend(samples)

    if not all_samples:
        raise RuntimeError(f"No samples extracted from {log_dir!r} for team {team_name!r}")

    observations = np.stack([sample["observation"] for sample in all_samples], axis=0).astype(np.float32)
    actions = np.array([sample["action"] for sample in all_samples], dtype=np.int64)
    rewards_proxy = np.array([sample["reward_proxy"] for sample in all_samples], dtype=np.float32)
    outcomes = np.array([sample["outcome"] for sample in all_samples])
    ranks = np.array([sample["rank"] for sample in all_samples], dtype=np.int64)
    steps = np.array([sample["step"] for sample in all_samples], dtype=np.int64)
    metadata = {
        "team_name": team_name,
        "log_dir": str(log_dir),
        "channel_names": list(CHANNEL_NAMES),
        "stats": stats,
        "options": options,
        "samples": [sample["metadata"] for sample in all_samples],
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
        rewards_proxy=rewards_proxy,
        outcomes=outcomes,
        ranks=ranks,
        steps=steps,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    return output_path, stats, len(all_samples)


def main():
    parser = argparse.ArgumentParser(description="Build imitation dataset from Bomberland replay logs.")
    parser.add_argument("--log_dir", default="logs/json")
    parser.add_argument("--team_name", default="HybridAgent")
    parser.add_argument("--output", required=True)
    parser.add_argument("--wins_only", type=str2bool, default=False)
    parser.add_argument("--max_samples_per_episode", type=int, default=None)
    parser.add_argument("--exclude_draws", type=str2bool, default=False)
    args = parser.parse_args()

    output_path, stats, total_samples = build_dataset(
        log_dir=args.log_dir,
        team_name=args.team_name,
        output=args.output,
        wins_only=args.wins_only,
        max_samples_per_episode=args.max_samples_per_episode,
        exclude_draws=args.exclude_draws,
    )
    print(f"Saved dataset: {output_path}")
    print(f"total_samples={total_samples}")
    print(f"episodes_used={stats['episodes_used']}")
    print(f"files_scanned={stats['files_scanned']}")
    print(
        "skipped: "
        f"malformed={stats['skipped_malformed_frames']}, "
        f"missing_action={stats['skipped_missing_action']}, "
        f"invalid_action={stats['skipped_invalid_action']}, "
        f"dead_agent={stats['skipped_dead_agent_frames']}, "
        f"no_team={stats['skipped_no_team']}, "
        f"filtered_episodes={stats['skipped_filtered_episodes']}"
    )


if __name__ == "__main__":
    main()
