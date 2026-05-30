from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.action_mask import legal_action_mask
from agent.rl_agent_pure.constants import MOVE_ACTIONS, MOVE_DELTAS, PLACE_BOMB, STOP
from agent.rl_agent_pure.model import BomberFeaturesExtractor
from agent.rl_agent_pure.utils import bomb_positions, boxes_in_blast, compute_danger_map, normalize_obs, reachable_area
from ml.envs.bomber_curriculum_env import BomberCurriculumEnv
from ml.envs.curriculum_scenarios import has_escape_after_bomb, is_in_blast_corridor, reachable_safe_tiles


MODE_TO_ID = {"escape_only": 0, "bomb_then_escape": 1, "bomb_box_value": 2, "full_game_mix": 3}


def _counterfactual_labels(obs_dict: dict, agent_id: int) -> dict:
    board, players, bombs, _step = normalize_obs(obs_dict)
    row, col = int(players[agent_id, 0]), int(players[agent_id, 1])
    danger = compute_danger_map(board, players, bombs)
    safe_now = danger[row, col] > 1
    safe_tiles_now = int(reachable_area(board, bomb_positions(bombs), danger, (row, col), max_depth=10).sum())
    safe_tiles_after = int(reachable_safe_tiles(obs_dict, (row, col), agent_id, max_depth=10, after_placing_bomb=True).sum())
    escape_after = bool(has_escape_after_bomb(obs_dict, agent_id))
    would_destroy = int(boxes_in_blast(board, players, row, col, agent_id))
    in_corridor = bool(is_in_blast_corridor(obs_dict, (row, col), agent_id))
    blast_distance = 99
    for bomb in bombs:
        if int(bomb[0]) == row:
            blast_distance = min(blast_distance, abs(int(bomb[1]) - col))
        if int(bomb[1]) == col:
            blast_distance = min(blast_distance, abs(int(bomb[0]) - row))
    return {
        "has_escape_path_now": int(safe_tiles_now > 1 and safe_now),
        "has_escape_after_bomb": int(escape_after),
        "would_destroy_boxes_if_bomb": would_destroy,
        "in_future_blast": int(danger[row, col] <= 7),
        "trapped_if_bomb": int(not escape_after),
        "safe_tiles_after_bomb_count": safe_tiles_after,
        "blast_corridor_distance": int(blast_distance if blast_distance != 99 else -1),
    }


def _scripted_action(obs_dict: dict, agent_id: int, mode: str, rng: random.Random) -> int:
    mask = legal_action_mask(obs_dict, agent_id)
    labels = _counterfactual_labels(obs_dict, agent_id)
    if mode in {"bomb_then_escape", "bomb_box_value"} and bool(mask[PLACE_BOMB]):
        if labels["has_escape_after_bomb"] and labels["would_destroy_boxes_if_bomb"] > 0:
            return PLACE_BOMB
    board, players, bombs, _step = normalize_obs(obs_dict)
    row, col = int(players[agent_id, 0]), int(players[agent_id, 1])
    danger = compute_danger_map(board, players, bombs)
    best_action = STOP
    best_score = -1e9
    for action in MOVE_ACTIONS:
        if not bool(mask[action]):
            continue
        dr, dc = MOVE_DELTAS[action]
        nr, nc = row + dr, col + dc
        score = float(danger[nr, nc]) + rng.random() * 0.1
        if score > best_score:
            best_action = int(action)
            best_score = score
    if best_action == STOP:
        valid = np.flatnonzero(mask)
        return int(rng.choice([int(v) for v in valid])) if valid.size else STOP
    return best_action


def _discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        out[idx] = running
    return out


def _postprocess_episode(rows: list[dict], gamma: float) -> None:
    rewards = [float(row["reward"]) for row in rows]
    returns = _discounted_returns(rewards, gamma)
    alive = [bool(row["alive"]) for row in rows]
    boxes = [int(row["boxes_destroyed"]) for row in rows]
    corridors = [bool(row["in_blast_corridor"]) for row in rows]
    for idx, row in enumerate(rows):
        horizon = rows[idx:min(len(rows), idx + 8)]
        row["death_within_7"] = int(alive[idx] and any(not bool(item["alive"]) for item in horizon))
        row["boxes_destroyed_future"] = int(max(boxes[idx:min(len(boxes), idx + 8)] or [boxes[idx]]) - boxes[idx])
        row["escaped_blast"] = int(corridors[idx] and any(bool(item["alive"]) and not bool(item["in_blast_corridor"]) for item in horizon[1:]))
        row["post_bomb_survival_steps"] = 0
        if row["own_bomb_recent"]:
            count = 0
            for item in horizon:
                if item["alive"]:
                    count += 1
                else:
                    break
            row["post_bomb_survival_steps"] = count
        row["discounted_return"] = float(returns[idx])


def collect(args):
    rng = random.Random(args.seed)
    policy_kwargs = {
        "features_extractor_class": BomberFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    policy = PPO.load(args.policy, device=args.device, custom_objects={"policy_kwargs": policy_kwargs})
    rows = []
    mode_counts = {mode: 0 for mode in args.modes}
    for mode in args.modes:
        env = BomberCurriculumEnv(
            mode=mode,
            agent_id=args.agent_id,
            opponent_pool=args.opponents,
            max_steps=args.max_steps,
            seed=args.seed,
            training_bomb_gate=False,
        )
        for episode in range(args.episodes_per_mode):
            obs, info = env.reset(seed=args.seed + episode)
            episode_rows = []
            done = False
            truncated = False
            while not (done or truncated):
                obs_dict = env.base.last_obs
                labels = _counterfactual_labels(obs_dict, args.agent_id)
                if rng.random() < args.scripted_action_prob:
                    action = _scripted_action(obs_dict, args.agent_id, mode, rng)
                else:
                    action, _state = policy.predict(obs, deterministic=False)
                    action = int(np.asarray(action).reshape(-1)[0])
                prev_reach = labels["safe_tiles_after_bomb_count"]
                next_obs, reward, done, truncated, next_info = env.step(action)
                next_labels = _counterfactual_labels(env.base.last_obs, args.agent_id)
                row = {
                    "obs": obs.astype(np.float32),
                    "action": int(action),
                    "next_obs": next_obs.astype(np.float32),
                    "reward": float(reward),
                    "done": bool(done or truncated),
                    "mode": mode,
                    "mode_id": MODE_TO_ID[mode],
                    "in_blast_corridor": int(next_info.get("in_blast_corridor", 0)),
                    "own_bomb_recent": int((0 <= int(next_info.get("bomb_age", -1)) <= 7) or next_labels["in_future_blast"]),
                    "boxes_destroyed": int(next_info.get("boxes_destroyed", 0)),
                    "reachable_area_delta": float(next_labels["safe_tiles_after_bomb_count"] - prev_reach),
                    "alive": bool(next_info.get("alive", False)),
                    **labels,
                }
                episode_rows.append(row)
                obs = next_obs
                info = next_info
            _postprocess_episode(episode_rows, args.gamma)
            rows.extend(episode_rows)
            mode_counts[mode] += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "observations": np.asarray([row["obs"] for row in rows], dtype=np.float32),
        "actions": np.asarray([row["action"] for row in rows], dtype=np.int64),
        "next_observations": np.asarray([row["next_obs"] for row in rows], dtype=np.float32),
        "rewards": np.asarray([row["reward"] for row in rows], dtype=np.float32),
        "dones": np.asarray([row["done"] for row in rows], dtype=np.int8),
        "mode_ids": np.asarray([row["mode_id"] for row in rows], dtype=np.int64),
        "death_within_7": np.asarray([row["death_within_7"] for row in rows], dtype=np.float32),
        "in_blast_corridor": np.asarray([row["in_blast_corridor"] for row in rows], dtype=np.float32),
        "escaped_blast": np.asarray([row["escaped_blast"] for row in rows], dtype=np.float32),
        "own_bomb_recent": np.asarray([row["own_bomb_recent"] for row in rows], dtype=np.float32),
        "boxes_destroyed_future": np.asarray([row["boxes_destroyed_future"] for row in rows], dtype=np.float32),
        "post_bomb_survival_steps": np.asarray([row["post_bomb_survival_steps"] for row in rows], dtype=np.float32),
        "reachable_area_delta": np.asarray([row["reachable_area_delta"] for row in rows], dtype=np.float32),
        "discounted_returns": np.asarray([row["discounted_return"] for row in rows], dtype=np.float32),
        "has_escape_path_now": np.asarray([row["has_escape_path_now"] for row in rows], dtype=np.float32),
        "has_escape_after_bomb": np.asarray([row["has_escape_after_bomb"] for row in rows], dtype=np.float32),
        "would_destroy_boxes_if_bomb": np.asarray([row["would_destroy_boxes_if_bomb"] for row in rows], dtype=np.float32),
        "in_future_blast": np.asarray([row["in_future_blast"] for row in rows], dtype=np.float32),
        "trapped_if_bomb": np.asarray([row["trapped_if_bomb"] for row in rows], dtype=np.float32),
        "safe_tiles_after_bomb_count": np.asarray([row["safe_tiles_after_bomb_count"] for row in rows], dtype=np.float32),
        "blast_corridor_distance": np.asarray([row["blast_corridor_distance"] for row in rows], dtype=np.float32),
    }
    np.savez_compressed(output, **arrays)
    stats = {
        "samples": len(rows),
        "mode_counts": mode_counts,
        "death_within_7_rate": float(arrays["death_within_7"].mean()),
        "escaped_blast_rate": float(arrays["escaped_blast"].mean()),
        "own_bomb_recent_rate": float(arrays["own_bomb_recent"].mean()),
        "has_escape_after_bomb_rate": float(arrays["has_escape_after_bomb"].mean()),
        "trapped_if_bomb_rate": float(arrays["trapped_if_bomb"].mean()),
        "would_destroy_boxes_mean": float(arrays["would_destroy_boxes_if_bomb"].mean()),
    }
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Collect targeted auxiliary curriculum data with counterfactual labels.")
    parser.add_argument("--policy", default="ml/checkpoints/rl_agent_pure/ppo_bomb_selector_best.zip")
    parser.add_argument("--output", default="ml/datasets/curriculum_aux_targeted.npz")
    parser.add_argument("--modes", nargs="+", default=["escape_only", "bomb_then_escape", "bomb_box_value"])
    parser.add_argument("--episodes_per_mode", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--scripted_action_prob", type=float, default=0.7)
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7100)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponents", nargs="*", default=["random", "simple"])
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
