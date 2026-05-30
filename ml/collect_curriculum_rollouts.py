from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.model import BomberFeaturesExtractor
from agent.rl_agent_pure.utils import bomb_positions, compute_danger_map, normalize_obs, reachable_area
from ml.envs.bomber_curriculum_env import CURRICULUM_MODES, BomberCurriculumEnv


MODE_TO_ID = {"escape_only": 0, "bomb_then_escape": 1, "bomb_box_value": 2, "full_game_mix": 3}


def _reachable_count(obs_dict, agent_id: int) -> int:
    board, players, bombs, _ = normalize_obs(obs_dict)
    pos = (int(players[agent_id, 0]), int(players[agent_id, 1]))
    danger = compute_danger_map(board, players, bombs)
    return int(reachable_area(board, bomb_positions(bombs), danger, pos, max_depth=10).sum())


def _own_bomb_recent(obs_dict, agent_id: int) -> bool:
    _board, _players, bombs, _ = normalize_obs(obs_dict)
    for bomb in bombs:
        if int(bomb[3]) == int(agent_id) and int(bomb[2]) <= 7:
            return True
    return False


def _discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        out[idx] = running
    return out


def _postprocess_episode(episode: list[dict], gamma: float) -> None:
    rewards = [float(row["reward"]) for row in episode]
    returns = _discounted_returns(rewards, gamma)
    alive = [bool(row["alive"]) for row in episode]
    boxes = [int(row["boxes_destroyed"]) for row in episode]
    for idx, row in enumerate(episode):
        horizon = episode[idx:min(len(episode), idx + 8)]
        death_future = any(not bool(item["alive"]) for item in horizon)
        row["death_within_7"] = int(death_future and alive[idx])
        row["boxes_destroyed_future"] = int(max(boxes[idx:min(len(boxes), idx + 8)] or [boxes[idx]]) - boxes[idx])
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
    policy_kwargs = {
        "features_extractor_class": BomberFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    model = PPO.load(args.policy, device=args.device, custom_objects={"policy_kwargs": policy_kwargs})
    modes = args.modes
    if "mixed" in modes:
        modes = ["escape_only", "bomb_then_escape", "bomb_box_value", "full_game_mix"]

    rows: list[dict] = []
    mode_counts = {mode: 0 for mode in modes}
    for mode in modes:
        env = BomberCurriculumEnv(
            mode=mode,
            agent_id=args.agent_id,
            opponent_pool=args.opponents,
            max_steps=args.max_steps,
            seed=args.seed,
            retain_full_game_ratio=0.0,
            training_bomb_gate=False,
        )
        for episode_idx in range(args.episodes_per_mode):
            obs, info = env.reset(seed=args.seed + episode_idx)
            episode_rows = []
            done = False
            truncated = False
            while not (done or truncated):
                action, _state = model.predict(obs, deterministic=not args.stochastic)
                action = int(np.asarray(action).reshape(-1)[0])
                prev_obs_dict = env.base.last_obs
                prev_reachable = _reachable_count(prev_obs_dict, args.agent_id)
                prev_corridor = bool(info.get("in_blast_corridor", False))
                next_obs, reward, done, truncated, next_info = env.step(action)
                next_obs_dict = env.base.last_obs
                next_reachable = _reachable_count(next_obs_dict, args.agent_id)
                current_corridor = bool(next_info.get("in_blast_corridor", False))
                alive = bool(next_info.get("alive", False))
                bomb_age = int(next_info.get("bomb_age", -1))
                own_bomb_recent = bool(0 <= bomb_age <= 7) or _own_bomb_recent(next_obs_dict, args.agent_id)
                row = {
                    "obs": obs.astype(np.float32),
                    "action": action,
                    "next_obs": next_obs.astype(np.float32),
                    "reward": float(reward),
                    "done": bool(done or truncated),
                    "mode": mode,
                    "mode_id": MODE_TO_ID.get(str(next_info.get("curriculum_mode", mode)), MODE_TO_ID.get(mode, 3)),
                    "in_blast_corridor": int(current_corridor),
                    "escaped_blast": int(prev_corridor and not current_corridor and alive),
                    "own_bomb_recent": int(own_bomb_recent),
                    "boxes_destroyed": int(next_info.get("boxes_destroyed", 0)),
                    "reachable_area_delta": float(next_reachable - prev_reachable),
                    "alive": alive,
                    "step": int(next_info.get("survival_step", 0)),
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
    }
    np.savez_compressed(output, **arrays)
    stats = {
        "samples": len(rows),
        "episodes_per_mode": args.episodes_per_mode,
        "mode_counts": mode_counts,
        "death_within_7_rate": float(arrays["death_within_7"].mean()) if rows else 0.0,
        "escaped_blast_rate": float(arrays["escaped_blast"].mean()) if rows else 0.0,
        "own_bomb_recent_rate": float(arrays["own_bomb_recent"].mean()) if rows else 0.0,
        "boxes_destroyed_future_mean": float(arrays["boxes_destroyed_future"].mean()) if rows else 0.0,
        "reachable_area_delta_mean": float(arrays["reachable_area_delta"].mean()) if rows else 0.0,
    }
    meta_path = output.with_suffix(".json")
    meta_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Collect curriculum rollout dataset for auxiliary heads.")
    parser.add_argument("--policy", default="ml/checkpoints/rl_agent_pure/ppo_bomb_selector_best.zip")
    parser.add_argument("--output", default="ml/datasets/curriculum_rollouts.npz")
    parser.add_argument("--modes", nargs="+", default=["escape_only", "bomb_then_escape", "bomb_box_value", "full_game_mix"], choices=CURRICULUM_MODES)
    parser.add_argument("--episodes_per_mode", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=6100)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--opponents", nargs="*", default=["random", "simple"])
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
