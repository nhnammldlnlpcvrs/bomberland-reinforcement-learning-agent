from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.action_mask import legal_action_mask
from agent.rl_agent_pure.constants import BOMB_TIMER, MOVE_ACTIONS, PLACE_BOMB
from agent.rl_agent_pure.encoder import encode_observation
from agent.rl_agent_pure.utils import (
    bomb_positions,
    boxes_in_blast,
    compute_danger_map,
    has_escape_after_bomb,
    normalize_obs,
    reachable_area,
)
from engine.game import BomberEnv
from ml.evaluate_rl_pure import OPPONENTS, make_eval_agents, prepare_agent_path

REASON_CODES = {
    "observed_death_after_bomb": 1,
    "no_escape_after_bomb": 2,
    "dangerous_current_or_future": 3,
    "no_bomb_value": 4,
    "trapped_after_bomb": 5,
}


def _load_policy(path: str, device: str):
    try:
        return PPO.load(path, device=device)
    except Exception:
        return None


def _bomb_probability(model, encoded_obs: np.ndarray, device: str) -> float:
    if model is None:
        return 0.0
    with torch.no_grad():
        obs = torch.as_tensor(encoded_obs[None], dtype=torch.float32, device=device)
        dist = model.policy.get_distribution(obs)
        probs = dist.distribution.probs.detach().cpu().numpy()[0]
    return float(probs[PLACE_BOMB])


def _enemy_in_blast(board, players, row: int, col: int, agent_id: int) -> bool:
    radius = 1 + max(0, int(players[agent_id, 4]))
    for enemy_id, player in enumerate(players):
        if enemy_id == agent_id or not int(player[2]):
            continue
        erow, ecol = int(player[0]), int(player[1])
        if erow == row and abs(ecol - col) <= radius:
            return True
        if ecol == col and abs(erow - row) <= radius:
            return True
    return False


def _post_bomb_reachable_area(board, players, bombs, agent_id: int) -> int:
    row, col = int(players[agent_id, 0]), int(players[agent_id, 1])
    placed = np.array([[row, col, BOMB_TIMER, agent_id]], dtype=np.int16)
    sim_bombs = placed if bombs.size == 0 else np.vstack([bombs, placed])
    danger = compute_danger_map(board, players, sim_bombs)
    area = reachable_area(board, bomb_positions(sim_bombs), danger, (row, col), max_depth=10)
    return int(area.sum())


def _candidate_features(obs, agent_id: int, encoded_obs: np.ndarray, bomb_prob: float, max_steps: int, min_policy_bomb_prob: float):
    board, players, bombs, step = normalize_obs(obs)
    row, col = int(players[agent_id, 0]), int(players[agent_id, 1])
    danger = compute_danger_map(board, players, bombs)
    current_danger = int(danger[row, col]) if 0 <= row < danger.shape[0] and 0 <= col < danger.shape[1] else 0
    has_escape = has_escape_after_bomb(board, players, bombs, agent_id)
    nearby_boxes = int(boxes_in_blast(board, players, row, col, agent_id))
    enemy_line = bool(_enemy_in_blast(board, players, row, col, agent_id))
    post_area = _post_bomb_reachable_area(board, players, bombs, agent_id)
    has_bomb_value_proxy = nearby_boxes > 0 or enemy_line
    tempting = has_bomb_value_proxy or bomb_prob >= min_policy_bomb_prob
    reasons = []
    if not has_escape:
        reasons.append(REASON_CODES["no_escape_after_bomb"])
    if current_danger <= 3:
        reasons.append(REASON_CODES["dangerous_current_or_future"])
    if nearby_boxes <= 0 and not enemy_line:
        reasons.append(REASON_CODES["no_bomb_value"])
    if post_area <= 2:
        reasons.append(REASON_CODES["trapped_after_bomb"])
    scalars = np.asarray([
        current_danger / 9999.0,
        current_danger / 9999.0,
        nearby_boxes / 7.0,
        step / max(1, max_steps),
        post_area / 10.0,
    ], dtype=np.float32)
    return {
        "encoded_obs": encoded_obs,
        "tempting": bool(tempting),
        "unsafe": bool(reasons),
        "reason": int(reasons[0]) if reasons else 0,
        "nearby_box_count": nearby_boxes,
        "predicted_bomb_prob": float(bomb_prob),
        "current_danger": current_danger,
        "future_danger": current_danger,
        "step": int(step),
        "post_bomb_reachable_area": post_area,
        "has_bomb_value_proxy": int(has_bomb_value_proxy),
        "scalar_features": scalars,
    }


def _append_sample(samples: list[dict], features: dict, reason: int | None = None, observed: bool = False):
    item = dict(features)
    if reason is not None:
        item["reason"] = int(reason)
    item["observed"] = int(observed)
    samples.append(item)


def mine(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    policy_agent_path = prepare_agent_path(args.policy)
    prob_model = _load_policy(args.policy, args.device)
    opponent_paths = [OPPONENTS.get(name, name) for name in args.opponents]
    samples: list[dict] = []
    reason_counts = {name: 0 for name in REASON_CODES}
    env = BomberEnv(max_steps=args.max_steps, seed=args.seed)

    for ep in range(args.episodes):
        opponent = opponent_paths[ep % len(opponent_paths)]
        roster = [policy_agent_path, opponent, opponent, opponent]
        agents = make_eval_agents(roster, seed=args.seed + ep)
        obs = {**env.reset(seed=args.seed + ep), "step": 0}
        pending_bombs: list[dict] = []
        done = False
        step = 0
        while not done and step < args.max_steps:
            encoded = encode_observation(obs, 0)
            features = _candidate_features(obs, 0, encoded, 0.0, args.max_steps, args.min_policy_bomb_prob)
            if args.use_policy_bomb_prob and not features["has_bomb_value_proxy"]:
                bomb_prob = _bomb_probability(prob_model, encoded, args.device)
                features = _candidate_features(obs, 0, encoded, bomb_prob, args.max_steps, args.min_policy_bomb_prob)
            mask = legal_action_mask(obs, 0)
            if mask[PLACE_BOMB] and features["tempting"] and features["unsafe"]:
                if features["reason"] == REASON_CODES["no_bomb_value"] and bomb_prob < args.min_no_value_bomb_prob:
                    continue
                _append_sample(samples, features)
                reason_name = next((k for k, v in REASON_CODES.items() if v == features["reason"]), "unknown")
                reason_counts[reason_name] = reason_counts.get(reason_name, 0) + 1

            actions = []
            policy_action = 0
            for idx, agent in enumerate(agents):
                try:
                    action = int(agent.act(obs))
                except Exception:
                    action = 0
                if not 0 <= action <= 5:
                    action = 0
                actions.append(action)
                if idx == 0:
                    policy_action = action

            if policy_action == PLACE_BOMB and mask[PLACE_BOMB]:
                pending_bombs.append({"step": step, "features": features, "death_recorded": False})

            obs, terminated, truncated = env.step(actions)
            step += 1
            obs = {**obs, "step": step}
            done = terminated or truncated
            alive = bool(obs["players"][0, 2])
            for event in pending_bombs:
                age = step - int(event["step"])
                if not event["death_recorded"] and not alive and 0 <= age <= 7:
                    event["death_recorded"] = True
                    _append_sample(samples, event["features"], REASON_CODES["observed_death_after_bomb"], observed=True)
                    reason_counts["observed_death_after_bomb"] += 1
        if args.max_samples and len(samples) >= args.max_samples:
            break

    if args.max_samples and len(samples) > args.max_samples:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No hard negatives mined")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=np.stack([s["encoded_obs"] for s in samples]).astype(np.float32),
        scalar_features=np.stack([s["scalar_features"] for s in samples]).astype(np.float32),
        labels=np.zeros(len(samples), dtype=np.float32),
        source=np.full(len(samples), 4, dtype=np.int8),
        negative_reason=np.asarray([s["reason"] for s in samples], dtype=np.int16),
        nearby_box_count=np.asarray([s["nearby_box_count"] for s in samples], dtype=np.int16),
        predicted_bomb_prob=np.asarray([s["predicted_bomb_prob"] for s in samples], dtype=np.float32),
        current_danger=np.asarray([s["current_danger"] for s in samples], dtype=np.int16),
        future_danger=np.asarray([s["future_danger"] for s in samples], dtype=np.int16),
        step=np.asarray([s["step"] for s in samples], dtype=np.int16),
        post_bomb_reachable_area=np.asarray([s["post_bomb_reachable_area"] for s in samples], dtype=np.int16),
        has_bomb_value_proxy=np.asarray([s["has_bomb_value_proxy"] for s in samples], dtype=np.int8),
        observed_outcome=np.asarray([s["observed"] for s in samples], dtype=np.int8),
    )
    stats = {
        "policy": args.policy,
        "opponents": args.opponents,
        "episodes": args.episodes,
        "output": str(output),
        "samples": int(len(samples)),
        "observed_unsafe_count": int(sum(s["observed"] for s in samples)),
        "reason_counts": reason_counts,
        "nearby_box_mean": float(np.mean([s["nearby_box_count"] for s in samples])),
        "value_proxy_count": int(sum(s["has_bomb_value_proxy"] for s in samples)),
        "predicted_bomb_prob_mean": float(np.mean([s["predicted_bomb_prob"] for s in samples])),
    }
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Mine rollout-derived tempting unsafe bomb contexts for offline selector training.")
    parser.add_argument("--policy", default="ml/checkpoints/rl_agent_pure/ppo_useful_safe_bomb_best.zip")
    parser.add_argument("--opponents", nargs="+", default=["random", "simple"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--output", default="ml/datasets/bomb_hard_negatives.npz")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--min_policy_bomb_prob", type=float, default=0.02)
    parser.add_argument("--min_no_value_bomb_prob", type=float, default=0.05)
    parser.add_argument("--use_policy_bomb_prob", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    mine(args)


if __name__ == "__main__":
    main()
