from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import RandomAgent, SimpleRuleAgent
from agent.rl_agent_pure.action_mask import legal_action_mask
from agent.rl_agent_pure.constants import PLACE_BOMB
from agent.rl_agent_pure.encoder import encode_observation
from agent.rl_agent_pure.utils import compute_danger_map, normalize_obs
from competition.evaluation.runtime_guard import load_agent_instance
from engine.game import BomberEnv
from ml.evaluate_rl_pure import _clear_submission_import_state


def _load_agent(path_or_name, agent_id):
    if path_or_name in {"random", "RandomAgent"}:
        return RandomAgent(agent_id)
    if path_or_name in {"simple", "SimpleRuleAgent"}:
        return SimpleRuleAgent(agent_id)
    path = Path(path_or_name)
    if path.is_dir():
        path = path / "agent.py"
    _clear_submission_import_state()
    return load_agent_instance(str(path), agent_id)


def _box_count(obs):
    return int((np.asarray(obs["map"]) == 2).sum())


def collect(args):
    rng = random.Random(args.seed)
    env = BomberEnv(max_steps=args.max_steps, seed=args.seed)
    observations = []
    actions = []
    did_place_bomb = []
    is_bomb_action = []
    survived_after_bomb = []
    teacher_survived_after_bomb = []
    boxes_destroyed_after_bomb = []
    post_bomb_escape = []
    is_post_bomb_escape = []
    bomb_context_id = []
    action_after_bomb_step_delta = []
    current_danger = []
    future_danger = []
    steps = []
    reward_proxy = []
    sample_weight = []

    action_counts = Counter()
    skipped_illegal = 0
    next_bomb_context_id = 1

    for episode in range(args.episodes):
        teacher_slot = rng.randrange(4) if args.randomize_teacher_slot else args.teacher_id
        roster = []
        for idx in range(4):
            roster.append(args.teacher if idx == teacher_slot else rng.choice(args.opponents))
        agents = [_load_agent(path, idx) for idx, path in enumerate(roster)]
        obs = {**env.reset(seed=args.seed + episode), "step": 0}
        active_bombs = []

        for step in range(args.max_steps):
            action_list = []
            teacher_action = 0
            teacher_obs = obs
            for idx, agent in enumerate(agents):
                try:
                    action = int(agent.act(obs))
                except Exception:
                    action = 0
                if not 0 <= action <= 5:
                    action = 0
                if idx == teacher_slot:
                    teacher_action = action
                action_list.append(action)

            mask = legal_action_mask(teacher_obs, teacher_slot)
            if 0 <= teacher_action <= 5 and mask[teacher_action]:
                idx = len(actions)
                board, players, bombs, _ = normalize_obs(teacher_obs)
                row, col = int(players[teacher_slot, 0]), int(players[teacher_slot, 1])
                danger = compute_danger_map(board, players, bombs)
                active_context = active_bombs[0] if active_bombs else None
                observations.append(encode_observation(teacher_obs, teacher_slot))
                actions.append(teacher_action)
                action_counts[teacher_action] += 1
                did_place_bomb.append(1 if teacher_action == PLACE_BOMB else 0)
                is_bomb_action.append(1 if teacher_action == PLACE_BOMB else 0)
                survived_after_bomb.append(0)
                teacher_survived_after_bomb.append(0)
                boxes_destroyed_after_bomb.append(0)
                post_bomb_escape.append(1 if active_bombs else 0)
                is_post_bomb_escape.append(1 if active_bombs else 0)
                bomb_context_id.append(active_context["context_id"] if active_context else 0)
                action_after_bomb_step_delta.append(step - active_context["placed_step"] if active_context else -1)
                current_danger.append(int(danger[row, col]) if 0 <= row < danger.shape[0] and 0 <= col < danger.shape[1] else 9999)
                future_danger.append(int(np.min(danger[danger < 9999])) if np.any(danger < 9999) else 9999)
                steps.append(step)

                weight = 1.0
                if teacher_action == PLACE_BOMB:
                    weight = args.bomb_weight
                    active_bombs.append({
                        "context_id": next_bomb_context_id,
                        "sample_index": idx,
                        "escape_indices": [],
                        "placed_step": step,
                        "initial_boxes": _box_count(teacher_obs),
                    })
                    next_bomb_context_id += 1
                elif active_bombs:
                    weight = args.escape_weight
                    active_bombs[0]["escape_indices"].append(idx)
                sample_weight.append(weight)
                reward_proxy.append(0.0)
            else:
                skipped_illegal += 1

            next_obs, terminated, truncated = env.step(action_list)
            next_obs = {**next_obs, "step": step + 1}
            alive = bool(next_obs["players"][teacher_slot, 2])
            current_boxes = _box_count(next_obs)
            for event in active_bombs:
                if step + 1 - event["placed_step"] < args.bomb_window:
                    continue
                sample_idx = event["sample_index"]
                related_indices = [sample_idx, *event.get("escape_indices", [])]
                survived_after_bomb[sample_idx] = 1 if alive else 0
                for related_idx in related_indices:
                    teacher_survived_after_bomb[related_idx] = 1 if alive else 0
                destroyed = max(0, event["initial_boxes"] - current_boxes)
                for related_idx in related_indices:
                    boxes_destroyed_after_bomb[related_idx] = destroyed
                    reward_proxy[related_idx] = float(destroyed * 25 + (10 if alive else -50))
            active_bombs = [
                event for event in active_bombs
                if step + 1 - event["placed_step"] < args.bomb_window
            ]
            obs = next_obs
            if terminated or truncated:
                break

        if (episode + 1) % max(1, args.log_every) == 0:
            print(f"episode={episode + 1}/{args.episodes} samples={len(actions)} actions={dict(action_counts)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        did_place_bomb=np.asarray(did_place_bomb, dtype=np.int8),
        is_bomb_action=np.asarray(is_bomb_action, dtype=np.int8),
        survived_after_bomb=np.asarray(survived_after_bomb, dtype=np.int8),
        teacher_survived_after_bomb=np.asarray(teacher_survived_after_bomb, dtype=np.int8),
        boxes_destroyed_after_bomb=np.asarray(boxes_destroyed_after_bomb, dtype=np.int16),
        post_bomb_escape=np.asarray(post_bomb_escape, dtype=np.int8),
        is_post_bomb_escape=np.asarray(is_post_bomb_escape, dtype=np.int8),
        bomb_context_id=np.asarray(bomb_context_id, dtype=np.int32),
        action_after_bomb_step_delta=np.asarray(action_after_bomb_step_delta, dtype=np.int16),
        current_danger=np.asarray(current_danger, dtype=np.int16),
        future_danger=np.asarray(future_danger, dtype=np.int16),
        step=np.asarray(steps, dtype=np.int16),
        reward_proxy=np.asarray(reward_proxy, dtype=np.float32),
        sample_weight=np.asarray(sample_weight, dtype=np.float32),
    )
    stats = {
        "samples": len(actions),
        "action_distribution": {str(k): int(v) for k, v in sorted(action_counts.items())},
        "bomb_samples": int(action_counts[PLACE_BOMB]),
        "bomb_fraction": float(action_counts[PLACE_BOMB] / max(1, len(actions))),
        "post_bomb_escape_samples": int(sum(post_bomb_escape)),
        "survived_bomb_context_samples": int(sum(teacher_survived_after_bomb)),
        "avg_boxes_destroyed_after_bomb": float(np.mean(boxes_destroyed_after_bomb)) if boxes_destroyed_after_bomb else 0.0,
        "skipped_illegal": int(skipped_illegal),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="agent/hybrid_agent_online_robust")
    parser.add_argument("--teacher_id", type=int, default=0)
    parser.add_argument("--randomize_teacher_slot", action="store_true")
    parser.add_argument("--opponents", nargs="+", default=["random", "simple"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="ml/datasets/rl_bc_stage1.npz")
    parser.add_argument("--bomb_window", type=int, default=8)
    parser.add_argument("--bomb_weight", type=float, default=12.0)
    parser.add_argument("--escape_weight", type=float, default=4.0)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
