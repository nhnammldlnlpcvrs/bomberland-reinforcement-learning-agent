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
from agent.rl_agent_pure.constants import PLACE_BOMB
from agent.rl_agent_pure.encoder import encode_observation
from agent.rl_agent_pure.utils import compute_danger_map, normalize_obs
from competition.evaluation.runtime_guard import load_agent_instance
from engine.game import BomberEnv
from ml.evaluate_rl_pure import _clear_submission_import_state


def _load_agent(path_or_name: str, agent_id: int):
    if path_or_name in {"random", "RandomAgent"}:
        return RandomAgent(agent_id)
    if path_or_name in {"simple", "SimpleRuleAgent"}:
        return SimpleRuleAgent(agent_id)
    path = Path(path_or_name)
    if path.is_dir():
        path = path / "agent.py"
    _clear_submission_import_state()
    return load_agent_instance(str(path), agent_id)


def _safe_action(agent, obs) -> int:
    try:
        action = int(agent.act(obs))
    except Exception:
        return 0
    return action if 0 <= action <= 5 else 0


def _danger_at(obs, agent_id: int) -> int:
    try:
        board, players, bombs, _ = normalize_obs(obs)
        row, col = int(players[agent_id, 0]), int(players[agent_id, 1])
        danger = compute_danger_map(board, players, bombs)
        return int(danger[row, col])
    except Exception:
        return 9999


def collect(args):
    rng = random.Random(args.seed)
    env = BomberEnv(max_steps=args.max_steps, seed=args.seed)
    episodes_obs = []
    episodes_actions = []
    episodes_steps = []
    episodes_alive = []
    episodes_danger = []
    episode_lengths = []
    outcomes = []
    action_counts = Counter()
    teacher_wins = teacher_deaths = teacher_draws = 0
    total_survival = 0
    total_bombs = 0
    total_post_bomb = 0

    for episode in range(args.episodes):
        teacher_slot = rng.randrange(4) if args.randomize_teacher_slot else args.teacher_id
        roster = [args.teacher if idx == teacher_slot else rng.choice(args.opponents) for idx in range(4)]
        agents = [_load_agent(path, idx) for idx, path in enumerate(roster)]
        obs = {**env.reset(seed=args.seed + episode), "step": 0}
        seq_obs = []
        seq_actions = []
        seq_steps = []
        seq_alive = []
        seq_danger = []
        recent_bomb_window = 0

        for step in range(args.max_steps):
            actions = [_safe_action(agent, obs) for agent in agents]
            teacher_action = actions[teacher_slot]
            seq_obs.append(encode_observation(obs, teacher_slot))
            seq_actions.append(teacher_action)
            seq_steps.append(step)
            seq_alive.append(int(bool(obs["players"][teacher_slot, 2])))
            seq_danger.append(_danger_at(obs, teacher_slot))
            action_counts[teacher_action] += 1
            if teacher_action == PLACE_BOMB:
                total_bombs += 1
                recent_bomb_window = args.post_bomb_window
            elif recent_bomb_window > 0:
                total_post_bomb += 1
                recent_bomb_window -= 1

            next_obs, terminated, truncated = env.step(actions)
            obs = {**next_obs, "step": step + 1}
            if terminated or truncated:
                break

        alive = bool(obs["players"][teacher_slot, 2])
        survivors = [idx for idx, player in enumerate(obs["players"]) if bool(player[2])]
        if alive and survivors == [teacher_slot]:
            outcome = "win"
            teacher_wins += 1
        elif alive:
            outcome = "draw"
            teacher_draws += 1
        else:
            outcome = "death"
            teacher_deaths += 1
        total_survival += len(seq_actions)
        episodes_obs.append(np.asarray(seq_obs, dtype=np.float32))
        episodes_actions.append(np.asarray(seq_actions, dtype=np.int64))
        episodes_steps.append(np.asarray(seq_steps, dtype=np.int16))
        episodes_alive.append(np.asarray(seq_alive, dtype=np.int8))
        episodes_danger.append(np.asarray(seq_danger, dtype=np.int16))
        episode_lengths.append(len(seq_actions))
        outcomes.append(outcome)

        if (episode + 1) % max(1, args.log_every) == 0:
            print(
                f"episode={episode + 1}/{args.episodes} "
                f"steps={sum(episode_lengths)} actions={dict(action_counts)}"
            )

    max_len = max(episode_lengths) if episode_lengths else 0
    n = len(episode_lengths)
    observations = np.zeros((n, max_len, 19, 13, 13), dtype=np.float32)
    actions = np.zeros((n, max_len), dtype=np.int64)
    valid_mask = np.zeros((n, max_len), dtype=np.bool_)
    episode_starts = np.zeros((n, max_len), dtype=np.bool_)
    steps = np.zeros((n, max_len), dtype=np.int16)
    alive = np.zeros((n, max_len), dtype=np.int8)
    current_danger = np.full((n, max_len), 9999, dtype=np.int16)
    is_bomb_action = np.zeros((n, max_len), dtype=np.bool_)
    is_post_bomb_escape = np.zeros((n, max_len), dtype=np.bool_)
    for idx, (obs_seq, act_seq, step_seq, alive_seq, danger_seq) in enumerate(
        zip(episodes_obs, episodes_actions, episodes_steps, episodes_alive, episodes_danger)
    ):
        length = len(act_seq)
        observations[idx, :length] = obs_seq
        actions[idx, :length] = act_seq
        valid_mask[idx, :length] = True
        episode_starts[idx, 0] = True
        steps[idx, :length] = step_seq
        alive[idx, :length] = alive_seq
        current_danger[idx, :length] = danger_seq
        is_bomb_action[idx, :length] = act_seq == PLACE_BOMB
        for t, action in enumerate(act_seq):
            if action == PLACE_BOMB:
                end = min(length, t + 1 + args.post_bomb_window)
                is_post_bomb_escape[idx, t + 1:end] = True

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=observations,
        actions=actions,
        valid_mask=valid_mask,
        episode_starts=episode_starts,
        lengths=np.asarray(episode_lengths, dtype=np.int16),
        steps=steps,
        alive=alive,
        current_danger=current_danger,
        is_bomb_action=is_bomb_action,
        is_post_bomb_escape=is_post_bomb_escape,
        outcomes=np.asarray(outcomes),
    )
    total_steps = int(sum(episode_lengths))
    stats = {
        "episodes": n,
        "total_steps": total_steps,
        "max_len": int(max_len),
        "action_distribution": {str(k): int(v) for k, v in sorted(action_counts.items())},
        "place_bomb_fraction": float(action_counts[PLACE_BOMB] / max(1, total_steps)),
        "post_bomb_escape_fraction": float(total_post_bomb / max(1, total_steps)),
        "avg_survival_step": float(total_survival / max(1, n)),
        "teacher_win_rate": float(teacher_wins / max(1, n)),
        "teacher_draw_rate": float(teacher_draws / max(1, n)),
        "teacher_death_rate": float(teacher_deaths / max(1, n)),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Collect episode-preserving BC trajectories for RecurrentPPO.")
    parser.add_argument("--teacher", default="agent/hybrid_agent_online_robust")
    parser.add_argument("--teacher_id", type=int, default=0)
    parser.add_argument("--randomize_teacher_slot", action="store_true")
    parser.add_argument("--opponents", nargs="+", default=["random", "simple"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=9800)
    parser.add_argument("--output", default="ml/datasets/recurrent_bc_online_robust.npz")
    parser.add_argument("--post_bomb_window", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
