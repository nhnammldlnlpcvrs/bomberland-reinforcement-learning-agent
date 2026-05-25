"""Evaluate rl_pure against baselines and online_robust with runtime metrics."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import BomberEnv
from scripts.participant.run_local_match import make_agents


OPPONENT_PATHS = {
    "random": "RandomAgent",
    "simple": "SimpleRuleAgent",
    "box_farmer": "BoxFarmerAgent",
    "smarter": "SmarterRuleAgent",
    "tactical": "TacticalRuleAgent",
    "genius": "GeniusRuleAgent",
    "online_robust": "agent/hybrid_agent_online_robust",
}


def ranks_from_deaths(prev_alive, alive, death_groups, step, survival_steps):
    deaths = []
    for idx, (before, now) in enumerate(zip(prev_alive, alive)):
        if before and not now:
            deaths.append(idx)
            survival_steps[idx] = step
    if deaths:
        death_groups.append(deaths)


def final_ranks(death_groups, survivors):
    ranks = [0, 0, 0, 0]
    ordered = list(death_groups)
    if survivors:
        ordered.append(list(survivors))
    for rank, group in enumerate(reversed(ordered)):
        for idx in group:
            ranks[idx] = rank
    return ranks


def run_suite(agent_path, opponent, episodes, max_steps, seed):
    env = BomberEnv(max_steps=max_steps, seed=seed)
    wins = draws = losses = deaths = self_kills = 0
    ranks = []
    survival = []
    inference_ms = []
    boxes_destroyed = []
    item_delta = []
    opp_path = OPPONENT_PATHS.get(opponent, opponent)
    for ep in range(episodes):
        positions = [agent_path, opp_path, opp_path, opp_path]
        rng = random.Random(seed + ep)
        rng.shuffle(positions)
        agent_slot = positions.index(agent_path)
        agents, _names = make_agents(positions, seed=seed + ep)
        obs = env.reset(seed=seed + ep)
        initial_boxes = int((np.asarray(obs["map"]) == 2).sum())
        initial_power = int(obs["players"][agent_slot, 3]) + int(obs["players"][agent_slot, 4])
        prev_alive = [bool(p[2]) for p in obs["players"]]
        death_groups = []
        survival_steps = [0, 0, 0, 0]
        done = False
        step = 0
        last_action = 0
        while not done and step < max_steps:
            actions = []
            for idx, agent in enumerate(agents):
                started = time.perf_counter()
                try:
                    action = int(agent.act(obs))
                except Exception:
                    action = 0
                elapsed = (time.perf_counter() - started) * 1000.0
                if idx == agent_slot:
                    inference_ms.append(elapsed)
                    last_action = action
                actions.append(action if 0 <= action <= 5 else 0)
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1
            alive = [bool(p[2]) for p in obs["players"]]
            if prev_alive[agent_slot] and not alive[agent_slot] and last_action == 5:
                self_kills += 1
            ranks_from_deaths(prev_alive, alive, death_groups, step, survival_steps)
            prev_alive = alive
        survivors = [idx for idx, alive in enumerate(prev_alive) if alive]
        for idx in survivors:
            survival_steps[idx] = step
        match_ranks = final_ranks(death_groups, survivors)
        rank = match_ranks[agent_slot]
        ranks.append(rank)
        survival.append(survival_steps[agent_slot])
        if survivors == [agent_slot]:
            wins += 1
        elif agent_slot in survivors:
            draws += 1
        else:
            losses += 1
            deaths += 1
        boxes_destroyed.append(max(0, initial_boxes - int((np.asarray(obs["map"]) == 2).sum())))
        final_power = int(obs["players"][agent_slot, 3]) + int(obs["players"][agent_slot, 4])
        item_delta.append(max(0, final_power - initial_power))
    timings = sorted(inference_ms) or [0.0]
    p50 = timings[int(0.50 * (len(timings) - 1))]
    p95 = timings[int(0.95 * (len(timings) - 1))]
    return {
        "opponent": opponent,
        "episodes": episodes,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / episodes,
        "draw_rate": draws / episodes,
        "loss_rate": losses / episodes,
        "avg_rank": statistics.mean(ranks),
        "death_rate": deaths / episodes,
        "self_kill_rate": self_kills / episodes,
        "avg_survival_steps": statistics.mean(survival),
        "avg_boxes_destroyed": statistics.mean(boxes_destroyed),
        "avg_item_collection": statistics.mean(item_delta),
        "inference_ms_p50": p50,
        "inference_ms_p95": p95,
        "inference_ms_max": max(timings),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_path", default="agent/hybrid_agent_rl_pure")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--opponents", nargs="+", default=["random", "simple", "box_farmer", "smarter", "tactical", "genius", "online_robust"])
    parser.add_argument("--head_to_head_online_robust", action="store_true")
    parser.add_argument("--output", default="logs/rl_pure_eval.json")
    args = parser.parse_args()
    opponents = ["online_robust"] if args.head_to_head_online_robust else args.opponents
    results = [run_suite(args.agent_path, opponent, args.episodes, args.max_steps, args.seed) for opponent in opponents]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
