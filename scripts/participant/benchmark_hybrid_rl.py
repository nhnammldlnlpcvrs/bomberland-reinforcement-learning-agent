"""Head-to-head benchmark for hybrid_agent_rl promotion checks."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from engine.game import BomberEnv
from scripts.participant.run_local_match import make_agents


def _final_ranks(n_players, death_groups, survivors):
    ranks = [0] * n_players
    groups = list(death_groups)
    if survivors:
        groups.append(list(survivors))
    for rank, group in enumerate(reversed(groups)):
        for player_id in group:
            ranks[player_id] = rank
    return ranks


def benchmark(args):
    roster = [args.candidate, args.reference] + list(args.opponents)
    totals = {
        "wins": 0,
        "draws": 0,
        "deaths": 0,
        "crashes": 0,
        "timeouts": 0,
        "rank_sum": 0,
        "survival_sum": 0,
        "steps_sum": 0,
    }
    score_sum = 0.0

    for ep in range(args.episodes):
        seed = args.seed + ep
        env = BomberEnv(seed=seed, max_steps=args.max_steps)
        agents, names = make_agents(roster, seed=seed)
        obs = env.reset(seed=seed)
        done = False
        step = 0
        prev_alive = [bool(p[2]) for p in obs["players"]]
        death_groups = []
        survival_steps = [0] * 4

        while not done and step < args.max_steps:
            obs_for_agent = dict(obs)
            obs_for_agent["step"] = step
            actions = []
            for i, agent in enumerate(agents):
                started = time.perf_counter()
                try:
                    action = int(agent.act(obs_for_agent))
                except Exception:
                    action = 0
                    if i == 0:
                        totals["crashes"] += 1
                elapsed = time.perf_counter() - started
                if i == 0 and elapsed > args.timeout_ms / 1000.0:
                    totals["timeouts"] += 1
                if action < 0 or action > 5:
                    action = 0
                    if i == 0:
                        totals["crashes"] += 1
                actions.append(action)

            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1

            alive_now = [bool(p[2]) for p in obs["players"]]
            deaths = []
            for i in range(4):
                if prev_alive[i] and not alive_now[i]:
                    deaths.append(i)
                    survival_steps[i] = step
            if deaths:
                death_groups.append(deaths)
            prev_alive = alive_now

        survivors = [i for i in range(4) if bool(obs["players"][i][2])]
        for i in survivors:
            survival_steps[i] = step
        ranks = _final_ranks(4, death_groups, survivors)
        best_rank = min(ranks)
        best_players = [i for i, rank in enumerate(ranks) if rank == best_rank]

        if ranks[0] == best_rank and len(best_players) == 1:
            totals["wins"] += 1
        elif ranks[0] == best_rank:
            totals["draws"] += 1
        if not bool(obs["players"][0][2]):
            totals["deaths"] += 1

        totals["rank_sum"] += ranks[0]
        totals["survival_sum"] += survival_steps[0]
        totals["steps_sum"] += step
        score_sum += 3 - ranks[0]

    n = max(1, args.episodes)
    print("=== Hybrid RL Benchmark ===")
    print(f"Candidate: {roster[0]}")
    print(f"Reference: {roster[1]}")
    print(f"Opponents: {roster[2:]}")
    print(f"Episodes: {args.episodes}")
    print(f"Win Rate: {100 * totals['wins'] / n:.1f}%")
    print(f"Draw Rate: {100 * totals['draws'] / n:.1f}%")
    print(f"Death Rate: {100 * totals['deaths'] / n:.1f}%")
    print(f"Average Rank: {totals['rank_sum'] / n:.2f}")
    print(f"Average Score: {score_sum / n:.2f}")
    print(f"Average Survival Step: {totals['survival_sum'] / n:.1f}")
    print(f"Average Match Steps: {totals['steps_sum'] / n:.1f}")
    print(f"Crash Count: {totals['crashes']}")
    print(f"Timeout Count: {totals['timeouts']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="agent/hybrid_agent_rl")
    parser.add_argument("--reference", default="agent/hybrid_agent_online_robust")
    parser.add_argument("--opponents", nargs=2, default=["TacticalRuleAgent", "SmarterRuleAgent"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout_ms", type=float, default=100.0)
    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
