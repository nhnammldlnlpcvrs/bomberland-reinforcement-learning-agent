from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import GeniusRuleAgent, SmarterRuleAgent, TacticalRuleAgent
from agent.rl_agent_pure.action_mask import legal_action_mask
from competition.evaluation.runtime_guard import load_agent_instance
from engine.game import BomberEnv


AGENTS = {
    "rl_strong": "agent/rl_strong",
    "online_robust": "agent/hybrid_agent_online_robust",
    "tactical_rule": "TacticalRuleAgent",
    "genius_rule": "GeniusRuleAgent",
    "smarter_rule": "SmarterRuleAgent",
}

BASELINES = {
    "TacticalRuleAgent": TacticalRuleAgent,
    "GeniusRuleAgent": GeniusRuleAgent,
    "SmarterRuleAgent": SmarterRuleAgent,
}
RL_STRONG_CACHE = {}

LOCAL_MODULE_NAMES = {
    "agent",
    "action_mask",
    "constants",
    "encoder",
    "features",
    "frame_buffer",
    "memory",
    "model",
    "policy",
    "rl_policy",
    "rule_policy",
    "safety",
    "utils",
}


@dataclass
class AgentStats:
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    rank_sum: float = 0.0
    survival_sum: float = 0.0
    self_bomb_deaths: int = 0
    timeout_count: int = 0
    error_count: int = 0
    invalid_action_count: int = 0
    act_calls: int = 0
    latency_sum_ms: float = 0.0
    latency_max_ms: float = 0.0
    bomb_count: int = 0
    action_counts: dict[str, int] = field(default_factory=lambda: {str(i): 0 for i in range(6)})


def _clear_submission_import_state() -> None:
    for name in LOCAL_MODULE_NAMES:
        sys.modules.pop(name, None)

    agent_root = ROOT / "agent"
    cleaned = []
    for entry in sys.path:
        try:
            path = Path(entry).resolve()
        except Exception:
            cleaned.append(entry)
            continue
        if path.parent == agent_root:
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned


def _make_agent(spec: str, agent_id: int):
    if spec in BASELINES:
        return BASELINES[spec](agent_id)

    path = ROOT / spec
    if path.is_dir():
        path = path / "agent.py"
    if not path.exists():
        raise FileNotFoundError(f"Agent path not found: {path}")

    cache_key = (spec, int(agent_id))
    if spec == "agent/rl_strong" and cache_key in RL_STRONG_CACHE:
        return RL_STRONG_CACHE[cache_key]

    _clear_submission_import_state()
    agent = load_agent_instance(str(path), agent_id)
    _clear_submission_import_state()
    if spec == "agent/rl_strong":
        RL_STRONG_CACHE[cache_key] = agent
    return agent


def _make_roster(agent_names: list[str], rng: random.Random) -> list[str]:
    combos = list(itertools.combinations(agent_names, 4))
    combo = list(combos[rng.randrange(len(combos))])
    rng.shuffle(combo)
    return combo


def _final_ranks(death_groups: list[list[int]], survivors: list[int]) -> tuple[list[int], list[int]]:
    groups = list(death_groups)
    if survivors:
        groups.append(list(survivors))
    ranks = [0] * 4
    rank_groups = list(reversed(groups))
    for rank, group in enumerate(rank_groups):
        for player_id in group:
            ranks[player_id] = rank
    best_group = rank_groups[0] if rank_groups else []
    return ranks, best_group


def _valid_bomb_placement(prev_obs: dict, player_id: int, action: int) -> bool:
    if action != 5:
        return False
    players = np.asarray(prev_obs["players"])
    bombs = np.asarray(prev_obs["bombs"])
    if not bool(players[player_id, 2]) or int(players[player_id, 3]) <= 0:
        return False
    row, col = int(players[player_id, 0]), int(players[player_id, 1])
    if bombs.size == 0:
        return True
    bombs = bombs.reshape(-1, bombs.shape[-1])
    return not any(int(bomb[0]) == row and int(bomb[1]) == col for bomb in bombs)


def run_episode(agent_names: list[str], specs: dict[str, str], seed: int, max_steps: int, timeout_s: float):
    rng = random.Random(seed)
    roster_names = _make_roster(agent_names, rng)
    agents = [_make_agent(specs[name], idx) for idx, name in enumerate(roster_names)]
    env = BomberEnv(max_steps=max_steps, seed=seed)
    obs = {**env.reset(seed=seed), "step": 0}

    prev_alive = [bool(player[2]) for player in obs["players"]]
    death_groups: list[list[int]] = []
    survival_steps = [0] * 4
    per_episode = {name: {"self_bomb_death": 0, "bomb_events": []} for name in roster_names}
    runtime_rows = []

    done = False
    step = 0
    while not done and step < max_steps:
        prev_obs = obs
        actions = []
        for idx, agent in enumerate(agents):
            name = roster_names[idx]
            if not bool(prev_obs["players"][idx, 2]):
                actions.append(0)
                continue

            started = time.perf_counter()
            error = False
            try:
                raw_action = agent.act(prev_obs)
                action = int(raw_action)
            except Exception:
                action = 0
                error = True
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            timeout = elapsed_ms > timeout_s * 1000.0
            invalid = not (0 <= action <= 5)
            if not invalid:
                try:
                    invalid = not bool(legal_action_mask(prev_obs, idx)[action])
                except Exception:
                    invalid = False

            if invalid:
                action = 0
            actions.append(action)
            runtime_rows.append({
                "name": name,
                "action": int(action),
                "latency_ms": elapsed_ms,
                "timeout": timeout,
                "error": error,
                "invalid": invalid,
            })

            if _valid_bomb_placement(prev_obs, idx, action):
                per_episode[name]["bomb_events"].append({
                    "placed_step": step,
                    "death_recorded": False,
                })

        obs, terminated, truncated = env.step(actions)
        step += 1
        obs = {**obs, "step": step}
        done = terminated or truncated

        alive_now = [bool(player[2]) for player in obs["players"]]
        deaths = []
        for idx in range(4):
            if prev_alive[idx] and not alive_now[idx]:
                deaths.append(idx)
                survival_steps[idx] = step
                name = roster_names[idx]
                for event in per_episode[name]["bomb_events"]:
                    age = step - int(event["placed_step"])
                    if not event["death_recorded"] and 0 <= age <= 7:
                        per_episode[name]["self_bomb_death"] = 1
                        event["death_recorded"] = True
        if deaths:
            death_groups.append(deaths)
        prev_alive = alive_now

    survivors = [idx for idx, alive in enumerate(prev_alive) if alive]
    for idx in survivors:
        survival_steps[idx] = step
    ranks, best_group = _final_ranks(death_groups, survivors)

    return {
        "seed": seed,
        "roster": roster_names,
        "steps": step,
        "ranks": ranks,
        "best_group": best_group,
        "survival_steps": survival_steps,
        "runtime": runtime_rows,
        "per_episode": per_episode,
    }


def _aggregate(episodes: list[dict], agent_names: list[str]) -> dict[str, AgentStats]:
    stats = {name: AgentStats() for name in agent_names}
    for episode in episodes:
        roster = episode["roster"]
        best_group = set(episode["best_group"])
        for idx, name in enumerate(roster):
            row = stats[name]
            row.matches += 1
            rank = int(episode["ranks"][idx])
            row.rank_sum += rank
            row.survival_sum += int(episode["survival_steps"][idx])
            if rank == 0 and len(best_group) == 1:
                row.wins += 1
            elif rank == 0:
                row.draws += 1
            else:
                row.losses += 1
            row.self_bomb_deaths += int(episode["per_episode"][name]["self_bomb_death"])
            row.bomb_count += len(episode["per_episode"][name]["bomb_events"])

        for runtime in episode["runtime"]:
            row = stats[runtime["name"]]
            row.act_calls += 1
            row.timeout_count += int(runtime["timeout"])
            row.error_count += int(runtime["error"])
            row.invalid_action_count += int(runtime["invalid"])
            row.latency_sum_ms += float(runtime["latency_ms"])
            row.latency_max_ms = max(row.latency_max_ms, float(runtime["latency_ms"]))
            row.action_counts[str(runtime["action"])] += 1
    return stats


def _summary_dict(stats: dict[str, AgentStats]) -> dict[str, dict]:
    rows = {}
    for name, row in stats.items():
        matches = max(1, row.matches)
        calls = max(1, row.act_calls)
        rows[name] = {
            "matches": row.matches,
            "wins": row.wins,
            "draws": row.draws,
            "losses": row.losses,
            "win_rate": row.wins / matches,
            "draw_rate": row.draws / matches,
            "loss_rate": row.losses / matches,
            "average_rank": row.rank_sum / matches,
            "average_survival_step": row.survival_sum / matches,
            "self_bomb_deaths": row.self_bomb_deaths,
            "timeout_count": row.timeout_count,
            "error_count": row.error_count,
            "invalid_action_count": row.invalid_action_count,
            "act_calls": row.act_calls,
            "average_act_latency_ms": row.latency_sum_ms / calls,
            "max_act_latency_ms": row.latency_max_ms,
            "bomb_count": row.bomb_count,
            "action_counts": row.action_counts,
        }
    return rows


def _verdict(summary: dict[str, dict]) -> tuple[str, list[str]]:
    rl = summary["rl_strong"]
    production = summary["online_robust"]
    reject_reasons = []
    if rl["timeout_count"] or rl["error_count"] or rl["invalid_action_count"]:
        reject_reasons.append("rl_strong has timeout/error/invalid actions")
    if rl["self_bomb_deaths"] > production["self_bomb_deaths"]:
        reject_reasons.append("rl_strong has more self-bomb deaths than online_robust")
    if rl["win_rate"] <= production["win_rate"]:
        reject_reasons.append("rl_strong win rate is not better than online_robust")
    if rl["loss_rate"] > production["loss_rate"]:
        reject_reasons.append("rl_strong loss rate is worse than online_robust")
    if rl["average_rank"] >= production["average_rank"]:
        reject_reasons.append("rl_strong average rank is worse than online_robust")
    if rl["average_survival_step"] < production["average_survival_step"]:
        reject_reasons.append("rl_strong average survival is worse than online_robust")

    better_than_production = (
        rl["win_rate"] > production["win_rate"]
        and rl["loss_rate"] <= production["loss_rate"]
        and rl["average_rank"] < production["average_rank"]
        and rl["average_survival_step"] >= production["average_survival_step"]
    )
    best_avg_rank = min(row["average_rank"] for row in summary.values())
    best_win_rate = max(row["win_rate"] for row in summary.values())

    if better_than_production and rl["average_rank"] <= best_avg_rank + 1e-9 and rl["win_rate"] >= best_win_rate - 1e-9:
        return "PROMOTION_CANDIDATE", [
            "rl_strong beats online_robust on win rate, loss rate, average rank, and survival",
            "rl_strong is best in the tested pool by average rank and win rate",
        ]
    if reject_reasons:
        return "REJECT_PROMOTION", reject_reasons
    return "REJECT_PROMOTION", [
        "rl_strong does not clearly beat online_robust and the strong baseline pool",
    ]


def _markdown_report(args, episodes: list[dict], summary: dict[str, dict], verdict: str, reasons: list[str]) -> str:
    total = len(episodes)
    seed_values = sorted({int(row["seed"]) for row in episodes})
    lines = [
        "# RL Strong Benchmark Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "Compared agents:",
        "",
        "- `agent/rl_strong`",
        "- `agent/hybrid_agent_online_robust`",
        "- `tactical_rule`",
        "- `genius_rule`",
        "- `smarter_rule`",
        "",
        "No production agent files were modified by this benchmark.",
        "",
        "## Configuration",
        "",
        f"- Total episodes: `{total}`",
        f"- Episode target: `{args.episodes}`",
        f"- Max steps: `{args.max_steps}`",
        f"- Inference timeout threshold: `{args.timeout_s:.3f}s`",
        f"- Base seeds: `{', '.join(str(seed) for seed in args.seeds)}`",
        f"- First/last episode seed: `{seed_values[0]}` / `{seed_values[-1]}`",
        "- Match sampling: each episode samples 4 of the 5 agents and shuffles slots.",
        "",
        "## Results",
        "",
        "| Agent | Matches | Win | Draw | Loss | Win Rate | Draw Rate | Loss Rate | Avg Rank | Avg Survival | Self-Bomb Deaths | Timeouts | Errors | Invalid | Avg Act ms | Max Act ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in sorted(summary.items(), key=lambda item: item[1]["average_rank"]):
        lines.append(
            f"| `{name}` | {row['matches']} | {row['wins']} | {row['draws']} | {row['losses']} | "
            f"{row['win_rate']:.3f} | {row['draw_rate']:.3f} | {row['loss_rate']:.3f} | "
            f"{row['average_rank']:.3f} | {row['average_survival_step']:.1f} | "
            f"{row['self_bomb_deaths']} | {row['timeout_count']} | {row['error_count']} | "
            f"{row['invalid_action_count']} | {row['average_act_latency_ms']:.3f} | {row['max_act_latency_ms']:.3f} |"
        )

    lines.extend([
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
    ])
    for reason in reasons:
        lines.append(f"- {reason}")

    if verdict != "PROMOTION_CANDIDATE":
        lines.extend([
            "",
            "`rl_strong` is rejected for promotion based on this benchmark. Do not export or submit it over `online_robust` without new evidence.",
        ])
    else:
        lines.extend([
            "",
            "`rl_strong` is a promotion candidate based on this benchmark. Next step would be exporting with `python -m ml.export_rl_strong_submission` and running a final submission precheck.",
        ])

    return "\n".join(lines) + "\n"


def _print_summary(summary: dict[str, dict], verdict: str, reasons: list[str]) -> None:
    headers = ("agent", "matches", "win", "draw", "loss", "avg_rank", "avg_surv", "self_bomb", "timeouts", "errors", "avg_ms")
    rows = [headers]
    for name, row in sorted(summary.items(), key=lambda item: item[1]["average_rank"]):
        rows.append((
            name,
            str(row["matches"]),
            f"{row['win_rate']:.3f}",
            f"{row['draw_rate']:.3f}",
            f"{row['loss_rate']:.3f}",
            f"{row['average_rank']:.3f}",
            f"{row['average_survival_step']:.1f}",
            str(row["self_bomb_deaths"]),
            str(row["timeout_count"]),
            str(row["error_count"]),
            f"{row['average_act_latency_ms']:.3f}",
        ))
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(headers))]
    for idx, row in enumerate(rows):
        print("  ".join(value.rjust(widths[col]) for col, value in enumerate(row)))
        if idx == 0:
            print("  ".join("-" * width for width in widths))
    print(f"\nVerdict: {verdict}")
    for reason in reasons:
        print(f"- {reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-seed pool benchmark for rl_strong vs production and strong baselines.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 3026, 4026])
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--timeout_s", type=float, default=0.1)
    parser.add_argument("--json_output", default="logs/rl_strong_pool_benchmark.json")
    parser.add_argument("--report", default="docs/RL_STRONG_BENCHMARK_REPORT.md")
    args = parser.parse_args()

    agent_names = list(AGENTS)
    episodes = []
    for episode_idx in range(args.episodes):
        base_seed = args.seeds[episode_idx % len(args.seeds)]
        seed = base_seed + episode_idx * 9973
        episodes.append(run_episode(agent_names, AGENTS, seed, args.max_steps, args.timeout_s))
        if (episode_idx + 1) % 25 == 0:
            print(f"completed {episode_idx + 1}/{args.episodes} episodes")

    stats = _aggregate(episodes, agent_names)
    summary = _summary_dict(stats)
    verdict, reasons = _verdict(summary)

    payload = {
        "config": {
            "episodes": args.episodes,
            "seeds": args.seeds,
            "max_steps": args.max_steps,
            "timeout_s": args.timeout_s,
            "agents": AGENTS,
        },
        "summary": summary,
        "episodes": episodes,
        "latency_distribution_ms": {
            name: {
                "mean": summary[name]["average_act_latency_ms"],
                "max": summary[name]["max_act_latency_ms"],
            }
            for name in agent_names
        },
        "verdict": verdict,
        "reasons": reasons,
    }

    json_output = ROOT / args.json_output
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown_report(args, episodes, summary, verdict, reasons), encoding="utf-8")

    _print_summary(summary, verdict, reasons)
    print(f"\nWrote {json_output}")
    print(f"Wrote {report_path}")
    return 0 if verdict == "PROMOTION_CANDIDATE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
