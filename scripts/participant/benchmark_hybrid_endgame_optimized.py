from __future__ import annotations

import argparse
import json
import os
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

from agent.rl_agent_pure.action_mask import legal_action_mask
from competition.evaluation.runtime_guard import load_agent_instance
from engine.game import BomberEnv
from scripts.participant.benchmark_rl_strong_pool import _clear_submission_import_state


BOARD_SIZE = 13
TILE_WALL = 1
TILE_BOX = 2
A_BOMB = 5
BLAST_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

AGENTS = {
    "hybrid_endgame_optimized": "agent/hybrid_agent_endgame_optimized",
    "production": "submission",
}
BLOCK_BASE_SEEDS = {1: 61000, 2: 71000, 3: 81000, 4: 91000, 5: 101000}


@dataclass
class CompactStats:
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    rank_sum: float = 0.0
    survival_sum: float = 0.0
    self_bomb_deaths: int = 0
    enemy_bomb_deaths: int = 0
    timeout_count: int = 0
    error_count: int = 0
    invalid_action_count: int = 0
    act_calls: int = 0
    latency_sum_ms: float = 0.0
    latency_values_ms: list[float] = field(default_factory=list)
    endgame_counters: dict[str, int] = field(default_factory=dict)
    endgame_reject_reason_counts: dict[str, int] = field(default_factory=dict)


def _set_env(endgame_enabled: bool) -> None:
    os.environ["HYBRID_MODEL_ENABLE"] = "true"
    os.environ["HYBRID_MODEL_MAX_LATENCY_MS"] = "5"
    os.environ["HYBRID_ENDGAME_ENABLE"] = "true" if endgame_enabled else "false"
    os.environ["HYBRID_ENDGAME_SEARCH_ENABLE"] = "true"
    os.environ["HYBRID_ENDGAME_SEARCH_BUDGET_MS"] = "30"
    os.environ["HYBRID_ENDGAME_MAX_DEPTH"] = "3"
    os.environ["HYBRID_ENDGAME_ONLY_SAFE_ACTIONS"] = "true"


def _load_agent(label: str, agent_id: int):
    _clear_submission_import_state()
    path = ROOT / AGENTS[label]
    if path.is_dir():
        path = path / "agent.py"
    agent = load_agent_instance(str(path), agent_id)
    _clear_submission_import_state()
    return agent


def _reset_agent_state(agent) -> None:
    if hasattr(agent, "position_history"):
        agent.position_history = []
    if hasattr(agent, "last_bomb_step"):
        agent.last_bomb_step = -10**9
    if hasattr(agent, "_future_cache"):
        agent._future_cache = {}
    if hasattr(agent, "_expansion_cache"):
        agent._expansion_cache = {}
    if hasattr(agent, "reject_reason_counts"):
        agent.reject_reason_counts = {}
    if hasattr(agent, "intervention_log"):
        agent.intervention_log = []
    if hasattr(agent, "counters"):
        loaded = 1 if getattr(getattr(agent, "model_ranker", None), "loaded", False) else 0
        agent.counters = {
            "model_loaded": loaded,
            "model_inference_errors": 0,
            "model_tiebreaker_used": 0,
            "model_action_accepted": 0,
            "model_action_rejected_by_safety": 0,
            "fallback_to_rule": 0,
        }
    if hasattr(agent, "endgame_counters"):
        agent.endgame_counters = {
            "endgame_active_steps": 0,
            "endgame_search_used": 0,
            "endgame_action_changed": 0,
            "endgame_bomb_accepted": 0,
            "endgame_bomb_rejected_safety": 0,
            "endgame_fallback_to_production": 0,
            "endgame_timeout_fallback": 0,
        }
    if hasattr(agent, "endgame_reject_reason_counts"):
        agent.endgame_reject_reason_counts = {}


def _agent_from_cache(cache: dict[tuple[str, int], object], label: str, agent_id: int):
    key = (label, int(agent_id))
    if key not in cache:
        cache[key] = _load_agent(label, agent_id)
    agent = cache[key]
    _reset_agent_state(agent)
    return agent


def disabled_parity_check(seeds: list[int], max_steps: int) -> dict:
    _set_env(False)
    os.environ["HYBRID_MODEL_MAX_LATENCY_MS"] = "1000"
    total = 0
    mismatches = []
    for seed in seeds:
        production_agents = [_load_agent("production", idx) for idx in range(4)]
        candidate_agents = [_load_agent("hybrid_endgame_optimized", idx) for idx in range(4)]
        env = BomberEnv(max_steps=max_steps, seed=seed)
        obs = {**env.reset(seed=seed), "step": 0}
        done = False
        step = 0
        while not done and step < max_steps:
            actions = []
            for idx in range(4):
                if not bool(obs["players"][idx, 2]):
                    actions.append(0)
                    continue
                prod_action = int(production_agents[idx].act(obs))
                cand_action = int(candidate_agents[idx].act(obs))
                total += 1
                if prod_action != cand_action:
                    mismatches.append({
                        "seed": seed, "step": step, "agent_id": idx,
                        "production": prod_action, "candidate": cand_action,
                    })
                actions.append(prod_action)
            obs, terminated, truncated = env.step(actions)
            step += 1
            obs = {**obs, "step": step}
            done = terminated or truncated
    return {"calls": total, "mismatches": len(mismatches), "mismatch_rate": len(mismatches) / max(1, total), "examples": mismatches[:20]}


def _in_bounds(row: int, col: int) -> bool:
    return 0 < row < BOARD_SIZE - 1 and 0 < col < BOARD_SIZE - 1


def _blast_cells(board, players, row: int, col: int, owner: int) -> frozenset[tuple[int, int]]:
    radius = 1 + max(0, int(players[owner][4])) if 0 <= owner < len(players) else 1
    cells = {(row, col)}
    for dr, dc in BLAST_DIRS:
        for distance in range(1, radius + 1):
            nr, nc = row + dr * distance, col + dc * distance
            if not _in_bounds(nr, nc):
                break
            if int(board[nr, nc]) == TILE_WALL:
                break
            cells.add((nr, nc))
            if int(board[nr, nc]) == TILE_BOX:
                break
    return frozenset(cells)


def _valid_bomb_placement(prev_obs: dict, player_id: int, action: int) -> bool:
    if action != A_BOMB:
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


def _roster(seed: int) -> list[str]:
    roster = ["hybrid_endgame_optimized", "hybrid_endgame_optimized", "production", "production"]
    random.Random(seed).shuffle(roster)
    return roster


def _final_ranks(death_groups: list[list[int]], survivors: list[int]) -> tuple[list[int], set[int]]:
    groups = list(death_groups)
    if survivors:
        groups.append(list(survivors))
    ranks = [0] * 4
    rank_groups = list(reversed(groups))
    for rank, group in enumerate(rank_groups):
        for player_id in group:
            ranks[player_id] = rank
    return ranks, set(rank_groups[0] if rank_groups else [])


def _merge_counter(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _run_episode(seed: int, max_steps: int, timeout_s: float, stats: dict[str, CompactStats], cache: dict[tuple[str, int], object]) -> None:
    roster = _roster(seed)
    agents = [_agent_from_cache(cache, label, idx) for idx, label in enumerate(roster)]
    env = BomberEnv(max_steps=max_steps, seed=seed)
    obs = {**env.reset(seed=seed), "step": 0}
    prev_alive = [bool(player[2]) for player in obs["players"]]
    survival_steps = [0] * 4
    death_groups: list[list[int]] = []
    bomb_events = []
    death_causes = [{"self_bomb": 0, "enemy_bomb": 0} for _ in range(4)]
    step = 0
    done = False
    while not done and step < max_steps:
        prev_obs = obs
        actions = []
        for idx, agent in enumerate(agents):
            label = roster[idx]
            if not bool(prev_obs["players"][idx, 2]):
                actions.append(0)
                continue
            started = time.perf_counter()
            error = False
            try:
                action = int(agent.act(prev_obs))
            except Exception:
                action = 0
                error = True
            latency_ms = (time.perf_counter() - started) * 1000.0
            timeout = latency_ms > timeout_s * 1000.0
            invalid = not (0 <= action <= 5)
            if not invalid:
                try:
                    invalid = not bool(legal_action_mask(prev_obs, idx)[action])
                except Exception:
                    invalid = False
            if invalid:
                action = 0
            row = stats[label]
            row.act_calls += 1
            row.latency_sum_ms += latency_ms
            row.latency_values_ms.append(latency_ms)
            row.timeout_count += int(timeout)
            row.error_count += int(error)
            row.invalid_action_count += int(invalid)
            actions.append(action)
            if _valid_bomb_placement(prev_obs, idx, action):
                r, c = int(prev_obs["players"][idx, 0]), int(prev_obs["players"][idx, 1])
                bomb_events.append({"owner_idx": idx, "placed_step": step, "cells": _blast_cells(prev_obs["map"], prev_obs["players"], r, c, idx)})
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
                death_pos = (int(prev_obs["players"][idx, 0]), int(prev_obs["players"][idx, 1]))
                own_hit = False
                enemy_hit = False
                for event in bomb_events:
                    age = step - int(event["placed_step"])
                    if 0 <= age <= 7 and death_pos in event["cells"]:
                        if int(event["owner_idx"]) == idx:
                            own_hit = True
                        else:
                            enemy_hit = True
                death_causes[idx]["self_bomb"] = int(own_hit)
                death_causes[idx]["enemy_bomb"] = int((not own_hit) and enemy_hit)
        if deaths:
            death_groups.append(deaths)
        prev_alive = alive_now
    survivors = [idx for idx, alive in enumerate(prev_alive) if alive]
    for idx in survivors:
        survival_steps[idx] = step
    ranks, best_group = _final_ranks(death_groups, survivors)
    for idx, label in enumerate(roster):
        row = stats[label]
        row.matches += 1
        rank = int(ranks[idx])
        row.rank_sum += rank
        row.survival_sum += int(survival_steps[idx])
        if rank == 0 and len(best_group) == 1:
            row.wins += 1
        elif rank == 0:
            row.draws += 1
        else:
            row.losses += 1
        row.self_bomb_deaths += int(death_causes[idx]["self_bomb"])
        row.enemy_bomb_deaths += int(death_causes[idx]["enemy_bomb"])
    for idx, agent in enumerate(agents):
        if roster[idx] != "hybrid_endgame_optimized":
            continue
        _merge_counter(stats["hybrid_endgame_optimized"].endgame_counters, getattr(agent, "endgame_counters", {}))
        _merge_counter(stats["hybrid_endgame_optimized"].endgame_reject_reason_counts, getattr(agent, "endgame_reject_reason_counts", {}))


def _summary_row(row: CompactStats) -> dict:
    matches = max(1, row.matches)
    calls = max(1, row.act_calls)
    values = sorted(row.latency_values_ms)
    p95_idx = min(len(values) - 1, int(round(0.95 * (len(values) - 1)))) if values else 0
    return {
        "matches": row.matches, "wins": row.wins, "draws": row.draws, "losses": row.losses,
        "win_rate": row.wins / matches, "draw_rate": row.draws / matches, "loss_rate": row.losses / matches,
        "average_rank": row.rank_sum / matches, "average_survival_step": row.survival_sum / matches,
        "self_bomb_deaths": row.self_bomb_deaths, "enemy_bomb_deaths": row.enemy_bomb_deaths,
        "timeout_count": row.timeout_count, "error_count": row.error_count, "invalid_action_count": row.invalid_action_count,
        "act_calls": row.act_calls, "average_act_latency_ms": row.latency_sum_ms / calls,
        "p95_act_latency_ms": values[p95_idx] if values else 0.0, "max_act_latency_ms": values[-1] if values else 0.0,
        "endgame_counters": dict(row.endgame_counters), "endgame_reject_reason_counts": dict(row.endgame_reject_reason_counts),
    }


def run_benchmark(episodes: int, base_seed: int, max_steps: int, timeout_s: float, progress_every: int) -> dict:
    _set_env(True)
    stats = {label: CompactStats() for label in AGENTS}
    cache: dict[tuple[str, int], object] = {}
    for idx in range(episodes):
        _run_episode(base_seed + idx * 9973, max_steps, timeout_s, stats, cache)
        if (idx + 1) % progress_every == 0:
            print(f"completed {idx + 1}/{episodes}")
    return {label: _summary_row(row) for label, row in stats.items()}


def _verdict(summary: dict[str, dict], parity: dict) -> tuple[str, list[str]]:
    cand = summary["hybrid_endgame_optimized"]
    prod = summary["production"]
    reasons = []
    if parity["mismatches"] != 0:
        reasons.append("disabled parity mismatch")
    if cand["average_rank"] >= prod["average_rank"]:
        reasons.append("candidate average rank is not lower than production")
    if cand["loss_rate"] > prod["loss_rate"]:
        reasons.append("candidate loss rate is higher than production")
    if cand["self_bomb_deaths"] > prod["self_bomb_deaths"]:
        reasons.append("candidate self-bomb deaths are higher than production")
    if cand["timeout_count"] or cand["error_count"] or cand["invalid_action_count"]:
        reasons.append("candidate has timeout/error/invalid actions")
    counters = cand.get("endgame_counters", {})
    if int(counters.get("endgame_active_steps", 0)) <= 0 or int(counters.get("endgame_search_used", 0)) <= 0:
        reasons.append("endgame logic did not activate")
    if int(counters.get("endgame_action_changed", 0)) <= 0:
        reasons.append("endgame logic did not change any action")
    if reasons:
        return "REJECT_PROMOTION", reasons
    return "PROMOTE_CANDIDATE", ["candidate passes endgame optimization gate"]


def _write_report(path: Path, payload: dict) -> None:
    summary = payload["summary"]
    lines = [
        "# Hybrid Endgame Optimization Report", "",
        f"Generated: {payload['generated']}", "",
        "## Scope", "",
        "- Candidate: `agent/hybrid_agent_endgame_optimized/`",
        "- Production baseline: current `submission/agent.py`",
        "- `submission/agent.py` was not modified.",
        "- `agent/hybrid_agent_online_robust/` was not modified.",
        "- No submission zip was generated.", "",
        "## Disabled Parity", "",
        f"- Calls compared: `{payload['parity']['calls']}`",
        f"- Mismatches: `{payload['parity']['mismatches']}`",
        f"- Mismatch rate: `{payload['parity']['mismatch_rate']:.6f}`", "",
        "## Results", "",
        "| Agent | Matches | Win | Draw | Loss | Win Rate | Loss Rate | Avg Rank | Avg Survival | Self-Bomb Deaths | Enemy-Bomb Deaths | Timeouts | Errors | Invalid | Avg ms | P95 ms | Max ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in sorted(summary.items(), key=lambda item: item[1]["average_rank"]):
        lines.append(
            f"| `{label}` | {row['matches']} | {row['wins']} | {row['draws']} | {row['losses']} | "
            f"{row['win_rate']:.3f} | {row['loss_rate']:.3f} | {row['average_rank']:.3f} | "
            f"{row['average_survival_step']:.1f} | {row['self_bomb_deaths']} | {row['enemy_bomb_deaths']} | "
            f"{row['timeout_count']} | {row['error_count']} | {row['invalid_action_count']} | "
            f"{row['average_act_latency_ms']:.3f} | {row['p95_act_latency_ms']:.3f} | {row['max_act_latency_ms']:.3f} |"
        )
    counters = summary["hybrid_endgame_optimized"]["endgame_counters"]
    rejects = summary["hybrid_endgame_optimized"]["endgame_reject_reason_counts"]
    lines.extend(["", "## Endgame Counters", ""])
    for key, value in sorted(counters.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "Reject reason counts:", ""])
    for key, value in sorted(rejects.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Verdict", "", f"**{payload['verdict']}**", ""])
    for reason in payload["reasons"]:
        lines.append(f"- {reason}")
    if payload["verdict"] == "PROMOTE_CANDIDATE":
        lines.append("\nStatus: `PROMOTE_CANDIDATE`. Do not update submission without a separate explicit promotion task.")
    else:
        lines.append("\nStatus: `REJECT_PROMOTION`. Production remains current `submission/agent.py`.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark hybrid endgame optimized candidate.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--base_seed", type=int, default=60610)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--timeout_s", type=float, default=0.1)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--parity_seeds", nargs="+", type=int, default=[9100, 9200, 9300])
    parser.add_argument("--json_output", default="logs/hybrid_endgame_optimization_benchmark.json")
    parser.add_argument("--report", default="docs/HYBRID_ENDGAME_OPTIMIZATION_REPORT.md")
    args = parser.parse_args()
    parity = disabled_parity_check(args.parity_seeds, args.max_steps)
    summary = run_benchmark(args.episodes, args.base_seed, args.max_steps, args.timeout_s, args.progress_every)
    verdict, reasons = _verdict(summary, parity)
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "parity": parity,
        "summary": summary,
        "verdict": verdict,
        "reasons": reasons,
    }
    out = ROOT / args.json_output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(ROOT / args.report, payload)
    print(f"Verdict: {verdict}")
    for reason in reasons:
        print(f"- {reason}")
    print(f"Wrote {out}")
    print(f"Wrote {ROOT / args.report}")
    return 0 if verdict == "PROMOTE_CANDIDATE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
