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
from scripts.participant.benchmark_hybrid_model_optimized import (
    disabled_parity_check,
    _set_model_env,
)
from scripts.participant.benchmark_rl_strong_pool import _clear_submission_import_state


BOARD_SIZE = 13
TILE_WALL = 1
TILE_BOX = 2
A_BOMB = 5
BLAST_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

AGENT_PATHS = {
    "hybrid_model_optimized": "agent/hybrid_agent_model_optimized",
    "online_robust": "agent/hybrid_agent_online_robust",
}


@dataclass
class FinalStats:
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
    latency_values_ms: list[float] = field(default_factory=list)
    model_counters: dict[str, int] = field(default_factory=dict)
    reject_reason_counts: dict[str, int] = field(default_factory=dict)


def _load_agent(label: str, agent_id: int):
    _clear_submission_import_state()
    path = ROOT / AGENT_PATHS[label] / "agent.py"
    agent = load_agent_instance(str(path), agent_id)
    _clear_submission_import_state()
    return agent


def _in_bounds(row: int, col: int) -> bool:
    return 0 < row < BOARD_SIZE - 1 and 0 < col < BOARD_SIZE - 1


def _blast_cells(board, players, row: int, col: int, owner: int) -> set[tuple[int, int]]:
    radius = 1
    if 0 <= owner < len(players):
        radius = 1 + max(0, int(players[owner][4]))
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
    return cells


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


def _roster(seed: int) -> list[str]:
    roster = ["hybrid_model_optimized", "hybrid_model_optimized", "online_robust", "online_robust"]
    rng = random.Random(seed)
    rng.shuffle(roster)
    return roster


def run_episode(seed: int, max_steps: int, timeout_s: float) -> dict:
    roster = _roster(seed)
    agents = [_load_agent(label, idx) for idx, label in enumerate(roster)]
    env = BomberEnv(max_steps=max_steps, seed=seed)
    obs = {**env.reset(seed=seed), "step": 0}
    prev_alive = [bool(player[2]) for player in obs["players"]]
    survival_steps = [0] * 4
    death_groups: list[list[int]] = []
    runtime_rows = []
    bomb_events = []
    death_causes = [{"self_bomb": 0, "enemy_bomb": 0} for _ in range(4)]

    done = False
    step = 0
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
                "label": label,
                "latency_ms": elapsed_ms,
                "timeout": timeout,
                "error": error,
                "invalid": invalid,
            })

            if _valid_bomb_placement(prev_obs, idx, action):
                row, col = int(prev_obs["players"][idx, 0]), int(prev_obs["players"][idx, 1])
                bomb_events.append({
                    "owner_idx": idx,
                    "owner_label": label,
                    "placed_step": step,
                    "cells": sorted(_blast_cells(prev_obs["map"], prev_obs["players"], row, col, idx)),
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
                death_pos = (int(prev_obs["players"][idx, 0]), int(prev_obs["players"][idx, 1]))
                own_hit = False
                enemy_hit = False
                for event in bomb_events:
                    age = step - int(event["placed_step"])
                    if not (0 <= age <= 7):
                        continue
                    if death_pos not in {tuple(cell) for cell in event["cells"]}:
                        continue
                    if int(event["owner_idx"]) == idx:
                        own_hit = True
                    else:
                        enemy_hit = True
                if own_hit:
                    death_causes[idx]["self_bomb"] = 1
                elif enemy_hit:
                    death_causes[idx]["enemy_bomb"] = 1
        if deaths:
            death_groups.append(deaths)
        prev_alive = alive_now

    survivors = [idx for idx, alive in enumerate(prev_alive) if alive]
    for idx in survivors:
        survival_steps[idx] = step
    ranks, best_group = _final_ranks(death_groups, survivors)

    agent_model_rows = []
    for idx, agent in enumerate(agents):
        if roster[idx] != "hybrid_model_optimized":
            continue
        agent_model_rows.append({
            "counters": dict(getattr(agent, "counters", {})),
            "reject_reason_counts": dict(getattr(agent, "reject_reason_counts", {})),
        })

    return {
        "seed": seed,
        "roster": roster,
        "steps": step,
        "ranks": ranks,
        "best_group": sorted(best_group),
        "survival_steps": survival_steps,
        "death_causes": death_causes,
        "runtime": runtime_rows,
        "model_rows": agent_model_rows,
    }


def _merge_counter(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def aggregate(episodes: list[dict]) -> dict[str, FinalStats]:
    stats = {name: FinalStats() for name in AGENT_PATHS}
    for episode in episodes:
        best_group = set(episode["best_group"])
        for idx, label in enumerate(episode["roster"]):
            row = stats[label]
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
            row.self_bomb_deaths += int(episode["death_causes"][idx]["self_bomb"])
            row.enemy_bomb_deaths += int(episode["death_causes"][idx]["enemy_bomb"])

        for runtime in episode["runtime"]:
            row = stats[runtime["label"]]
            row.act_calls += 1
            row.timeout_count += int(runtime["timeout"])
            row.error_count += int(runtime["error"])
            row.invalid_action_count += int(runtime["invalid"])
            row.latency_values_ms.append(float(runtime["latency_ms"]))

        for model_row in episode["model_rows"]:
            _merge_counter(stats["hybrid_model_optimized"].model_counters, model_row["counters"])
            _merge_counter(stats["hybrid_model_optimized"].reject_reason_counts, model_row["reject_reason_counts"])
    return stats


def summary_dict(stats: dict[str, FinalStats]) -> dict[str, dict]:
    output = {}
    for label, row in stats.items():
        matches = max(1, row.matches)
        values = sorted(row.latency_values_ms)
        calls = max(1, row.act_calls)
        p95_idx = min(len(values) - 1, int(round(0.95 * (len(values) - 1)))) if values else 0
        output[label] = {
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
            "enemy_bomb_deaths": row.enemy_bomb_deaths,
            "timeout_count": row.timeout_count,
            "error_count": row.error_count,
            "invalid_action_count": row.invalid_action_count,
            "act_calls": row.act_calls,
            "average_act_latency_ms": statistics.fmean(values) if values else 0.0,
            "p95_act_latency_ms": values[p95_idx] if values else 0.0,
            "max_act_latency_ms": max(values) if values else 0.0,
            "model_counters": row.model_counters,
            "reject_reason_counts": row.reject_reason_counts,
        }
    return output


def verdict(summary: dict[str, dict], parity: dict) -> tuple[str, list[str]]:
    candidate = summary["hybrid_model_optimized"]
    production = summary["online_robust"]
    reasons = []
    if parity["mismatches"] != 0:
        reasons.append("disabled parity has mismatches")
    if candidate["average_rank"] >= production["average_rank"]:
        reasons.append("candidate average rank is not lower than online_robust")
    if candidate["loss_rate"] > production["loss_rate"]:
        reasons.append("candidate loss rate is higher than online_robust")
    if candidate["self_bomb_deaths"] > production["self_bomb_deaths"]:
        reasons.append("candidate self-bomb deaths are higher than online_robust")
    if candidate["timeout_count"] > production["timeout_count"]:
        reasons.append("candidate has timeout regression")
    if candidate["error_count"] > production["error_count"]:
        reasons.append("candidate has error regression")
    if candidate["invalid_action_count"] > production["invalid_action_count"]:
        reasons.append("candidate has invalid-action regression")
    if reasons:
        return "REJECT_PROMOTION", reasons
    return "PROMOTE_HYBRID_MODEL_OPTIMIZED", [
        "candidate passes final validation gates against online_robust",
    ]


def write_outputs(args, parity: dict, block_payloads: list[dict], output_path: Path, report_path: Path) -> tuple[str, list[str]]:
    episodes = [episode for block in block_payloads for episode in block["episodes"]]
    stats = aggregate(episodes)
    summary = summary_dict(stats)
    final_verdict, reasons = verdict(summary, parity)
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "parity": parity,
        "blocks": block_payloads,
        "summary": summary,
        "verdict": final_verdict,
        "reasons": reasons,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown_report(payload), encoding="utf-8")
    return final_verdict, reasons


def markdown_report(payload: dict) -> str:
    args = payload["config"]
    summary = payload["summary"]
    lines = [
        "# Hybrid Model Final Validation",
        "",
        f"Generated: {payload['generated']}",
        "",
        "## Scope",
        "",
        "- Candidate: `agent/hybrid_agent_model_optimized/`",
        "- Production baseline: `agent/hybrid_agent_online_robust/`",
        "- `submission/agent.py` was not modified.",
        "- `agent/hybrid_agent_online_robust/` was not modified.",
        "",
        "## Configuration",
        "",
        f"- Seed blocks: `{args['blocks']}`",
        f"- Episodes per block: `{args['episodes_per_block']}`",
        f"- Total requested episodes: `{args['blocks'] * args['episodes_per_block']}`",
        f"- Max steps: `{args['max_steps']}`",
        f"- Timeout threshold: `{args['timeout_s']:.3f}s`",
        f"- Base seeds: `{', '.join(str(seed) for seed in args['base_seeds'])}`",
        "",
        "## Disabled Parity",
        "",
        f"- Calls compared: `{payload['parity']['calls']}`",
        f"- Mismatches: `{payload['parity']['mismatches']}`",
        f"- Mismatch rate: `{payload['parity']['mismatch_rate']:.6f}`",
        "",
        "## Aggregate Results",
        "",
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
    candidate = summary["hybrid_model_optimized"]
    lines.extend([
        "",
        "## Model Intervention Counts",
        "",
    ])
    for key, value in sorted(candidate["model_counters"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "Reject reason counts:", ""])
    for key, value in sorted(candidate["reject_reason_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Verdict",
        "",
        f"**{payload['verdict']}**",
        "",
    ])
    for reason in payload["reasons"]:
        lines.append(f"- {reason}")
    if payload["verdict"].startswith("PROMOTE"):
        lines.extend(["", "Recommendation: promote `hybrid_agent_model_optimized` after final packaging/precheck."])
    else:
        lines.extend(["", "Recommendation: keep production as `online_robust`."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Final independent validation for hybrid_agent_model_optimized.")
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--episodes_per_block", type=int, default=300)
    parser.add_argument("--base_seeds", nargs="+", type=int, default=[11000, 21000, 31000, 41000, 51000])
    parser.add_argument("--parity_seeds", nargs="+", type=int, default=[8100, 8200, 8300])
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--timeout_s", type=float, default=0.1)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--max_latency_ms", type=float, default=5.0)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--json_output", default="logs/hybrid_model_final_validation.json")
    parser.add_argument("--report", default="docs/HYBRID_MODEL_FINAL_VALIDATION.md")
    args = parser.parse_args()

    _set_model_env(True, args.checkpoint, args.max_latency_ms)
    parity = disabled_parity_check(args.parity_seeds, args.max_steps)
    _set_model_env(True, args.checkpoint, args.max_latency_ms)
    output_path = ROOT / args.json_output
    report_path = ROOT / args.report
    block_payloads = []

    for block_idx in range(args.blocks):
        base_seed = args.base_seeds[block_idx % len(args.base_seeds)]
        episodes = []
        started = time.perf_counter()
        for episode_idx in range(args.episodes_per_block):
            seed = base_seed + episode_idx * 9973
            episodes.append(run_episode(seed, args.max_steps, args.timeout_s))
            if (episode_idx + 1) % args.progress_every == 0:
                print(f"block {block_idx + 1}/{args.blocks}: completed {episode_idx + 1}/{args.episodes_per_block}")
        elapsed_s = time.perf_counter() - started
        block_payloads.append({
            "block": block_idx + 1,
            "base_seed": base_seed,
            "episodes": episodes,
            "elapsed_s": elapsed_s,
        })
        verdict_value, reasons = write_outputs(args, parity, block_payloads, output_path, report_path)
        print(f"wrote checkpoint after block {block_idx + 1}: {verdict_value}")
        for reason in reasons:
            print(f"- {reason}")

    final_verdict, final_reasons = write_outputs(args, parity, block_payloads, output_path, report_path)
    print(f"Final verdict: {final_verdict}")
    for reason in final_reasons:
        print(f"- {reason}")
    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    return 0 if final_verdict.startswith("PROMOTE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
