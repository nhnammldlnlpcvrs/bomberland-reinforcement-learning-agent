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
    _set_model_env,
    disabled_parity_check,
)
from scripts.participant.benchmark_rl_strong_pool import _clear_submission_import_state


BOARD_SIZE = 13
TILE_WALL = 1
TILE_BOX = 2
A_BOMB = 5
BLAST_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

AGENTS = {
    "hybrid_model_optimized": "agent/hybrid_agent_model_optimized",
    "online_robust": "agent/hybrid_agent_online_robust",
}
BLOCK_BASE_SEEDS = {
    1: 11000,
    2: 21000,
    3: 31000,
    4: 41000,
    5: 51000,
}


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
    model_counters: dict[str, int] = field(default_factory=dict)
    reject_reason_counts: dict[str, int] = field(default_factory=dict)


def _ensure_enabled_env(checkpoint: str, max_latency_ms: float) -> None:
    _set_model_env(True, checkpoint, max_latency_ms)
    if os.environ.get("HYBRID_MODEL_ENABLE", "").lower() != "true":
        raise RuntimeError("HYBRID_MODEL_ENABLE must be true for enabled validation blocks")


def _load_agent(label: str, agent_id: int):
    _clear_submission_import_state()
    agent = load_agent_instance(str(ROOT / AGENTS[label] / "agent.py"), agent_id)
    _clear_submission_import_state()
    return agent


def _agent_from_cache(agent_cache: dict[tuple[str, int], object], label: str, agent_id: int):
    key = (label, int(agent_id))
    if key not in agent_cache:
        agent_cache[key] = _load_agent(label, agent_id)
    agent = agent_cache[key]
    _reset_agent_state(agent)
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


def _in_bounds(row: int, col: int) -> bool:
    return 0 < row < BOARD_SIZE - 1 and 0 < col < BOARD_SIZE - 1


def _blast_cells(board, players, row: int, col: int, owner: int) -> frozenset[tuple[int, int]]:
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
    roster = ["hybrid_model_optimized", "hybrid_model_optimized", "online_robust", "online_robust"]
    rng = random.Random(seed)
    rng.shuffle(roster)
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


def _update_latency(row: CompactStats, latency_ms: float) -> None:
    row.act_calls += 1
    row.latency_sum_ms += latency_ms
    row.latency_values_ms.append(latency_ms)


def _update_model_counters(stats: dict[str, CompactStats], agents, roster: list[str]) -> None:
    row = stats["hybrid_model_optimized"]
    for idx, agent in enumerate(agents):
        if roster[idx] != "hybrid_model_optimized":
            continue
        _merge_counter(row.model_counters, getattr(agent, "counters", {}))
        _merge_counter(row.reject_reason_counts, getattr(agent, "reject_reason_counts", {}))


def _run_episode(seed: int, max_steps: int, timeout_s: float,
                 stats: dict[str, CompactStats],
                 agent_cache: dict[tuple[str, int], object]) -> None:
    roster = _roster(seed)
    agents = [_agent_from_cache(agent_cache, label, idx) for idx, label in enumerate(roster)]
    env = BomberEnv(max_steps=max_steps, seed=seed)
    obs = {**env.reset(seed=seed), "step": 0}
    prev_alive = [bool(player[2]) for player in obs["players"]]
    survival_steps = [0] * 4
    death_groups: list[list[int]] = []
    bomb_events: list[dict] = []
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
            _update_latency(row, latency_ms)
            row.timeout_count += int(timeout)
            row.error_count += int(error)
            row.invalid_action_count += int(invalid)

            actions.append(action)
            if _valid_bomb_placement(prev_obs, idx, action):
                row_pos = int(prev_obs["players"][idx, 0])
                col_pos = int(prev_obs["players"][idx, 1])
                bomb_events.append({
                    "owner_idx": idx,
                    "placed_step": step,
                    "cells": _blast_cells(prev_obs["map"], prev_obs["players"], row_pos, col_pos, idx),
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
                    if death_pos not in event["cells"]:
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

    _update_model_counters(stats, agents, roster)


def _summary_row(row: CompactStats) -> dict:
    matches = max(1, row.matches)
    calls = max(1, row.act_calls)
    values = sorted(row.latency_values_ms)
    p95_idx = min(len(values) - 1, int(round(0.95 * (len(values) - 1)))) if values else 0
    return {
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
        "average_act_latency_ms": row.latency_sum_ms / calls,
        "p95_act_latency_ms": values[p95_idx] if values else 0.0,
        "max_act_latency_ms": values[-1] if values else 0.0,
        "model_counters": dict(row.model_counters),
        "reject_reason_counts": dict(row.reject_reason_counts),
    }


def _block_valid(summary: dict[str, dict]) -> tuple[bool, list[str]]:
    row = summary["hybrid_model_optimized"]
    counters = row.get("model_counters", {})
    reject_reasons = row.get("reject_reason_counts", {})
    reasons = []
    if os.environ.get("HYBRID_MODEL_ENABLE", "").lower() != "true":
        reasons.append("HYBRID_MODEL_ENABLE is not true")
    if int(counters.get("model_loaded", 0)) <= 0:
        reasons.append("model_loaded is zero")
    non_disabled_rejects = sum(
        int(value) for key, value in reject_reasons.items() if key != "disabled"
    )
    if int(counters.get("model_tiebreaker_used", 0)) <= 0 and non_disabled_rejects <= 0:
        reasons.append("model never ran and reject reasons are only disabled/empty")
    if reject_reasons and set(reject_reasons) == {"disabled"}:
        reasons.append("block counters are all disabled")
    return not reasons, reasons


def run_block(args) -> dict:
    if args.block not in BLOCK_BASE_SEEDS:
        raise ValueError("--block must be between 1 and 5")
    _ensure_enabled_env(args.checkpoint, args.max_latency_ms)
    base_seed = BLOCK_BASE_SEEDS[args.block]
    stats = {label: CompactStats() for label in AGENTS}
    agent_cache: dict[tuple[str, int], object] = {}
    started = time.perf_counter()
    for episode_idx in range(args.episodes):
        seed = base_seed + episode_idx * 9973
        _run_episode(seed, args.max_steps, args.timeout_s, stats, agent_cache)
        if (episode_idx + 1) % args.progress_every == 0:
            print(f"block {args.block}: completed {episode_idx + 1}/{args.episodes}")
    summary = {label: _summary_row(row) for label, row in stats.items()}
    valid, validity_reasons = _block_valid(summary)
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "block": args.block,
        "base_seed": base_seed,
        "seed_start": base_seed,
        "seed_end": base_seed + (args.episodes - 1) * 9973,
        "episode_count": args.episodes,
        "max_steps": args.max_steps,
        "timeout_s": args.timeout_s,
        "elapsed_s": time.perf_counter() - started,
        "hybrid_model_enable": os.environ.get("HYBRID_MODEL_ENABLE", ""),
        "valid": valid,
        "validity_reasons": validity_reasons,
        "summary": summary,
    }
    path = ROOT / f"logs/hybrid_model_final_validation_block{args.block}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    print(f"Block valid: {valid}")
    for reason in validity_reasons:
        print(f"- {reason}")
    return payload


def _sum_summary_rows(rows: list[dict]) -> dict:
    total = CompactStats()
    for row in rows:
        total.matches += int(row["matches"])
        total.wins += int(row["wins"])
        total.draws += int(row["draws"])
        total.losses += int(row["losses"])
        total.rank_sum += float(row["average_rank"]) * int(row["matches"])
        total.survival_sum += float(row["average_survival_step"]) * int(row["matches"])
        total.self_bomb_deaths += int(row["self_bomb_deaths"])
        total.enemy_bomb_deaths += int(row["enemy_bomb_deaths"])
        total.timeout_count += int(row["timeout_count"])
        total.error_count += int(row["error_count"])
        total.invalid_action_count += int(row["invalid_action_count"])
        calls = int(row["act_calls"])
        total.act_calls += calls
        total.latency_sum_ms += float(row["average_act_latency_ms"]) * calls
        _merge_counter(total.model_counters, row.get("model_counters", {}))
        _merge_counter(total.reject_reason_counts, row.get("reject_reason_counts", {}))
    merged = _summary_row(total)
    merged["p95_act_latency_ms"] = max(
        (float(row["p95_act_latency_ms"]) for row in rows),
        default=0.0,
    )
    merged["p95_note"] = "max block p95, compact upper bound"
    merged["max_act_latency_ms"] = max((float(row["max_act_latency_ms"]) for row in rows), default=0.0)
    return merged


def _promotion_verdict(blocks: list[dict], parity: dict, summary: dict[str, dict]) -> tuple[str, list[str]]:
    candidate = summary["hybrid_model_optimized"]
    production = summary["online_robust"]
    reasons = []
    if len(blocks) != 5:
        reasons.append("not all 5 blocks are present")
    for block in blocks:
        if not block.get("valid", False):
            reasons.append(f"block {block.get('block')} invalid: {', '.join(block.get('validity_reasons', []))}")
    if parity["mismatches"] != 0:
        reasons.append("disabled parity is not 0 mismatch")
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
    return "PROMOTE_CANDIDATE", ["candidate passes all final validation gates"]


def _markdown(payload: dict) -> str:
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
        "- No submission zip was generated.",
        "",
        "## Disabled Parity",
        "",
        f"- Calls compared: `{payload['parity']['calls']}`",
        f"- Mismatches: `{payload['parity']['mismatches']}`",
        f"- Mismatch rate: `{payload['parity']['mismatch_rate']:.6f}`",
        "",
        "## Block Validity",
        "",
        "| Block | Episodes | Seed Start | Seed End | Valid | Notes |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for block in payload["blocks"]:
        notes = "; ".join(block.get("validity_reasons", [])) or "ok"
        lines.append(
            f"| {block['block']} | {block['episode_count']} | {block['seed_start']} | "
            f"{block['seed_end']} | {block['valid']} | {notes} |"
        )
    lines.extend([
        "",
        "## Aggregate Results",
        "",
        "| Agent | Matches | Win | Draw | Loss | Win Rate | Loss Rate | Avg Rank | Avg Survival | Self-Bomb Deaths | Enemy-Bomb Deaths | Timeouts | Errors | Invalid | Avg ms | P95 ms | Max ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for label, row in sorted(summary.items(), key=lambda item: item[1]["average_rank"]):
        lines.append(
            f"| `{label}` | {row['matches']} | {row['wins']} | {row['draws']} | {row['losses']} | "
            f"{row['win_rate']:.3f} | {row['loss_rate']:.3f} | {row['average_rank']:.3f} | "
            f"{row['average_survival_step']:.1f} | {row['self_bomb_deaths']} | {row['enemy_bomb_deaths']} | "
            f"{row['timeout_count']} | {row['error_count']} | {row['invalid_action_count']} | "
            f"{row['average_act_latency_ms']:.3f} | {row['p95_act_latency_ms']:.3f} | {row['max_act_latency_ms']:.3f} |"
        )
    lines.extend([
        "",
        "Latency note: aggregate p95 is the maximum p95 across the five compact block summaries.",
    ])
    candidate = summary["hybrid_model_optimized"]
    lines.extend(["", "## Model Intervention Counts", ""])
    for key, value in sorted(candidate["model_counters"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "Reject reason counts:", ""])
    for key, value in sorted(candidate["reject_reason_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Verdict", "", f"**{payload['verdict']}**", ""])
    for reason in payload["reasons"]:
        lines.append(f"- {reason}")
    if payload["verdict"] == "PROMOTE_CANDIDATE":
        lines.extend(["", "Status: `PROMOTE_CANDIDATE`. Production should remain `online_robust` until an explicit promotion task updates submission files."])
    else:
        lines.extend(["", "Status: `REJECT_PROMOTION`. Production remains `online_robust`."])
    return "\n".join(lines) + "\n"


def merge(args) -> dict:
    block_paths = [ROOT / f"logs/hybrid_model_final_validation_block{idx}.json" for idx in range(1, 6)]
    blocks = []
    for path in block_paths:
        if not path.exists():
            continue
        blocks.append(json.loads(path.read_text(encoding="utf-8")))

    parity = disabled_parity_check(args.parity_seeds, args.max_steps)
    _ensure_enabled_env(args.checkpoint, args.max_latency_ms)

    summary = {
        label: _sum_summary_rows([block["summary"][label] for block in blocks if label in block.get("summary", {})])
        for label in AGENTS
    }
    final_verdict, reasons = _promotion_verdict(blocks, parity, summary)
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "parity": parity,
        "blocks": blocks,
        "summary": summary,
        "verdict": final_verdict,
        "reasons": reasons,
    }
    output_path = ROOT / "logs/hybrid_model_final_validation.json"
    report_path = ROOT / "docs/HYBRID_MODEL_FINAL_VALIDATION.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(_markdown(payload), encoding="utf-8")
    print(f"Verdict: {final_verdict}")
    for reason in reasons:
        print(f"- {reason}")
    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Compact final validation for hybrid_agent_model_optimized.")
    parser.add_argument("--block", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--timeout_s", type=float, default=0.1)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--max_latency_ms", type=float, default=5.0)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--parity_seeds", nargs="+", type=int, default=[8100, 8200, 8300])
    args = parser.parse_args()

    if args.merge:
        payload = merge(args)
        return 0 if payload["verdict"] == "PROMOTE_CANDIDATE" else 1
    if args.block is None:
        parser.error("provide --block 1..5 or --merge")
    payload = run_block(args)
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
