#!/usr/bin/env python
"""Collect compact late-game samples from current production self-play only."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.action_mask import legal_action_mask
from engine.game import BomberEnv

A_STOP = 0
A_BOMB = 5
DIRS = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
PASSABLE = {0, 3, 4}
TILE_WALL = 1
TILE_BOX = 2
TILE_RADIUS = 3
TILE_CAPACITY = 4
BLAST_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
ACTION_NAMES = {
    0: "STOP",
    1: "LEFT",
    2: "RIGHT",
    3: "UP",
    4: "DOWN",
    5: "PLACE_BOMB",
}
WORKER_AGENTS: list[Any] | None = None


def load_production_class(path: Path):
    spec = importlib.util.spec_from_file_location("late_game_production_agent", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load production agent: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not bool(getattr(module, "HYBRID_MODEL_ENABLE", False)):
        raise RuntimeError(
            "Current production model is disabled. Unset HYBRID_MODEL_ENABLE or set it to true."
        )
    return module.Agent, module


def reset_agent(agent: Any) -> None:
    if hasattr(agent, "position_history"):
        agent.position_history = []
    if hasattr(agent, "last_bomb_step"):
        agent.last_bomb_step = -(10**9)
    if hasattr(agent, "_future_cache"):
        agent._future_cache = {}
    if hasattr(agent, "_expansion_cache"):
        agent._expansion_cache = {}
    if hasattr(agent, "reject_reason_counts"):
        agent.reject_reason_counts = {}
    if hasattr(agent, "intervention_log"):
        agent.intervention_log = []


def in_bounds(board: np.ndarray, row: int, col: int) -> bool:
    return 0 <= row < board.shape[0] and 0 <= col < board.shape[1]


def blast_cells(
    board: np.ndarray, row: int, col: int, radius: int
) -> set[tuple[int, int]]:
    cells = {(row, col)}
    for dr, dc in BLAST_DIRS:
        for distance in range(1, radius + 1):
            nr, nc = row + dr * distance, col + dc * distance
            if not in_bounds(board, nr, nc) or int(board[nr, nc]) == TILE_WALL:
                break
            cells.add((nr, nc))
            if int(board[nr, nc]) == TILE_BOX:
                break
    return cells


def bomb_records(env: BomberEnv) -> list[dict[str, int]]:
    return [
        {
            "row": int(bomb.x),
            "col": int(bomb.y),
            "timer": int(bomb.timer),
            "owner": int(bomb.owner_id),
            "radius": int(bomb.radius),
        }
        for bomb in env.bombs
    ]


def effective_bomb_timers(
    board: np.ndarray, bombs: list[dict[str, int]]
) -> list[int]:
    timers = [int(bomb["timer"]) for bomb in bombs]
    blasts = [
        blast_cells(board, bomb["row"], bomb["col"], bomb["radius"])
        for bomb in bombs
    ]
    changed = True
    while changed:
        changed = False
        for source_idx, cells in enumerate(blasts):
            for target_idx, target in enumerate(bombs):
                if source_idx == target_idx:
                    continue
                target_pos = (target["row"], target["col"])
                if target_pos in cells and timers[target_idx] > timers[source_idx]:
                    timers[target_idx] = timers[source_idx]
                    changed = True
    return timers


def danger_time(
    board: np.ndarray, bombs: list[dict[str, int]]
) -> dict[tuple[int, int], int]:
    danger: dict[tuple[int, int], int] = {}
    timers = effective_bomb_timers(board, bombs)
    for bomb, timer in zip(bombs, timers):
        for cell in blast_cells(
            board, bomb["row"], bomb["col"], bomb["radius"]
        ):
            danger[cell] = min(danger.get(cell, 10**9), int(timer))
    return danger


def passable(
    board: np.ndarray,
    row: int,
    col: int,
    bomb_positions: set[tuple[int, int]],
) -> bool:
    return (
        in_bounds(board, row, col)
        and int(board[row, col]) in PASSABLE
        and (row, col) not in bomb_positions
    )


def reachable_safe_cells(
    board: np.ndarray,
    start: tuple[int, int],
    bombs: list[dict[str, int]],
    horizon: int = 8,
) -> set[tuple[int, int]]:
    danger = danger_time(board, bombs)
    bomb_positions = {(bomb["row"], bomb["col"]) for bomb in bombs}
    queue = deque([(start, 0)])
    seen_states = {(start, 0)}
    safe_cells = {start}
    while queue:
        (row, col), elapsed = queue.popleft()
        if elapsed >= horizon:
            continue
        for dr, dc in DIRS.values():
            nr, nc = row + dr, col + dc
            arrival = elapsed + 1
            if (nr, nc) != start and not passable(
                board, nr, nc, bomb_positions
            ):
                continue
            if danger.get((nr, nc), 10**9) <= arrival:
                continue
            state = ((nr, nc), arrival)
            if state in seen_states:
                continue
            seen_states.add(state)
            safe_cells.add((nr, nc))
            queue.append(state)
    return safe_cells


def can_escape_hypothetical_bomb(
    board: np.ndarray,
    start: tuple[int, int],
    bombs: list[dict[str, int]],
    hypothetical: dict[str, int],
) -> bool:
    combined = list(bombs) + [hypothetical]
    effective_timer = effective_bomb_timers(board, combined)[-1]
    hypothetical_blast = blast_cells(
        board,
        hypothetical["row"],
        hypothetical["col"],
        hypothetical["radius"],
    )
    danger = danger_time(board, combined)
    bomb_positions = {(bomb["row"], bomb["col"]) for bomb in combined}
    queue = deque([(start, 0)])
    seen = {(start, 0)}
    while queue:
        (row, col), elapsed = queue.popleft()
        if (
            elapsed > 0
            and (row, col) not in hypothetical_blast
            and danger.get((row, col), 10**9) > elapsed
        ):
            return True
        if elapsed >= effective_timer:
            continue
        for action in range(5):
            dr, dc = DIRS[action]
            nr, nc = row + dr, col + dc
            arrival = elapsed + 1
            if (nr, nc) != start and not passable(
                board, nr, nc, bomb_positions
            ):
                continue
            if danger.get((nr, nc), 10**9) <= arrival:
                continue
            state = ((nr, nc), arrival)
            if state in seen:
                continue
            seen.add(state)
            queue.append(state)
    return False


def nearest_distance(
    origin: tuple[int, int], targets: Iterable[tuple[int, int]]
) -> int | None:
    distances = [
        abs(origin[0] - row) + abs(origin[1] - col) for row, col in targets
    ]
    return min(distances) if distances else None


def action_position(
    board: np.ndarray,
    origin: tuple[int, int],
    action: int,
    bombs: list[dict[str, int]],
) -> tuple[int, int]:
    if action not in DIRS:
        return origin
    dr, dc = DIRS[action]
    target = (origin[0] + dr, origin[1] + dc)
    bomb_positions = {(bomb["row"], bomb["col"]) for bomb in bombs}
    return (
        target
        if passable(board, target[0], target[1], bomb_positions)
        else origin
    )


def map_snapshot(board: np.ndarray) -> list[str]:
    return ["".join(str(int(cell)) for cell in row) for row in board]


def state_metrics(
    env: BomberEnv, obs: dict, our_id: int, action: int
) -> dict[str, Any]:
    board = np.asarray(obs["map"])
    players = np.asarray(obs["players"])
    bombs = bomb_records(env)
    ours = players[our_id]
    our_pos = (int(ours[0]), int(ours[1]))
    alive_enemies = [
        (idx, row)
        for idx, row in enumerate(players)
        if idx != our_id and bool(row[2])
    ]
    enemy_positions = [
        (int(row[0]), int(row[1])) for _, row in alive_enemies
    ]
    safe_areas = {}
    for idx, player in enumerate(players):
        if bool(player[2]):
            pos = (int(player[0]), int(player[1]))
            safe_areas[str(idx)] = len(
                reachable_safe_cells(board, pos, bombs)
            )

    hypothetical = {
        "row": our_pos[0],
        "col": our_pos[1],
        "timer": 7,
        "owner": our_id,
        "radius": 1 + max(0, int(ours[4])),
    }
    bomb_positions = {(bomb["row"], bomb["col"]) for bomb in bombs}
    can_place = (
        bool(ours[2])
        and int(ours[3]) > 0
        and our_pos not in bomb_positions
    )
    hypothetical_blast = blast_cells(
        board, our_pos[0], our_pos[1], hypothetical["radius"]
    )
    enemies_in_radius = [
        idx
        for idx, row in alive_enemies
        if (int(row[0]), int(row[1])) in hypothetical_blast
    ]
    can_escape = can_place and can_escape_hypothetical_bomb(
        board, our_pos, bombs, hypothetical
    )
    enemy_escape_areas = {
        str(idx): len(
            reachable_safe_cells(
                board,
                (int(row[0]), int(row[1])),
                list(bombs) + [hypothetical],
                horizon=7,
            )
        )
        for idx, row in alive_enemies
        if (int(row[0]), int(row[1])) in hypothetical_blast
    }
    trapped_enemies = [
        int(idx) for idx, area in enemy_escape_areas.items() if area <= 2
    ]
    safe_bomb_reasons = []
    if enemies_in_radius:
        safe_bomb_reasons.append("enemy_in_blast")
    if trapped_enemies:
        safe_bomb_reasons.append("enemy_low_escape_area")
    possible_safe_bomb = bool(can_escape and safe_bomb_reasons)

    danger = danger_time(board, bombs)
    immediate_danger = danger.get(our_pos, 10**9) <= 2
    min_enemy_distance = nearest_distance(our_pos, enemy_positions)
    open_cells = int(np.isin(board, list(PASSABLE)).sum())
    box_count = int((board == TILE_BOX).sum())
    potentially_walkable_cells = max(1, open_cells + box_count)
    target_pos = action_position(board, our_pos, action, bombs)
    item_positions = [
        (int(row), int(col))
        for row, col in np.argwhere(
            np.logical_or(board == TILE_RADIUS, board == TILE_CAPACITY)
        )
    ]
    before_item = nearest_distance(our_pos, item_positions)
    after_item = nearest_distance(target_pos, item_positions)
    after_enemy = nearest_distance(target_pos, enemy_positions)
    item_farming = bool(
        action in DIRS
        and before_item is not None
        and after_item is not None
        and after_item < before_item
        and (
            min_enemy_distance is None
            or after_enemy is None
            or after_enemy >= min_enemy_distance
        )
    )
    missed_pressure = bool(
        not immediate_danger
        and safe_areas.get(str(our_id), 0) >= 8
        and min_enemy_distance is not None
        and min_enemy_distance <= 5
        and action != A_BOMB
        and (after_enemy is None or after_enemy >= min_enemy_distance)
    )
    return {
        "our_position": list(our_pos),
        "enemy_positions": [
            {
                "player_id": idx,
                "position": [int(row[0]), int(row[1])],
            }
            for idx, row in alive_enemies
        ],
        "bombs": bombs,
        "action": int(action),
        "action_name": ACTION_NAMES.get(int(action), "INVALID"),
        "production_placed_bomb": bool(action == A_BOMB and can_place),
        "enemies_within_bomb_radius": enemies_in_radius,
        "reachable_safe_area": safe_areas,
        "possible_safe_bomb_opportunity": possible_safe_bomb,
        "possible_safe_bomb_reasons": safe_bomb_reasons,
        "possible_enemy_trapped": bool(trapped_enemies),
        "possible_trapped_enemy_ids": trapped_enemies,
        "enemy_escape_area_if_bombed": enemy_escape_areas,
        "hypothetical_bomb_escape_exists": bool(can_escape),
        "boxes_within_bomb_radius": sum(
            1
            for row, col in hypothetical_blast
            if int(board[row, col]) == TILE_BOX
        ),
        "minimum_enemy_distance": min_enemy_distance,
        "map_open_ratio": open_cells / potentially_walkable_cells,
        "map_box_count": box_count,
        "item_farming_marker": item_farming,
        "late_game_pressure_missed_marker": missed_pressure,
        "our_bombs_left": int(ours[3]),
        "our_bomb_radius": int(hypothetical["radius"]),
        "our_immediate_bomb_danger": immediate_danger,
    }


def explosion_owners_next_step(
    env: BomberEnv,
) -> list[tuple[int, set[tuple[int, int]]]]:
    bombs = bomb_records(env)
    if not bombs:
        return []
    timers = effective_bomb_timers(np.asarray(env.map.grid), bombs)
    return [
        (
            int(bomb["owner"]),
            blast_cells(
                np.asarray(env.map.grid),
                bomb["row"],
                bomb["col"],
                bomb["radius"],
            ),
        )
        for bomb, timer in zip(bombs, timers)
        if timer <= 1
    ]


def final_ranks(
    death_groups: list[list[int]], survivors: list[int]
) -> tuple[list[int], set[int]]:
    groups = list(death_groups)
    if survivors:
        groups.append(list(survivors))
    ranks = [0] * 4
    rank_groups = list(reversed(groups))
    for rank, group in enumerate(rank_groups):
        for player_id in group:
            ranks[player_id] = rank
    return ranks, set(rank_groups[0] if rank_groups else [])


def run_episode(
    seed: int,
    our_id: int,
    agents: list[Any],
    max_steps: int,
    late_start: int,
    sample_interval: int,
    timeout_ms: float,
) -> dict[str, Any]:
    for agent in agents:
        reset_agent(agent)
    env = BomberEnv(max_steps=max_steps, seed=seed)
    obs = {**env.reset(seed=seed), "step": 0}
    previous_alive = [bool(player[2]) for player in obs["players"]]
    death_groups: list[list[int]] = []
    action_trace: list[list[int]] = []
    alive_trace: list[list[int]] = []
    samples: list[dict[str, Any]] = []
    enemy_kills_by_us = 0
    runtime = Counter()
    latency_values: list[float] = []
    step = 0
    done = False
    selected_our_id: int | None = None

    while not done and step < max_steps:
        previous_obs = obs
        actions = []
        for idx, agent in enumerate(agents):
            if not bool(previous_obs["players"][idx, 2]):
                actions.append(A_STOP)
                continue
            started = time.perf_counter()
            error = False
            try:
                action = int(agent.act(previous_obs))
            except Exception:
                action = A_STOP
                error = True
            latency_ms = (time.perf_counter() - started) * 1000.0
            latency_values.append(latency_ms)
            invalid = not (0 <= action <= 5)
            if not invalid:
                try:
                    invalid = not bool(
                        legal_action_mask(previous_obs, idx)[action]
                    )
                except Exception:
                    invalid = False
            if invalid:
                action = A_STOP
            runtime["errors"] += int(error)
            runtime["invalid"] += int(invalid)
            runtime["timeouts"] += int(latency_ms > timeout_ms)
            actions.append(action)

        if step >= late_start and selected_our_id is None:
            alive_ids = [
                idx
                for idx, player in enumerate(previous_obs["players"])
                if bool(player[2])
            ]
            for offset in range(4):
                candidate = (our_id + offset) % 4
                if candidate in alive_ids:
                    selected_our_id = candidate
                    break
        active_our_id = (
            selected_our_id if selected_our_id is not None else our_id
        )
        if step >= late_start:
            alive_trace.append(
                [
                    step,
                    int(
                        sum(
                            bool(player[2])
                            for player in previous_obs["players"]
                        )
                    ),
                ]
            )
        if step >= late_start and bool(
            previous_obs["players"][active_our_id, 2]
        ):
            action_trace.append([step, int(actions[active_our_id])])
            metrics = state_metrics(
                env,
                previous_obs,
                active_our_id,
                actions[active_our_id],
            )
            metrics["step"] = step
            metrics["alive_players"] = alive_trace[-1][1]
            if (step - late_start) % sample_interval == 0:
                metrics["map"] = map_snapshot(
                    np.asarray(previous_obs["map"])
                )
            samples.append(metrics)

        pending_explosions = explosion_owners_next_step(env)
        obs, terminated, truncated = env.step(actions)
        step += 1
        obs = {**obs, "step": step}
        done = terminated or truncated
        alive_now = [bool(player[2]) for player in obs["players"]]
        deaths = [
            idx
            for idx in range(4)
            if previous_alive[idx] and not alive_now[idx]
        ]
        if deaths:
            death_groups.append(deaths)
            for victim in deaths:
                if victim == active_our_id:
                    continue
                victim_pos = (
                    int(previous_obs["players"][victim, 0]),
                    int(previous_obs["players"][victim, 1]),
                )
                if any(
                    owner == active_our_id and victim_pos in cells
                    for owner, cells in pending_explosions
                ):
                    enemy_kills_by_us += 1
        previous_alive = alive_now

    survivors = [
        idx for idx, alive in enumerate(previous_alive) if alive
    ]
    ranks, best_group = final_ranks(death_groups, survivors)
    active_our_id = (
        selected_our_id if selected_our_id is not None else our_id
    )
    return {
        "seed": seed,
        "preferred_player_id": our_id,
        "observed_player_id": active_our_id,
        "reached_late_game": step >= late_start,
        "final_rank": int(ranks[active_our_id]) + 1,
        "strict_win": bool(
            ranks[active_our_id] == 0 and len(best_group) == 1
        ),
        "draw_best_group": bool(
            ranks[active_our_id] == 0 and len(best_group) > 1
        ),
        "final_step": step,
        "alive_at_end": bool(previous_alive[active_our_id]),
        "alive_players_over_time": alive_trace,
        "action_trace": action_trace,
        "samples": samples,
        "enemy_kills_by_us": enemy_kills_by_us,
        "runtime": {
            "timeouts": int(runtime["timeouts"]),
            "errors": int(runtime["errors"]),
            "invalid": int(runtime["invalid"]),
            "average_latency_ms": (
                statistics.fmean(latency_values)
                if latency_values
                else 0.0
            ),
            "p95_latency_ms": (
                sorted(latency_values)[
                    min(
                        len(latency_values) - 1,
                        math.ceil(0.95 * len(latency_values)) - 1,
                    )
                ]
                if latency_values
                else 0.0
            ),
        },
    }


def worker_init(production_path: str) -> None:
    global WORKER_AGENTS
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    agent_class, _module = load_production_class(Path(production_path))
    WORKER_AGENTS = [agent_class(idx) for idx in range(4)]
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def worker_collect(task: tuple[int, int, int, int, int, float]) -> dict[str, Any]:
    if WORKER_AGENTS is None:
        raise RuntimeError("worker agent cache was not initialized")
    seed, our_id, max_steps, late_start, sample_interval, timeout_ms = task
    return run_episode(
        seed=seed,
        our_id=our_id,
        agents=WORKER_AGENTS,
        max_steps=max_steps,
        late_start=late_start,
        sample_interval=sample_interval,
        timeout_ms=timeout_ms,
    )


def summarize(
    episodes: list[dict[str, Any]], late_start: int
) -> dict[str, Any]:
    selected = [
        episode for episode in episodes if episode["reached_late_game"]
    ]
    samples = [
        sample
        for episode in selected
        for sample in episode["samples"]
    ]
    two_player_episodes = sum(
        any(row[1] == 2 for row in episode["alive_players_over_time"])
        for episode in selected
    )
    safe_opportunities = [
        sample
        for sample in samples
        if sample["possible_safe_bomb_opportunity"]
    ]
    declined_safe = [
        sample
        for sample in safe_opportunities
        if not sample["production_placed_bomb"]
    ]
    trapped = [
        sample for sample in samples if sample["possible_enemy_trapped"]
    ]
    trapped_declined = [
        sample
        for sample in trapped
        if not sample["production_placed_bomb"]
    ]
    far_enemy = [
        sample
        for sample in samples
        if sample["minimum_enemy_distance"] is not None
        and sample["minimum_enemy_distance"] >= 7
    ]
    open_map = [
        sample for sample in samples if sample["map_open_ratio"] >= 0.70
    ]
    item_farming = [
        sample for sample in samples if sample["item_farming_marker"]
    ]
    missed_pressure = [
        sample
        for sample in samples
        if sample["late_game_pressure_missed_marker"]
    ]
    bomb_actions = [
        sample for sample in samples if sample["production_placed_bomb"]
    ]
    rank_counts = Counter(str(ep["final_rank"]) for ep in selected)
    capacity_buckets: dict[str, Counter] = {}
    radius_buckets: dict[str, Counter] = {}
    for sample in samples:
        for bucket_map, key in (
            (capacity_buckets, str(sample["our_bombs_left"])),
            (radius_buckets, str(sample["our_bomb_radius"])),
        ):
            row = bucket_map.setdefault(key, Counter())
            row["samples"] += 1
            row["bomb_actions"] += int(sample["production_placed_bomb"])
            row["safe_opportunities"] += int(
                sample["possible_safe_bomb_opportunity"]
            )
            row["declined_safe_opportunities"] += int(
                sample["possible_safe_bomb_opportunity"]
                and not sample["production_placed_bomb"]
            )

    def ratio(count: int, total: int) -> float | None:
        return count / total if total else None

    return {
        "episodes_run": len(episodes),
        "late_start_step": late_start,
        "late_game_episodes": len(selected),
        "late_game_episode_frequency": ratio(
            len(selected), len(episodes)
        ),
        "rank_distribution": dict(sorted(rank_counts.items())),
        "strict_wins": sum(ep["strict_win"] for ep in selected),
        "timeout_finishes": sum(
            ep["final_step"] >= 500 for ep in selected
        ),
        "two_player_endgame_episodes": two_player_episodes,
        "two_player_endgame_frequency": ratio(
            two_player_episodes, len(selected)
        ),
        "sampled_frames": len(samples),
        "map_snapshots": sum("map" in sample for sample in samples),
        "safe_bomb_opportunities": len(safe_opportunities),
        "safe_bomb_opportunities_declined": len(declined_safe),
        "safe_bomb_decline_frequency": ratio(
            len(declined_safe), len(safe_opportunities)
        ),
        "enemy_trapped_markers": len(trapped),
        "enemy_trapped_markers_declined": len(trapped_declined),
        "far_enemy_samples": len(far_enemy),
        "far_enemy_frequency": ratio(len(far_enemy), len(samples)),
        "open_map_samples": len(open_map),
        "open_map_frequency": ratio(len(open_map), len(samples)),
        "item_farming_samples": len(item_farming),
        "item_farming_frequency": ratio(
            len(item_farming), len(samples)
        ),
        "late_game_pressure_missed_samples": len(missed_pressure),
        "late_game_pressure_missed_frequency": ratio(
            len(missed_pressure), len(samples)
        ),
        "bomb_action_samples": len(bomb_actions),
        "bomb_action_frequency": ratio(len(bomb_actions), len(samples)),
        "enemy_kills_by_observed_production": sum(
            ep["enemy_kills_by_us"] for ep in selected
        ),
        "bomb_capacity_buckets": {
            key: dict(value)
            for key, value in sorted(capacity_buckets.items())
        },
        "bomb_radius_buckets": {
            key: dict(value)
            for key, value in sorted(radius_buckets.items())
        },
        "runtime": {
            "timeouts": sum(
                ep["runtime"]["timeouts"] for ep in episodes
            ),
            "errors": sum(ep["runtime"]["errors"] for ep in episodes),
            "invalid": sum(ep["runtime"]["invalid"] for ep in episodes),
        },
    }


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def bucket_table(
    buckets: dict[str, dict[str, int]], label: str
) -> list[str]:
    lines = [
        f"| {label} | Samples | Bomb actions | Safe opportunities | Declined |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in buckets.items():
        lines.append(
            f"| {key} | {row.get('samples', 0)} | "
            f"{row.get('bomb_actions', 0)} | "
            f"{row.get('safe_opportunities', 0)} | "
            f"{row.get('declined_safe_opportunities', 0)} |"
        )
    return lines


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    config = payload["config"]
    frame_count = summary["sampled_frames"]
    safe_frame_frequency = (
        summary["safe_bomb_opportunities"] / frame_count
        if frame_count
        else None
    )
    trapped_frame_frequency = (
        summary["enemy_trapped_markers"] / frame_count
        if frame_count
        else None
    )
    trapped_decline_frequency = (
        summary["enemy_trapped_markers_declined"]
        / summary["enemy_trapped_markers"]
        if summary["enemy_trapped_markers"]
        else None
    )
    lines = [
        "# Late-Game Conversion Samples",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Scope",
        "",
        "This is targeted Phase 1.5 data collection. All four players used the current production submission/agent.py; no candidate behavior or benchmark comparison was introduced.",
        "",
        f"Snapshots begin at step 350 and are stored every {config['sample_interval']} steps. Opportunity markers are heuristic diagnostics, not ground-truth tactical labels.",
        "",
        "## Dataset",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Episodes run | {summary['episodes_run']} |",
        f"| Episodes reaching step {summary['late_start_step']} | {summary['late_game_episodes']} ({fmt_pct(summary['late_game_episode_frequency'])}) |",
        f"| Sampled late-game frames | {frame_count} |",
        f"| Stored map snapshots | {summary['map_snapshots']} |",
        f"| Timeout finishes | {summary['timeout_finishes']} |",
        f"| Strict wins by observed player | {summary['strict_wins']} |",
        f"| Two-player endgames | {summary['two_player_endgame_episodes']} ({fmt_pct(summary['two_player_endgame_frequency'])}) |",
        f"| Rank distribution | {json.dumps(summary['rank_distribution'], sort_keys=True)} |",
        f"| Runtime timeout / error / invalid | {summary['runtime']['timeouts']} / {summary['runtime']['errors']} / {summary['runtime']['invalid']} |",
        "",
        "## Conversion Signals",
        "",
        "| Signal | Count | Frequency |",
        "| --- | ---: | ---: |",
        f"| Possible safe bomb opportunities | {summary['safe_bomb_opportunities']} | {fmt_pct(safe_frame_frequency)} |",
        f"| Safe bomb opportunities declined | {summary['safe_bomb_opportunities_declined']} | {fmt_pct(summary['safe_bomb_decline_frequency'])} of opportunities |",
        f"| Possible enemy trapped markers | {summary['enemy_trapped_markers']} | {fmt_pct(trapped_frame_frequency)} |",
        f"| Trapped markers without bomb | {summary['enemy_trapped_markers_declined']} | {fmt_pct(trapped_decline_frequency)} |",
        f"| Enemies too far away, distance >=7 | {summary['far_enemy_samples']} | {fmt_pct(summary['far_enemy_frequency'])} |",
        f"| Open-map frames, >=70% passable | {summary['open_map_samples']} | {fmt_pct(summary['open_map_frequency'])} |",
        f"| Item-farming movement markers | {summary['item_farming_samples']} | {fmt_pct(summary['item_farming_frequency'])} |",
        f"| Missed pressure markers | {summary['late_game_pressure_missed_samples']} | {fmt_pct(summary['late_game_pressure_missed_frequency'])} |",
        f"| Production bomb actions in sampled frames | {summary['bomb_action_samples']} | {fmt_pct(summary['bomb_action_frequency'])} |",
        f"| Enemy kills attributed to observed production bombs | {summary['enemy_kills_by_observed_production']} | n/a |",
        "",
        "## Bomb Capacity",
        "",
    ]
    lines.extend(
        bucket_table(summary["bomb_capacity_buckets"], "Bombs left")
    )
    lines.extend(["", "## Bomb Radius", ""])
    lines.extend(
        bucket_table(summary["bomb_radius_buckets"], "Radius")
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Two-player endgames occurred in {fmt_pct(summary['two_player_endgame_frequency'])} of late-game episodes.",
            f"- Production declined {fmt_pct(summary['safe_bomb_decline_frequency'])} of heuristic safe enemy-pressure bomb opportunities.",
            f"- Enemies were at least seven cells away in {fmt_pct(summary['far_enemy_frequency'])} of sampled frames.",
            f"- The map remained highly open in {fmt_pct(summary['open_map_frequency'])} of sampled frames.",
            f"- Item-directed movement was detected in {fmt_pct(summary['item_farming_frequency'])} of sampled frames.",
            "",
            "These measurements identify where to inspect production conversion behavior, but they do not authorize a behavior change. Safe-bomb and trapped-enemy markers should be manually spot-checked against stored snapshots before Phase 2 design.",
            "",
            "## Files",
            "",
            "- Dataset: logs/late_game_conversion_samples.json",
            "- Collector: scripts/participant/collect_late_game_conversion_samples.py",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=71000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--late-start", type=int, default=350)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--timeout-ms", type=float, default=100.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/late_game_conversion_samples.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/LATE_GAME_CONVERSION_SAMPLES.md"),
    )
    args = parser.parse_args()
    if not 1 <= args.episodes <= 100:
        raise ValueError("--episodes must be between 1 and 100")
    if args.late_start < 0 or args.late_start >= args.max_steps:
        raise ValueError("--late-start must be within the episode")
    if args.sample_interval not in (5, 10):
        raise ValueError("--sample-interval must be 5 or 10")
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be between 1 and 8")

    production_path = ROOT / "submission" / "agent.py"
    agent_class, module = load_production_class(production_path)
    tasks = [
        (
            args.seed_start + episode_idx,
            episode_idx % 4,
            args.max_steps,
            args.late_start,
            args.sample_interval,
            args.timeout_ms,
        )
        for episode_idx in range(args.episodes)
    ]
    episodes = []
    if args.workers == 1:
        agents = [agent_class(idx) for idx in range(4)]
        for task in tasks:
            episodes.append(
                run_episode(
                    seed=task[0],
                    our_id=task[1],
                    agents=agents,
                    max_steps=task[2],
                    late_start=task[3],
                    sample_interval=task[4],
                    timeout_ms=task[5],
                )
            )
            if len(episodes) % args.progress_every == 0:
                print(f"completed {len(episodes)}/{args.episodes} episodes")
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=worker_init,
            initargs=(str(production_path),),
        ) as executor:
            for episode in executor.map(worker_collect, tasks):
                episodes.append(episode)
                if len(episodes) % args.progress_every == 0:
                    print(
                        f"completed {len(episodes)}/{args.episodes} episodes"
                    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "conversion-optimization-phase-1.5",
        "production_path": str(production_path.relative_to(ROOT)),
        "production_team_id": getattr(agent_class, "team_id", None),
        "config": {
            "episodes": args.episodes,
            "seed_start": args.seed_start,
            "max_steps": args.max_steps,
            "late_start": args.late_start,
            "sample_interval": args.sample_interval,
            "workers": args.workers,
            "timeout_ms": args.timeout_ms,
            "hybrid_model_enabled": bool(
                getattr(module, "HYBRID_MODEL_ENABLE", False)
            ),
            "all_four_players_use_current_production": True,
        },
        "marker_definitions": {
            "possible_safe_bomb_opportunity": (
                "Production can place a bomb, a time-aware escape exists, "
                "and an enemy is in the hypothetical blast or has "
                "estimated escape area <=2."
            ),
            "possible_enemy_trapped": (
                "An enemy is in the hypothetical blast and its estimated "
                "reachable safe area under that bomb is <=2 cells."
            ),
            "item_farming_marker": (
                "The sampled movement reduces distance to an item without "
                "reducing distance to the nearest enemy."
            ),
            "late_game_pressure_missed_marker": (
                "Our cell is not in immediate bomb danger, reachable safe "
                "area is >=8, nearest enemy is within 5 cells, and the "
                "selected action neither bombs nor reduces enemy distance."
            ),
        },
        "summary": summarize(episodes, args.late_start),
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
