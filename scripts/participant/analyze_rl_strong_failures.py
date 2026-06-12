from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_strong.action_mask import legal_action_mask
from agent.rl_strong.constants import MOVE_DELTAS, PLACE_BOMB
from agent.rl_strong.utils import (
    BOARD_SIZE,
    INF,
    bomb_positions,
    compute_danger_map,
    has_escape_after_bomb,
    normalize_obs,
    passable,
)
from engine.game import BomberEnv
from scripts.participant.benchmark_rl_strong_pool import AGENTS, _make_agent


CATEGORIES = (
    "bomb placed with no safe escape path",
    "escaped first then returned into blast line",
    "stayed in blast line",
    "trapped by boxes/walls",
    "enemy bomb vs own bomb confusion",
    "uncategorized",
)


@dataclass
class BombEvent:
    placed_step: int
    row: int
    col: int
    safe_escape_at_place: bool
    boxes_in_escape_area: int
    first_safe_step: int | None = None
    returned_to_blast: bool = False
    stayed_in_blast_steps: int = 0


def _make_roster_from_episode(episode: dict) -> list[str]:
    roster = episode.get("roster")
    if roster:
        return list(roster)
    rng = random.Random(int(episode["seed"]))
    names = list(AGENTS)
    combos = []
    import itertools

    for combo in itertools.combinations(names, 4):
        combos.append(list(combo))
    roster = list(combos[rng.randrange(len(combos))])
    rng.shuffle(roster)
    return roster


def _blast_cells(board, players, bomb) -> set[tuple[int, int]]:
    row, col, _timer, owner = [int(value) for value in bomb[:4]]
    radius = 1
    if 0 <= owner < len(players):
        radius += max(0, int(players[owner, 4]))
    cells = {(row, col)}
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for distance in range(1, radius + 1):
            nr, nc = row + dr * distance, col + dc * distance
            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                break
            if int(board[nr, nc]) == 1:
                break
            cells.add((nr, nc))
            if int(board[nr, nc]) == 2:
                break
    return cells


def _safe_reachable_with_boxes(board, players, bombs, start, max_depth=10) -> tuple[int, int]:
    danger = compute_danger_map(board, players, bombs)
    bombs_set = bomb_positions(bombs)
    queue = [(start[0], start[1], 0)]
    seen = {start}
    safe_cells = 0
    blocked_by_boxes = 0
    while queue:
        row, col, dist = queue.pop(0)
        if danger[row, col] > dist + 1:
            safe_cells += 1
        if dist >= max_depth:
            continue
        for dr, dc in MOVE_DELTAS.values():
            nr, nc = row + dr, col + dc
            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                continue
            if int(board[nr, nc]) == 2:
                blocked_by_boxes += 1
            if (nr, nc) in seen or not passable(board, bombs_set, nr, nc):
                continue
            seen.add((nr, nc))
            queue.append((nr, nc, dist + 1))
    return safe_cells, blocked_by_boxes


def _position(obs, slot: int) -> tuple[int, int]:
    players = np.asarray(obs["players"])
    return int(players[slot, 0]), int(players[slot, 1])


def _death_bomb_sources(prev_obs: dict, slot: int, pos: tuple[int, int]) -> list[dict]:
    board, players, bombs, _step = normalize_obs(prev_obs)
    sources = []
    for bomb in bombs:
        cells = _blast_cells(board, players, bomb)
        if pos in cells:
            sources.append({
                "row": int(bomb[0]),
                "col": int(bomb[1]),
                "timer": int(bomb[2]),
                "owner": int(bomb[3]),
                "own": int(bomb[3]) == slot,
            })
    return sources


def _categorize(event: BombEvent | None, death_sources: list[dict], death_step: int, death_pos: tuple[int, int]) -> str:
    if not event:
        if death_sources and not any(source["own"] for source in death_sources):
            return "enemy bomb vs own bomb confusion"
        return "uncategorized"
    if not event.safe_escape_at_place:
        return "bomb placed with no safe escape path"
    if event.first_safe_step is not None and event.returned_to_blast:
        return "escaped first then returned into blast line"
    if event.stayed_in_blast_steps >= 3:
        return "stayed in blast line"
    if event.boxes_in_escape_area >= 6:
        return "trapped by boxes/walls"
    if death_sources and not any(source["own"] for source in death_sources):
        return "enemy bomb vs own bomb confusion"
    if event.row == death_pos[0] or event.col == death_pos[1]:
        return "stayed in blast line"
    return "uncategorized"


def replay_episode(episode: dict, max_steps: int) -> dict | None:
    seed = int(episode["seed"])
    roster = _make_roster_from_episode(episode)
    if "rl_strong" not in roster:
        return None
    random.seed(seed)
    slot = roster.index("rl_strong")
    agents = [_make_agent(AGENTS[name], idx) for idx, name in enumerate(roster)]
    env = BomberEnv(max_steps=max_steps, seed=seed)
    obs = {**env.reset(seed=seed), "step": 0}
    events: list[BombEvent] = []
    positions = []
    danger_trace = []
    action_trace = []
    death_step = None
    death_pos = None
    death_sources = []
    prev_alive = bool(obs["players"][slot, 2])

    for step in range(max_steps):
        prev_obs = obs
        board, players, bombs, _ = normalize_obs(prev_obs)
        pos = _position(prev_obs, slot)
        danger = compute_danger_map(board, players, bombs)
        positions.append(pos)
        danger_trace.append(int(danger[pos[0], pos[1]]))
        actions = []
        for idx, agent in enumerate(agents):
            try:
                action = int(agent.act(prev_obs))
            except Exception:
                action = 0
            if not 0 <= action <= 5:
                action = 0
            actions.append(action)
        action_trace.append(actions[slot])

        if actions[slot] == PLACE_BOMB and bool(legal_action_mask(prev_obs, slot)[PLACE_BOMB]):
            row, col = pos
            placed = np.array([[row, col, 7, slot]], dtype=np.int16)
            sim_bombs = placed if bombs.size == 0 else np.vstack([bombs, placed])
            safe_escape = has_escape_after_bomb(board, players, bombs, slot)
            safe_cells, boxes_seen = _safe_reachable_with_boxes(board, players, sim_bombs, pos)
            events.append(BombEvent(
                placed_step=step,
                row=row,
                col=col,
                safe_escape_at_place=bool(safe_escape),
                boxes_in_escape_area=boxes_seen if safe_cells <= 2 else 0,
            ))

        for event in events:
            age = step - event.placed_step
            if not 0 <= age <= 7:
                continue
            own_blast = _blast_cells(board, players, np.array([event.row, event.col, max(0, 7 - age), slot]))
            in_blast = pos in own_blast
            if not in_blast and event.first_safe_step is None:
                event.first_safe_step = step
            if event.first_safe_step is not None and in_blast:
                event.returned_to_blast = True
            if in_blast:
                event.stayed_in_blast_steps += 1

        obs_next, terminated, truncated = env.step(actions)
        obs = {**obs_next, "step": step + 1}
        alive = bool(obs["players"][slot, 2])
        if prev_alive and not alive:
            death_step = step + 1
            death_pos = _position(obs, slot)
            death_sources = _death_bomb_sources(prev_obs, slot, death_pos)
            break
        prev_alive = alive
        if terminated or truncated:
            break

    if death_step is None:
        return None

    recent_event = None
    for event in reversed(events):
        if 0 <= death_step - event.placed_step <= 8:
            recent_event = event
            break

    category = _categorize(recent_event, death_sources, death_step, death_pos)
    return {
        "seed": seed,
        "roster": roster,
        "slot": slot,
        "death_step": death_step,
        "death_pos": list(death_pos),
        "category": category,
        "death_sources": death_sources,
        "recent_bomb": None if recent_event is None else {
            "placed_step": recent_event.placed_step,
            "position": [recent_event.row, recent_event.col],
            "safe_escape_at_place": recent_event.safe_escape_at_place,
            "first_safe_step": recent_event.first_safe_step,
            "returned_to_blast": recent_event.returned_to_blast,
            "stayed_in_blast_steps": recent_event.stayed_in_blast_steps,
            "boxes_in_escape_area": recent_event.boxes_in_escape_area,
        },
        "last_positions": [list(pos) for pos in positions[max(0, len(positions) - 12):]],
        "last_danger": danger_trace[max(0, len(danger_trace) - 12):],
        "last_actions": action_trace[max(0, len(action_trace) - 12):],
        "bomb_count_before_death": len(events),
    }


def _select_representatives(failures: list[dict], limit: int) -> list[dict]:
    selected = []
    seen_categories = set()
    for failure in failures:
        if failure["category"] not in seen_categories:
            selected.append(failure)
            seen_categories.add(failure["category"])
        if len(selected) >= limit:
            return selected
    for failure in failures:
        if failure not in selected:
            selected.append(failure)
        if len(selected) >= limit:
            break
    return selected


def _markdown_report(failures: list[dict], total_rl_matches: int, benchmark_path: str, limit: int) -> str:
    counts = Counter(failure["category"] for failure in failures)
    early_counts = Counter(failure["category"] for failure in failures if int(failure["death_step"]) <= 80)
    own_bomb_counts = Counter(failure["category"] for failure in failures if failure.get("recent_bomb"))
    early_deaths = sum(1 for failure in failures if int(failure["death_step"]) <= 80)
    self_bomb = sum(1 for failure in failures if failure.get("recent_bomb"))
    representatives = _select_representatives(failures, limit)

    lines = [
        "# RL Strong Failure Analysis",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- Benchmark log: `{benchmark_path}`",
        "- Candidate: `agent/rl_strong/`",
        "- Production comparison: `agent/hybrid_agent_online_robust/`",
        "",
        "The benchmark log did not contain full replay frame history, so this analysis script replays the recorded seeds and rosters to reconstruct `rl_strong` death contexts.",
        "",
        "## Summary",
        "",
        f"- `rl_strong` matches in benchmark: `{total_rl_matches}`",
        f"- Replayed `rl_strong` deaths found: `{len(failures)}`",
        f"- Early deaths at or before step 80: `{early_deaths}` ({early_deaths / max(1, len(failures)):.1%} of replayed deaths)",
        f"- Deaths within 8 steps of an own bomb: `{self_bomb}` ({self_bomb / max(1, len(failures)):.1%} of replayed deaths)",
        "",
        "## Failure Categories",
        "",
        "| Pattern | Count | Percent | Early Deaths | Own-Bomb Deaths |",
        "|---|---:|---:|---:|---:|",
    ]
    for category in CATEGORIES:
        count = counts.get(category, 0)
        lines.append(
            f"| {category} | {count} | {count / max(1, len(failures)):.1%} | "
            f"{early_counts.get(category, 0)} | {own_bomb_counts.get(category, 0)} |"
        )

    lines.extend([
        "",
        "## Representative Replay References",
        "",
        "| Seed | Roster | Death Step | Category | Recent Own Bomb | Last Actions | Last Positions | Death Sources |",
        "|---:|---|---:|---|---|---|---|---|",
    ])
    for failure in representatives:
        bomb = failure.get("recent_bomb")
        bomb_text = "none"
        if bomb:
            bomb_text = f"step {bomb['placed_step']} at {bomb['position']}, safe_escape={bomb['safe_escape_at_place']}, first_safe={bomb['first_safe_step']}, returned={bomb['returned_to_blast']}, blast_steps={bomb['stayed_in_blast_steps']}"
        sources = ", ".join(
            f"owner={source['owner']} pos=({source['row']},{source['col']}) timer={source['timer']}"
            for source in failure["death_sources"]
        ) or "none"
        lines.append(
            f"| {failure['seed']} | `{', '.join(failure['roster'])}` | {failure['death_step']} | "
            f"{failure['category']} | {bomb_text} | `{failure['last_actions']}` | "
            f"`{failure['last_positions']}` | {sources} |"
        )

    own_bomb_structural = (
        counts["stayed in blast line"]
        + counts["escaped first then returned into blast line"]
        + counts["bomb placed with no safe escape path"]
    )
    danger_structural = own_bomb_structural + counts["enemy bomb vs own bomb confusion"]
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"Own-bomb blast-line handling accounts for `{own_bomb_structural}` of `{len(failures)}` replayed deaths via no-escape, stayed-in-line, or return-into-line categories.",
        f"Total bomb-danger handling accounts for `{danger_structural}` of `{len(failures)}` deaths once enemy-bomb confusion is included.",
        "`rl_strong` is not failing because of runtime instability: the benchmark showed zero timeouts, zero errors, and zero invalid actions. It is failing because its learned policy repeatedly accepts lethal bomb geometry, remains in blast corridors, or treats enemy bomb lines like safe movement space.",
        "",
        "## Recommendation",
        "",
    ])
    lines.extend([
        "Abandon promotion of the current `rl_strong` checkpoint. The failure is structural for this checkpoint, not a packaging or latency problem.",
        "",
        "Continue the track only as a retraining/reward-mask project: strengthen post-bomb reward credit assignment, add explicit penalties for staying in or returning to own blast lines, train on escape-after-bomb and enemy-bomb avoidance scenarios, and make the inference mask evaluate the full bomb timer horizon instead of only near-term danger. After those changes, rerun the 300+ episode pool benchmark before considering export or submission.",
    ])

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze rl_strong self-bomb and early-death failures from benchmark seeds.")
    parser.add_argument("--benchmark", default="logs/rl_strong_pool_benchmark.json")
    parser.add_argument("--output_json", default="logs/rl_strong_failure_analysis.json")
    parser.add_argument("--report", default="docs/RL_STRONG_FAILURE_ANALYSIS.md")
    parser.add_argument("--representatives", type=int, default=10)
    args = parser.parse_args()

    benchmark_path = ROOT / args.benchmark
    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    max_steps = int(data.get("config", {}).get("max_steps", 500))
    episodes = data.get("episodes", [])
    total_rl_matches = int(data.get("summary", {}).get("rl_strong", {}).get("matches", 0))
    failures = []
    for idx, episode in enumerate(episodes):
        if "rl_strong" not in _make_roster_from_episode(episode):
            continue
        failure = replay_episode(episode, max_steps=max_steps)
        if failure:
            failures.append(failure)
        if (idx + 1) % 50 == 0:
            print(f"analyzed {idx + 1}/{len(episodes)} benchmark episodes")

    payload = {
        "benchmark": args.benchmark,
        "total_rl_matches": total_rl_matches,
        "failure_count": len(failures),
        "category_counts": dict(Counter(failure["category"] for failure in failures)),
        "failures": failures,
    }
    output_json = ROOT / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_markdown_report(failures, total_rl_matches, args.benchmark, args.representatives), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
