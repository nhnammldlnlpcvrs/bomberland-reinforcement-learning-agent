from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.action_mask import legal_action_mask
from agent.rl_agent_pure.constants import BLAST_DELTAS, BOARD_SIZE, BOMB_TIMER, MOVE_ACTIONS, MOVE_DELTAS, PLACE_BOMB, TILE_BOX, TILE_WALL
from agent.rl_agent_pure.encoder import encode_observation
from agent.rl_agent_pure.utils import bomb_positions, boxes_in_blast, compute_danger_map, normalize_obs, passable, reachable_area
from engine.game import BomberEnv
from ml.evaluate_rl_pure import OPPONENTS, make_eval_agents, prepare_agent_path

REASONS = {
    "no_escape_after_bomb": 1,
    "dangerous_current_or_future": 2,
    "escape_path_too_narrow": 3,
    "dead_end_after_bomb": 4,
    "no_value_no_pressure": 5,
}


def _enemy_in_blast_line(board, players, row: int, col: int, agent_id: int) -> bool:
    radius = 1 + max(0, int(players[agent_id, 4]))
    for idx, player in enumerate(players):
        if idx == agent_id or not int(player[2]):
            continue
        erow, ecol = int(player[0]), int(player[1])
        if erow == row and abs(ecol - col) <= radius:
            return True
        if ecol == col and abs(erow - row) <= radius:
            return True
    return False


def _nearby_box_count(board, row: int, col: int, radius: int = 2) -> int:
    count = 0
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if abs(dr) + abs(dc) > radius:
                continue
            rr, cc = row + dr, col + dc
            if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and int(board[rr, cc]) == TILE_BOX:
                count += 1
    return count


def _farming_spot(board, row: int, col: int) -> bool:
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        return False
    open_neighbors = 0
    box_neighbors = 0
    for dr, dc in BLAST_DELTAS:
        rr, cc = row + dr, col + dc
        if not (0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE):
            continue
        tile = int(board[rr, cc])
        open_neighbors += int(tile != TILE_WALL and tile != TILE_BOX)
        box_neighbors += int(tile == TILE_BOX)
    return box_neighbors > 0 and open_neighbors <= 2


def _place_bomb_sim(board, players, bombs, agent_id: int):
    row, col = int(players[agent_id, 0]), int(players[agent_id, 1])
    placed = np.array([[row, col, BOMB_TIMER, agent_id]], dtype=np.int16)
    return placed if bombs.size == 0 else np.vstack([bombs, placed])


def _escape_search(board, players, bombs, agent_id: int, max_steps: int):
    row, col = int(players[agent_id, 0]), int(players[agent_id, 1])
    sim_bombs = _place_bomb_sim(board, players, bombs, agent_id)
    danger = compute_danger_map(board, players, sim_bombs)
    bombs_set = bomb_positions(sim_bombs)
    q = deque([(row, col, 0)])
    seen = {(row, col)}
    safe_cells = 0
    exits = 0
    while q:
        rr, cc, dist = q.popleft()
        if danger[rr, cc] > dist + 1:
            safe_cells += 1
            if danger[rr, cc] > BOMB_TIMER:
                exits += 1
        if dist >= max_steps:
            continue
        for action in MOVE_ACTIONS:
            dr, dc = MOVE_DELTAS[action]
            nr, nc = rr + dr, cc + dc
            if (nr, nc) in seen or not passable(board, bombs_set, nr, nc):
                continue
            if danger[nr, nc] <= dist + 2:
                continue
            seen.add((nr, nc))
            q.append((nr, nc, dist + 1))
    return safe_cells, exits, danger


def _make_sample(obs, agent_id: int, args):
    board, players, bombs, step = normalize_obs(obs)
    if not int(players[agent_id, 2]):
        return None
    mask = legal_action_mask(obs, agent_id)
    if not mask[PLACE_BOMB]:
        return None
    row, col = int(players[agent_id, 0]), int(players[agent_id, 1])
    would_destroy = int(boxes_in_blast(board, players, row, col, agent_id))
    nearby_boxes = _nearby_box_count(board, row, col, args.nearby_radius)
    enemy_pressure = _enemy_in_blast_line(board, players, row, col, agent_id)
    farming = _farming_spot(board, row, col)
    if would_destroy <= 0 and nearby_boxes <= 0 and not enemy_pressure and not farming:
        return None

    safe_cells, exits, danger = _escape_search(board, players, bombs, agent_id, args.escape_search_steps)
    current_danger = int(danger[row, col])
    future_danger = current_danger
    reachable_after = int(reachable_area(board, bomb_positions(_place_bomb_sim(board, players, bombs, agent_id)), danger, (row, col), max_depth=10).sum())
    has_escape = safe_cells > 1 and exits > 0
    reasons = []
    if current_danger <= args.min_safe_danger:
        reasons.append(REASONS["dangerous_current_or_future"])
    if not has_escape:
        reasons.append(REASONS["no_escape_after_bomb"])
    if safe_cells <= args.min_safe_cells and (would_destroy > 0 or enemy_pressure):
        reasons.append(REASONS["escape_path_too_narrow"])
    if reachable_after <= args.dead_end_area:
        reasons.append(REASONS["dead_end_after_bomb"])
    if would_destroy <= 0 and not enemy_pressure:
        reasons.append(REASONS["no_value_no_pressure"])
    if not reasons:
        return None

    encoded = encode_observation(obs, agent_id)
    scalars = np.asarray([
        current_danger / 9999.0,
        future_danger / 9999.0,
        would_destroy / 7.0,
        step / max(1, args.max_steps),
        safe_cells / max(1, args.escape_search_steps + 1),
    ], dtype=np.float32)
    return {
        "obs": encoded,
        "scalars": scalars,
        "reason": int(reasons[0]),
        "nearby_box_count": int(nearby_boxes),
        "would_destroy_boxes": int(would_destroy),
        "has_escape_after_bomb": int(has_escape),
        "danger_current": current_danger,
        "danger_future": future_danger,
        "reachable_after_bomb": reachable_after,
        "safe_escape_cells": int(safe_cells),
        "escape_exits": int(exits),
        "enemy_pressure": int(enemy_pressure),
        "farming_spot": int(farming),
        "step": int(step),
    }


def generate(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    policy_agent_path = prepare_agent_path(args.policy)
    opponent_paths = [OPPONENTS.get(name, name) for name in args.opponents]
    env = BomberEnv(max_steps=args.max_steps, seed=args.seed)
    samples = []
    candidates_scanned = 0
    reason_counts = {name: 0 for name in REASONS}

    for ep in range(args.episodes):
        opponent = opponent_paths[ep % len(opponent_paths)]
        roster = [policy_agent_path, opponent, opponent, opponent]
        agents = make_eval_agents(roster, seed=args.seed + ep)
        obs = {**env.reset(seed=args.seed + ep), "step": 0}
        done = False
        step = 0
        while not done and step < args.max_steps:
            sample = _make_sample(obs, 0, args)
            if sample is not None:
                candidates_scanned += 1
                if len(samples) < args.max_samples:
                    samples.append(sample)
                    reason_name = next((name for name, code in REASONS.items() if code == sample["reason"]), "unknown")
                    reason_counts[reason_name] = reason_counts.get(reason_name, 0) + 1
            actions = []
            for agent in agents:
                try:
                    action = int(agent.act(obs))
                except Exception:
                    action = 0
                actions.append(action if 0 <= action <= 5 else 0)
            obs, terminated, truncated = env.step(actions)
            step += 1
            obs = {**obs, "step": step}
            done = terminated or truncated
        if len(samples) >= args.max_samples:
            break

    if not samples:
        raise ValueError("No counterfactual negatives generated")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=np.stack([s["obs"] for s in samples]).astype(np.float32),
        scalar_features=np.stack([s["scalars"] for s in samples]).astype(np.float32),
        labels=np.zeros(len(samples), dtype=np.float32),
        source=np.full(len(samples), 5, dtype=np.int8),
        sample_type=np.full(len(samples), 4, dtype=np.int8),
        negative_reason=np.asarray([s["reason"] for s in samples], dtype=np.int16),
        nearby_box_count=np.asarray([s["nearby_box_count"] for s in samples], dtype=np.int16),
        would_destroy_boxes=np.asarray([s["would_destroy_boxes"] for s in samples], dtype=np.int16),
        has_escape_after_bomb=np.asarray([s["has_escape_after_bomb"] for s in samples], dtype=np.int8),
        danger_current=np.asarray([s["danger_current"] for s in samples], dtype=np.int16),
        danger_future=np.asarray([s["danger_future"] for s in samples], dtype=np.int16),
        reachable_after_bomb=np.asarray([s["reachable_after_bomb"] for s in samples], dtype=np.int16),
        safe_escape_cells=np.asarray([s["safe_escape_cells"] for s in samples], dtype=np.int16),
        escape_exits=np.asarray([s["escape_exits"] for s in samples], dtype=np.int16),
        enemy_pressure=np.asarray([s["enemy_pressure"] for s in samples], dtype=np.int8),
        farming_spot=np.asarray([s["farming_spot"] for s in samples], dtype=np.int8),
        step=np.asarray([s["step"] for s in samples], dtype=np.int16),
    )
    stats = {
        "policy": args.policy,
        "opponents": args.opponents,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "output": str(output),
        "candidate_states_scanned": int(candidates_scanned),
        "negatives_generated": int(len(samples)),
        "reason_counts": reason_counts,
        "nearby_box_count_mean": float(np.mean([s["nearby_box_count"] for s in samples])),
        "would_destroy_boxes_mean": float(np.mean([s["would_destroy_boxes"] for s in samples])),
        "has_escape_rate": float(np.mean([s["has_escape_after_bomb"] for s in samples])),
        "value_proxy_count": int(sum(1 for s in samples if s["would_destroy_boxes"] > 0 or s["nearby_box_count"] > 0 or s["enemy_pressure"] or s["farming_spot"])),
    }
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Generate counterfactual tempting-but-unsafe bomb contexts.")
    parser.add_argument("--policy", default="ml/checkpoints/rl_agent_pure/ppo_bomb_selector_best.zip")
    parser.add_argument("--opponents", nargs="+", default=["random", "simple"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--output", default="ml/datasets/bomb_counterfactual_negatives.npz")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--nearby_radius", type=int, default=2)
    parser.add_argument("--escape_search_steps", type=int, default=7)
    parser.add_argument("--min_safe_danger", type=int, default=3)
    parser.add_argument("--min_safe_cells", type=int, default=3)
    parser.add_argument("--dead_end_area", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
