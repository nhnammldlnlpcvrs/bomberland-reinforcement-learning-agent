from __future__ import annotations

from collections import deque

import numpy as np

from agent.rl_agent_pure.constants import BOARD_SIZE, BOMB_TIMER, MOVE_ACTIONS, MOVE_DELTAS, PLACE_BOMB, STOP
from agent.rl_agent_pure.utils import (
    blast_cells,
    bomb_positions,
    boxes_in_blast,
    compute_danger_map,
    normalize_obs,
    passable,
    reachable_area,
)


def _self_pos(players: np.ndarray, agent_id: int) -> tuple[int, int]:
    return int(players[agent_id, 0]), int(players[agent_id, 1])


def _sim_bomb(board: np.ndarray, players: np.ndarray, bombs: np.ndarray, agent_id: int, timer: int = BOMB_TIMER) -> np.ndarray:
    row, col = _self_pos(players, agent_id)
    placed = np.array([[row, col, timer, agent_id]], dtype=np.int16)
    return placed if bombs.size == 0 else np.vstack([bombs, placed])


def _enemy_in_blast(board: np.ndarray, players: np.ndarray, row: int, col: int, agent_id: int) -> bool:
    bomb = np.array([row, col, BOMB_TIMER, agent_id], dtype=np.int16)
    cells = set(blast_cells(board, players, bomb))
    for idx, player in enumerate(players):
        if idx != agent_id and int(player[2]) and (int(player[0]), int(player[1])) in cells:
            return True
    return False


def would_destroy_boxes(obs, pos=None, radius=None, agent_id: int = 0) -> int:
    """Return the number of boxes a bomb at pos would hit under current board geometry."""
    board, players, _bombs, _step = normalize_obs(obs)
    if pos is None:
        pos = _self_pos(players, agent_id)
    row, col = int(pos[0]), int(pos[1])
    if radius is not None:
        players = players.copy()
        players[agent_id, 4] = max(0, int(radius) - 1)
    return int(boxes_in_blast(board, players, row, col, agent_id))


def is_in_blast_corridor(obs, pos=None, agent_id: int = 0) -> bool:
    """True when pos is on a currently threatened bomb line."""
    board, players, bombs, _step = normalize_obs(obs)
    if bombs.size == 0:
        return False
    if pos is None:
        pos = _self_pos(players, agent_id)
    target = (int(pos[0]), int(pos[1]))
    for bomb in bombs:
        if target in set(blast_cells(board, players, bomb)):
            return True
    return False


def reachable_safe_tiles(obs, pos=None, agent_id: int = 0, max_depth: int = 10, after_placing_bomb: bool = False) -> np.ndarray:
    board, players, bombs, _step = normalize_obs(obs)
    if pos is None:
        pos = _self_pos(players, agent_id)
    if after_placing_bomb:
        bombs = _sim_bomb(board, players, bombs, agent_id)
    danger = compute_danger_map(board, players, bombs)
    return reachable_area(board, bomb_positions(bombs), danger, (int(pos[0]), int(pos[1])), max_depth=max_depth)


def has_escape_after_bomb(obs, agent_id: int = 0, min_safe_tiles: int = 2) -> bool:
    board, players, bombs, _step = normalize_obs(obs)
    if not int(players[agent_id, 2]):
        return False
    row, col = _self_pos(players, agent_id)
    if (row, col) in bomb_positions(bombs):
        return False
    area = reachable_safe_tiles(obs, (row, col), agent_id=agent_id, max_depth=10, after_placing_bomb=True)
    return bool(area.sum() >= min_safe_tiles)


def find_escape_only_state(obs, agent_id: int = 0) -> dict:
    board, players, bombs, _step = normalize_obs(obs)
    if not int(players[agent_id, 2]) or bombs.size == 0:
        return {"ok": False, "reason": "no_alive_agent_or_bomb"}
    row, col = _self_pos(players, agent_id)
    danger = compute_danger_map(board, players, bombs)
    in_corridor = is_in_blast_corridor(obs, (row, col), agent_id)
    area = reachable_area(board, bomb_positions(bombs), danger, (row, col), max_depth=8)
    ok = in_corridor and area.sum() >= 2
    return {
        "ok": bool(ok),
        "reason": "ok" if ok else "no_escape_corridor_pattern",
        "safe_tiles": int(area.sum()),
        "current_danger": int(danger[row, col]),
    }


def find_bomb_then_escape_state(obs, agent_id: int = 0) -> dict:
    board, players, bombs, _step = normalize_obs(obs)
    if not int(players[agent_id, 2]):
        return {"ok": False, "reason": "agent_dead"}
    row, col = _self_pos(players, agent_id)
    bombs_left = int(players[agent_id, 3])
    useful = boxes_in_blast(board, players, row, col, agent_id) > 0 or _enemy_in_blast(board, players, row, col, agent_id)
    escape = has_escape_after_bomb(obs, agent_id)
    ok = bombs_left > 0 and (row, col) not in bomb_positions(bombs) and useful and escape
    return {
        "ok": bool(ok),
        "reason": "ok" if ok else "no_useful_safe_bomb",
        "would_destroy_boxes": int(boxes_in_blast(board, players, row, col, agent_id)),
        "has_escape_after_bomb": bool(escape),
    }


def find_bomb_box_value_state(obs, agent_id: int = 0) -> dict:
    base = find_bomb_then_escape_state(obs, agent_id)
    ok = bool(base.get("ok")) and int(base.get("would_destroy_boxes", 0)) > 0
    base["ok"] = ok
    base["reason"] = "ok" if ok else "no_safe_box_bomb"
    return base


def shortest_safe_escape_action(obs, agent_id: int = 0) -> int:
    """Best-effort helper for diagnostics/curriculum fallbacks; not used in inference."""
    board, players, bombs, _step = normalize_obs(obs)
    row, col = _self_pos(players, agent_id)
    danger = compute_danger_map(board, players, bombs)
    bombs_set = bomb_positions(bombs)
    q = deque([((row, col), STOP)])
    seen = {(row, col)}
    while q:
        (cr, cc), first_action = q.popleft()
        if danger[cr, cc] > BOMB_TIMER and not is_in_blast_corridor(obs, (cr, cc), agent_id):
            return int(first_action)
        for action in MOVE_ACTIONS:
            dr, dc = MOVE_DELTAS[action]
            nr, nc = cr + dr, cc + dc
            if (nr, nc) in seen or not passable(board, bombs_set, nr, nc):
                continue
            seen.add((nr, nc))
            q.append(((nr, nc), action if first_action == STOP else first_action))
    return STOP


def scenario_summary(obs, agent_id: int = 0) -> dict:
    board, players, bombs, _step = normalize_obs(obs)
    row, col = _self_pos(players, agent_id)
    danger = compute_danger_map(board, players, bombs)
    return {
        "pos": (row, col),
        "bombs": int(len(bombs)),
        "boxes_in_blast": int(boxes_in_blast(board, players, row, col, agent_id)),
        "current_danger": int(danger[row, col]),
        "in_blast_corridor": bool(is_in_blast_corridor(obs, (row, col), agent_id)),
        "safe_tiles": int(reachable_safe_tiles(obs, (row, col), agent_id).sum()),
        "safe_tiles_after_bomb": int(reachable_safe_tiles(obs, (row, col), agent_id, after_placing_bomb=True).sum()),
    }
