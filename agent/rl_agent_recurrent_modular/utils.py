from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

try:
    from .constants import (
        BLAST_DELTAS,
        BOARD_SIZE,
        BOMB_TIMER,
        MOVE_ACTIONS,
        MOVE_DELTAS,
        TILE_BOX,
        TILE_GRASS,
        TILE_ITEM_CAPACITY,
        TILE_ITEM_RADIUS,
        TILE_WALL,
    )
except ImportError:  # Loaded as a submission-local module.
    from constants import (
        BLAST_DELTAS,
        BOARD_SIZE,
        BOMB_TIMER,
        MOVE_ACTIONS,
        MOVE_DELTAS,
        TILE_BOX,
        TILE_GRASS,
        TILE_ITEM_CAPACITY,
        TILE_ITEM_RADIUS,
        TILE_WALL,
    )

INF = 9999


def normalize_obs(obs):
    board = np.asarray(obs.get("map"), dtype=np.int16)
    if board.shape != (BOARD_SIZE, BOARD_SIZE):
        fixed = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int16)
        rows = min(BOARD_SIZE, board.shape[0] if board.ndim >= 1 else 0)
        cols = min(BOARD_SIZE, board.shape[1] if board.ndim >= 2 else 0)
        if rows and cols:
            fixed[:rows, :cols] = board[:rows, :cols]
        board = fixed

    players = np.asarray(obs.get("players", np.zeros((4, 5))), dtype=np.int16)
    if players.size == 0:
        players = np.zeros((4, 5), dtype=np.int16)
    if players.ndim == 1:
        players = players.reshape(1, -1)
    fixed_players = np.zeros((4, 5), dtype=np.int16)
    fixed_players[: min(4, players.shape[0]), : min(5, players.shape[1])] = players[
        : min(4, players.shape[0]), : min(5, players.shape[1])
    ]

    bombs = np.asarray(obs.get("bombs", np.zeros((0, 4))), dtype=np.int16)
    if bombs.size == 0:
        bombs = np.zeros((0, 4), dtype=np.int16)
    elif bombs.ndim == 1:
        bombs = bombs.reshape(1, -1)
    fixed_bombs = np.zeros((bombs.shape[0], 4), dtype=np.int16)
    if bombs.shape[0]:
        fixed_bombs[:, : min(4, bombs.shape[1])] = bombs[:, : min(4, bombs.shape[1])]

    step = int(obs.get("step", obs.get("_step", 0)) or 0)
    return board, fixed_players, fixed_bombs, step


def in_bounds(row, col):
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def interior(row, col):
    return 0 < row < BOARD_SIZE - 1 and 0 < col < BOARD_SIZE - 1


def bomb_positions(bombs):
    return {(int(b[0]), int(b[1])) for b in np.asarray(bombs).reshape(-1, 4)}


def passable(board, bombs_set, row, col):
    if not interior(row, col):
        return False
    if int(board[row, col]) in (TILE_WALL, TILE_BOX):
        return False
    return (row, col) not in bombs_set


def blast_cells(board, players, bomb):
    row, col, _timer, owner = [int(x) for x in bomb[:4]]
    radius = 1
    if 0 <= owner < len(players):
        radius += max(0, int(players[owner, 4]))
    cells = [(row, col)]
    for dr, dc in BLAST_DELTAS:
        for dist in range(1, radius + 1):
            nr, nc = row + dr * dist, col + dc * dist
            if not in_bounds(nr, nc) or int(board[nr, nc]) == TILE_WALL:
                break
            cells.append((nr, nc))
            if int(board[nr, nc]) == TILE_BOX:
                break
    return cells


def compute_danger_map(board, players, bombs):
    danger = np.full((BOARD_SIZE, BOARD_SIZE), INF, dtype=np.int16)
    if bombs.size == 0:
        return danger
    timers = [max(0, int(b[2])) for b in bombs]
    blasts = [set(blast_cells(board, players, b)) for b in bombs]
    for _ in range(len(timers)):
        changed = False
        for i, cells in enumerate(blasts):
            if timers[i] <= 0:
                continue
            for j, other in enumerate(bombs):
                if i != j and timers[j] > 0 and (int(other[0]), int(other[1])) in cells and timers[i] < timers[j]:
                    timers[j] = timers[i]
                    changed = True
        if not changed:
            break
    for timer, cells in zip(timers, blasts):
        if timer <= 0:
            continue
        for row, col in cells:
            danger[row, col] = min(danger[row, col], timer)
    return danger


def reachable_area(board, bombs_set, danger, start, max_depth=12):
    out = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if not in_bounds(*start):
        return out
    q = deque([(start[0], start[1], 0)])
    seen = {start}
    while q:
        row, col, dist = q.popleft()
        if danger[row, col] > dist + 1:
            out[row, col] = 1.0
        if dist >= max_depth:
            continue
        for action in MOVE_ACTIONS:
            dr, dc = MOVE_DELTAS[action]
            nr, nc = row + dr, col + dc
            if (nr, nc) in seen or not passable(board, bombs_set, nr, nc):
                continue
            seen.add((nr, nc))
            q.append((nr, nc, dist + 1))
    return out


def boxes_in_blast(board, players, row, col, agent_id):
    bomb = np.array([row, col, BOMB_TIMER, agent_id], dtype=np.int16)
    return sum(1 for r, c in blast_cells(board, players, bomb) if int(board[r, c]) == TILE_BOX)


def has_escape_after_bomb(board, players, bombs, agent_id):
    row, col = int(players[agent_id, 0]), int(players[agent_id, 1])
    placed = np.array([[row, col, BOMB_TIMER, agent_id]], dtype=np.int16)
    sim_bombs = placed if bombs.size == 0 else np.vstack([bombs, placed])
    danger = compute_danger_map(board, players, sim_bombs)
    area = reachable_area(board, bomb_positions(sim_bombs), danger, (row, col), max_depth=10)
    return bool(area.sum() > 1)


def repo_root_from_file(file):
    return Path(file).resolve().parents[2]
