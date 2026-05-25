"""Compact Dueling DQN model and Bomberland feature/safety utilities."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - agent.py handles missing torch.
    torch = None
    nn = None


BOARD_SIZE = 13
NUM_ACTIONS = 6
SPATIAL_CHANNELS = 18
SCALAR_DIM = 8
INF = 9999
BOMB_TIMER = 7

TILE_GRASS = 0
TILE_WALL = 1
TILE_BOX = 2
TILE_RADIUS = 3
TILE_CAPACITY = 4

STOP = 0
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4
PLACE_BOMB = 5
MOVE_DELTAS = {
    STOP: (0, 0),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
    UP: (0, -1),
    DOWN: (0, 1),
}
MOVE_ACTIONS = (LEFT, RIGHT, UP, DOWN)
BLAST_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


if nn is not None:
    class DuelingDQN(nn.Module):
        def __init__(self, spatial_channels=SPATIAL_CHANNELS, scalar_dim=SCALAR_DIM, num_actions=NUM_ACTIONS, hidden_dim=128):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(spatial_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.scalar = nn.Sequential(
                nn.Linear(scalar_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
            )
            conv_dim = 64 * BOARD_SIZE * BOARD_SIZE
            feat_dim = conv_dim + 32
            self.value = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.advantage = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions),
            )

        def forward(self, spatial, scalar):
            spatial_feat = self.conv(spatial).flatten(1)
            scalar_feat = self.scalar(scalar)
            feat = torch.cat([spatial_feat, scalar_feat], dim=1)
            value = self.value(feat)
            advantage = self.advantage(feat)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
else:
    DuelingDQN = None


def _array(value, shape=None, dtype=np.float32):
    if value is None:
        return np.zeros(shape or (0,), dtype=dtype)
    arr = np.asarray(value, dtype=dtype)
    if shape is not None and arr.shape != shape:
        out = np.zeros(shape, dtype=dtype)
        if arr.ndim >= 2:
            rows = min(shape[0], arr.shape[0])
            cols = min(shape[1], arr.shape[1])
            out[:rows, :cols] = arr[:rows, :cols]
        return out
    return arr


def normalize_obs(obs):
    board = _array(obs.get("map"), (BOARD_SIZE, BOARD_SIZE), np.int16)
    players = _array(obs.get("players"), dtype=np.int16)
    if players.size == 0:
        players = np.zeros((4, 5), dtype=np.int16)
    if players.ndim == 1:
        players = players.reshape(1, -1)
    if players.shape[0] < 4 or players.shape[1] < 5:
        padded = np.zeros((4, 5), dtype=np.int16)
        rows = min(4, players.shape[0])
        cols = min(5, players.shape[1])
        padded[:rows, :cols] = players[:rows, :cols]
        players = padded
    bombs = _array(obs.get("bombs"), dtype=np.int16)
    if bombs.size == 0:
        bombs = np.zeros((0, 4), dtype=np.int16)
    elif bombs.ndim == 1:
        bombs = bombs.reshape(1, -1)
    if bombs.shape[1] < 4:
        padded = np.zeros((bombs.shape[0], 4), dtype=np.int16)
        padded[:, :bombs.shape[1]] = bombs
        bombs = padded
    step = int(obs.get("step", obs.get("_step", 0)) or 0)
    return board, players[:4, :5], bombs[:, :4], step


def in_bounds(row, col):
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def interior(row, col):
    return 0 < row < BOARD_SIZE - 1 and 0 < col < BOARD_SIZE - 1


def bomb_positions(bombs: np.ndarray) -> set[tuple[int, int]]:
    return {(int(b[0]), int(b[1])) for b in bombs}


def blast_cells(board, players, bomb):
    row, col, timer, owner = [int(x) for x in bomb[:4]]
    radius = 1
    if 0 <= owner < len(players):
        radius = 1 + max(0, int(players[owner, 4]))
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
    blast_sets = [set(blast_cells(board, players, b)) for b in bombs]
    for _ in range(len(bombs)):
        changed = False
        for i, cells in enumerate(blast_sets):
            if timers[i] <= 0:
                continue
            for j, other in enumerate(bombs):
                if i == j or timers[j] <= 0:
                    continue
                if (int(other[0]), int(other[1])) in cells and timers[i] < timers[j]:
                    timers[j] = timers[i]
                    changed = True
        if not changed:
            break
    for timer, cells in zip(timers, blast_sets):
        if timer <= 0:
            continue
        for row, col in cells:
            danger[row, col] = min(danger[row, col], timer)
    return danger


def passable(board, bombs_set, row, col):
    if not interior(row, col):
        return False
    if int(board[row, col]) in (TILE_WALL, TILE_BOX):
        return False
    return (row, col) not in bombs_set


def reachable_safe_plane(board, bombs_set, danger, start, max_depth=10):
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
            if danger[nr, nc] <= dist + 1:
                continue
            seen.add((nr, nc))
            q.append((nr, nc, dist + 1))
    return out


def has_escape_after_bomb(board, players, bombs, agent_id, max_depth=8):
    my_row, my_col = int(players[agent_id, 0]), int(players[agent_id, 1])
    placed = np.array([[my_row, my_col, BOMB_TIMER, agent_id]], dtype=np.int16)
    sim_bombs = placed if bombs.size == 0 else np.vstack([bombs, placed])
    sim_danger = compute_danger_map(board, players, sim_bombs)
    sim_bombs_set = bomb_positions(sim_bombs)
    reachable = reachable_safe_plane(board, sim_bombs_set, sim_danger, (my_row, my_col), max_depth=max_depth)
    return bool(reachable.sum() > 0)


def boxes_in_blast(board, players, row, col, agent_id):
    radius = 1 + max(0, int(players[agent_id, 4]))
    count = 0
    for dr, dc in BLAST_DELTAS:
        for dist in range(1, radius + 1):
            nr, nc = row + dr * dist, col + dc * dist
            if not in_bounds(nr, nc) or int(board[nr, nc]) == TILE_WALL:
                break
            if int(board[nr, nc]) == TILE_BOX:
                count += 1
                break
    return count


def enemy_in_blast(board, players, row, col, agent_id):
    radius = 1 + max(0, int(players[agent_id, 4]))
    for dr, dc in BLAST_DELTAS:
        for dist in range(1, radius + 1):
            nr, nc = row + dr * dist, col + dc * dist
            if not in_bounds(nr, nc) or int(board[nr, nc]) == TILE_WALL:
                break
            for idx, player in enumerate(players):
                if idx != agent_id and int(player[2]) and int(player[0]) == nr and int(player[1]) == nc:
                    return True
            if int(board[nr, nc]) == TILE_BOX:
                break
    return False


def safe_action_mask(obs, agent_id):
    try:
        board, players, bombs, _step = normalize_obs(obs)
        agent_id = int(agent_id)
        if agent_id < 0 or agent_id >= len(players) or not int(players[agent_id, 2]):
            return np.array([True, False, False, False, False, False], dtype=bool)
        my_row, my_col = int(players[agent_id, 0]), int(players[agent_id, 1])
        bombs_set = bomb_positions(bombs)
        danger = compute_danger_map(board, players, bombs)
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        if danger[my_row, my_col] > 1:
            mask[STOP] = True
        for action in MOVE_ACTIONS:
            dr, dc = MOVE_DELTAS[action]
            nr, nc = my_row + dr, my_col + dc
            if passable(board, bombs_set, nr, nc) and danger[nr, nc] > 1:
                mask[action] = True
        if int(players[agent_id, 3]) > 0 and (my_row, my_col) not in bombs_set and danger[my_row, my_col] > 1:
            meaningful = boxes_in_blast(board, players, my_row, my_col, agent_id) > 0 or enemy_in_blast(board, players, my_row, my_col, agent_id)
            if meaningful and has_escape_after_bomb(board, players, bombs, agent_id):
                mask[PLACE_BOMB] = True
        if danger[my_row, my_col] <= 3:
            escape = first_safe_escape_action(board, bombs_set, danger, (my_row, my_col))
            if escape is not None:
                emergency = np.zeros(NUM_ACTIONS, dtype=bool)
                emergency[escape] = True
                return emergency
        if not mask.any():
            mask[STOP] = True
        return mask
    except Exception:
        return np.array([True, False, False, False, False, False], dtype=bool)


def first_safe_escape_action(board, bombs_set, danger, start, max_depth=12):
    q = deque([(start[0], start[1], None, 0)])
    seen = {start}
    while q:
        row, col, first, dist = q.popleft()
        if dist > 0 and danger[row, col] == INF:
            return first
        if dist >= max_depth:
            continue
        for action in MOVE_ACTIONS:
            dr, dc = MOVE_DELTAS[action]
            nr, nc = row + dr, col + dc
            if (nr, nc) in seen or not passable(board, bombs_set, nr, nc):
                continue
            if danger[nr, nc] <= dist + 2:
                continue
            seen.add((nr, nc))
            q.append((nr, nc, action if first is None else first, dist + 1))
    return None


def encode_observation(obs, agent_id):
    board, players, bombs, step = normalize_obs(obs)
    agent_id = max(0, min(int(agent_id), len(players) - 1))
    danger = compute_danger_map(board, players, bombs)
    bombs_set = bomb_positions(bombs)
    my_row, my_col = int(players[agent_id, 0]), int(players[agent_id, 1])

    planes = [
        (board == TILE_WALL).astype(np.float32),
        (board == TILE_BOX).astype(np.float32),
        (board == TILE_GRASS).astype(np.float32),
        (board == TILE_RADIUS).astype(np.float32),
        (board == TILE_CAPACITY).astype(np.float32),
    ]
    my_pos = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if int(players[agent_id, 2]) and in_bounds(my_row, my_col):
        my_pos[my_row, my_col] = 1.0
    enemy_pos = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for idx, player in enumerate(players):
        row, col, alive = int(player[0]), int(player[1]), int(player[2])
        if idx != agent_id and alive and in_bounds(row, col):
            enemy_pos[row, col] = 1.0
    bomb_plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    bomb_timer = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    my_bomb = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for bomb in bombs:
        row, col, timer, owner = [int(x) for x in bomb[:4]]
        if in_bounds(row, col):
            bomb_plane[row, col] = 1.0
            bomb_timer[row, col] = max(bomb_timer[row, col], timer / BOMB_TIMER)
            if owner == agent_id:
                my_bomb[row, col] = 1.0
    danger_planes = [((danger <= horizon).astype(np.float32)) for horizon in range(1, 8)]
    walkable = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    walkable[(board == TILE_WALL) | (board == TILE_BOX)] = 0.0
    for row, col in bombs_set:
        if in_bounds(row, col):
            walkable[row, col] = 0.0
    safe_reachable = reachable_safe_plane(board, bombs_set, danger, (my_row, my_col), max_depth=10)
    spatial = np.stack([
        *planes,
        my_pos,
        enemy_pos,
        bomb_plane,
        bomb_timer,
        my_bomb,
        *danger_planes,
        safe_reachable,
    ]).astype(np.float32)
    alive_enemies = sum(1 for idx, p in enumerate(players) if idx != agent_id and int(p[2]))
    scalar = np.array([
        float(players[agent_id, 3]) / 5.0,
        float(players[agent_id, 4]) / 4.0,
        float(alive_enemies) / 3.0,
        float(step) / 500.0,
        float(danger[my_row, my_col] if in_bounds(my_row, my_col) and danger[my_row, my_col] < INF else BOMB_TIMER) / BOMB_TIMER,
        float(safe_reachable.sum()) / (BOARD_SIZE * BOARD_SIZE),
        float((board == TILE_BOX).sum()) / (BOARD_SIZE * BOARD_SIZE),
        float(len(bombs)) / 16.0,
    ], dtype=np.float32)
    return spatial, scalar


def fallback_policy(obs, agent_id, q_values: Iterable[float] | None = None):
    mask = safe_action_mask(obs, agent_id)
    if q_values is not None:
        q = np.asarray(list(q_values), dtype=np.float32)
        if q.shape[0] >= NUM_ACTIONS:
            masked = np.where(mask, q[:NUM_ACTIONS], -1.0e9)
            return int(np.argmax(masked))
    try:
        board, players, bombs, _ = normalize_obs(obs)
        my_row, my_col = int(players[agent_id, 0]), int(players[agent_id, 1])
        danger = compute_danger_map(board, players, bombs)
        if danger[my_row, my_col] <= 3:
            escape = first_safe_escape_action(board, bomb_positions(bombs), danger, (my_row, my_col))
            if escape is not None and mask[escape]:
                return int(escape)
        for action in (PLACE_BOMB, LEFT, RIGHT, UP, DOWN, STOP):
            if mask[action]:
                return int(action)
    except Exception:
        pass
    safe = np.flatnonzero(mask)
    return int(safe[0]) if safe.size else STOP
