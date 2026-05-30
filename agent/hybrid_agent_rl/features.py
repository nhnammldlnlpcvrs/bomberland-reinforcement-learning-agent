from collections import deque

import numpy as np

from constants import (
    BLAST_DIRS,
    BOARD_SIZE,
    DIRS,
    INF,
    MOVE_ACTIONS,
    TILE_BOX,
    TILE_CAPACITY,
    TILE_RADIUS,
    TILE_WALL,
)
from safety import compute_danger_map
from utils import bomb_set, in_bounds, is_passable

CHANNEL_NAMES = [
    "wall",
    "box",
    "radius_item",
    "capacity_item",
    "self_position",
    "enemy_positions",
    "bomb_positions",
    "bomb_timer_normalized",
    "danger_now",
    "danger_future",
    "reachable",
    "dead_end",
    "frontier_box_adjacent",
    "safe_item_target",
    "center_control",
]
NUM_CHANNELS = len(CHANNEL_NAMES)


def _reachable_plane(start, game_map, bombs, danger):
    reachable = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if not is_passable(start[0], start[1], game_map, bombs):
        return reachable
    q = deque([(start, 0)])
    visited = {start}
    while q:
        pos, dist = q.popleft()
        if danger[pos[0], pos[1]] <= dist + 1:
            continue
        reachable[pos[0], pos[1]] = 1.0
        for action in MOVE_ACTIONS:
            dr, dc = DIRS[action]
            nxt = (pos[0] + dr, pos[1] + dc)
            if nxt in visited:
                continue
            if not is_passable(nxt[0], nxt[1], game_map, bombs):
                continue
            visited.add(nxt)
            q.append((nxt, dist + 1))
    return reachable


def _dead_end_plane(game_map, bombs):
    dead_end = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for r in range(1, BOARD_SIZE - 1):
        for c in range(1, BOARD_SIZE - 1):
            if not is_passable(r, c, game_map, bombs):
                continue
            free = 0
            for dr, dc in BLAST_DIRS:
                if is_passable(r + dr, c + dc, game_map, bombs):
                    free += 1
            if free <= 1:
                dead_end[r, c] = 1.0
            elif free == 2:
                dead_end[r, c] = 0.45
    return dead_end


def _frontier_plane(game_map, bombs):
    frontier = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for r in range(1, BOARD_SIZE - 1):
        for c in range(1, BOARD_SIZE - 1):
            if not is_passable(r, c, game_map, bombs):
                continue
            boxes = 0
            for dr, dc in BLAST_DIRS:
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc) and int(game_map[nr, nc]) == TILE_BOX:
                    boxes += 1
            frontier[r, c] = min(1.0, boxes / 3.0)
    return frontier


def _center_plane():
    center = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    mid = BOARD_SIZE // 2
    max_dist = float((BOARD_SIZE - 1) * 2)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            center[r, c] = 1.0 - ((abs(r - mid) + abs(c - mid)) / max_dist)
    return center


def encode_obs(obs, agent_id, max_steps=500):
    del max_steps
    game_map = np.asarray(obs["map"])
    players = np.asarray(obs["players"])
    bombs = np.asarray(obs["bombs"], dtype=np.int32)
    danger = compute_danger_map(game_map, players, bombs)
    bset = bomb_set(bombs)

    wall = (game_map == TILE_WALL).astype(np.float32)
    box = (game_map == TILE_BOX).astype(np.float32)
    radius_item = (game_map == TILE_RADIUS).astype(np.float32)
    capacity_item = (game_map == TILE_CAPACITY).astype(np.float32)

    self_position = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    enemy_positions = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    start = (int(players[agent_id][0]), int(players[agent_id][1]))
    for idx, player in enumerate(players):
        if int(player[2]) != 1:
            continue
        r, c = int(player[0]), int(player[1])
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            continue
        if idx == agent_id:
            self_position[r, c] = 1.0
            start = (r, c)
        else:
            enemy_positions[r, c] = 1.0

    bomb_positions = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    bomb_timer = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs.size > 0:
        if bombs.ndim == 1:
            bombs = bombs.reshape(1, -1)
        for br, bc, timer, _owner in bombs:
            br, bc = int(br), int(bc)
            if 0 <= br < BOARD_SIZE and 0 <= bc < BOARD_SIZE:
                bomb_positions[br, bc] = 1.0
                bomb_timer[br, bc] = max(0.0, min(float(timer) / 7.0, 1.0))

    danger_now = (danger <= 1).astype(np.float32)
    danger_future = np.where(danger < INF, np.clip((8.0 - danger) / 7.0, 0.0, 1.0), 0.0).astype(np.float32)
    reachable = _reachable_plane(start, game_map, bset, danger)
    dead_end = _dead_end_plane(game_map, bset)
    frontier = _frontier_plane(game_map, bset)
    safe_item = (((radius_item > 0) | (capacity_item > 0)) & (reachable > 0)).astype(np.float32)
    center_control = _center_plane() * reachable

    return np.stack(
        [
            wall,
            box,
            radius_item,
            capacity_item,
            self_position,
            enemy_positions,
            bomb_positions,
            bomb_timer,
            danger_now,
            danger_future,
            reachable,
            dead_end,
            frontier,
            safe_item,
            center_control,
        ],
        axis=0,
    ).astype(np.float32)
