"""Feature extraction for Bomberland replay imitation datasets."""

from collections import deque

import numpy as np


BOARD_SIZE = 13
INF = 9999

TILE_GRASS = 0
TILE_WALL = 1
TILE_BOX = 2
TILE_RADIUS = 3
TILE_CAPACITY = 4

CHANNEL_NAMES = [
    "walls",
    "crates",
    "bombs",
    "bomb_danger",
    "flames",
    "items",
    "self_position",
    "enemy_positions",
    "walkable_cells",
    "safe_cells",
    "center_distance",
    "territory_open_space",
]


def _as_array(value, shape=None, dtype=np.float32):
    if value is None:
        if shape is None:
            return np.zeros((0,), dtype=dtype)
        return np.zeros(shape, dtype=dtype)
    arr = np.asarray(value, dtype=dtype)
    if shape is not None and arr.shape != shape:
        out = np.zeros(shape, dtype=dtype)
        rows = min(shape[0], arr.shape[0]) if arr.ndim >= 1 else 0
        cols = min(shape[1], arr.shape[1]) if arr.ndim >= 2 else 0
        if rows and cols:
            out[:rows, :cols] = arr[:rows, :cols]
        return out
    return arr


def _frame_map(frame):
    board = frame.get("map", frame.get("board", frame.get("grid")))
    return _as_array(board, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int16)


def _frame_players(frame):
    players = frame.get("players", frame.get("player_state", frame.get("agents")))
    arr = _as_array(players, dtype=np.int16)
    if arr.size == 0:
        return np.zeros((0, 5), dtype=np.int16)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 5:
        padded = np.zeros((arr.shape[0], 5), dtype=np.int16)
        padded[:, :arr.shape[1]] = arr
        arr = padded
    return arr


def _frame_bombs(frame):
    bombs = frame.get("bombs")
    arr = _as_array(bombs, dtype=np.int16)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.int16)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 4:
        padded = np.zeros((arr.shape[0], 4), dtype=np.int16)
        padded[:, :arr.shape[1]] = arr
        arr = padded
    return arr


def _in_bounds(row, col):
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def _blast_cells(board, players, bomb):
    row, col, _timer, owner = [int(x) for x in bomb[:4]]
    radius = 1
    if 0 <= owner < len(players):
        radius = 1 + max(0, int(players[owner, 4]))

    cells = [(row, col)]
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for distance in range(1, radius + 1):
            nr, nc = row + dr * distance, col + dc * distance
            if not _in_bounds(nr, nc):
                break
            if int(board[nr, nc]) == TILE_WALL:
                break
            cells.append((nr, nc))
            if int(board[nr, nc]) == TILE_BOX:
                break
    return cells


def compute_danger(board, players, bombs):
    """Compute a simple bomb danger map normalized later by the encoder."""
    danger = np.full((BOARD_SIZE, BOARD_SIZE), INF, dtype=np.float32)
    for bomb in bombs:
        timer = int(bomb[2]) if len(bomb) > 2 else 0
        if timer <= 0:
            continue
        for row, col in _blast_cells(board, players, bomb):
            danger[row, col] = min(danger[row, col], float(timer))
    return danger


def _mark_points(plane, points, value=1.0):
    for item in points or []:
        if isinstance(item, dict):
            row = int(item.get("row", item.get("x", item.get("r", -1))))
            col = int(item.get("col", item.get("y", item.get("c", -1))))
        else:
            if len(item) < 2:
                continue
            row, col = int(item[0]), int(item[1])
        if _in_bounds(row, col):
            plane[row, col] = value


def _walkable_plane(board, bombs):
    walkable = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    walkable[(board == TILE_WALL) | (board == TILE_BOX)] = 0.0
    for bomb in bombs:
        row, col = int(bomb[0]), int(bomb[1])
        if _in_bounds(row, col):
            walkable[row, col] = 0.0
    return walkable


def _territory_open_space(walkable):
    territory = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if walkable[row, col] <= 0:
                continue
            free = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = row + dr, col + dc
                if _in_bounds(nr, nc) and walkable[nr, nc] > 0:
                    free += 1
            territory[row, col] = free / 4.0
    return territory


def encode_observation(frame, team_name=None):
    """
    Convert replay frame into compact ML-ready features.

    Returns a dict with:
    - tensor: float32 array shaped (12, 13, 13)
    - channel_names: names for each plane
    - agent_index: encoded self index
    """
    del team_name  # The builder injects _agent_index into frames.

    board = _frame_map(frame)
    players = _frame_players(frame)
    bombs = _frame_bombs(frame)
    agent_index = int(frame.get("_agent_index", 0))

    walls = (board == TILE_WALL).astype(np.float32)
    crates = (board == TILE_BOX).astype(np.float32)
    bomb_plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for bomb in bombs:
        row, col = int(bomb[0]), int(bomb[1])
        if _in_bounds(row, col):
            bomb_plane[row, col] = 1.0

    danger = compute_danger(board, players, bombs)
    bomb_danger = np.where(danger < INF, np.clip((8.0 - danger) / 7.0, 0.0, 1.0), 0.0)

    flames = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    _mark_points(flames, frame.get("flames", frame.get("explosions")), value=1.0)

    items = ((board == TILE_RADIUS) | (board == TILE_CAPACITY)).astype(np.float32)
    self_position = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    enemy_positions = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    alive_mask = []
    for idx, player in enumerate(players):
        row, col, alive = int(player[0]), int(player[1]), int(player[2])
        alive_mask.append(alive)
        if not alive or not _in_bounds(row, col):
            continue
        if idx == agent_index:
            self_position[row, col] = 1.0
        else:
            enemy_positions[row, col] = 1.0

    walkable = _walkable_plane(board, bombs)
    safe = ((walkable > 0) & (danger > 1)).astype(np.float32)

    center_distance = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    center = (BOARD_SIZE // 2, BOARD_SIZE // 2)
    max_dist = float((BOARD_SIZE - 1) * 2)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            dist = abs(row - center[0]) + abs(col - center[1])
            center_distance[row, col] = 1.0 - (dist / max_dist)

    territory = _territory_open_space(walkable)

    tensor = np.stack(
        [
            walls,
            crates,
            bomb_plane,
            bomb_danger,
            flames,
            items,
            self_position,
            enemy_positions,
            walkable,
            safe,
            center_distance,
            territory,
        ],
        axis=0,
    ).astype(np.float32)

    return {
        "tensor": tensor,
        "channel_names": list(CHANNEL_NAMES),
        "agent_index": agent_index,
        "alive_mask": alive_mask,
        "step": int(frame.get("step", frame.get("_step", 0)) or 0),
    }


def handcrafted_features(obs):
    """
    Extract compact tabular features from a replay frame or obs-like dict.

    This is intentionally small and dependency-free; future models can use it
    for tiny linear/Q-head baselines.
    """
    encoded = encode_observation(obs)
    tensor = encoded["tensor"]
    return np.array(
        [
            float(tensor[8].sum()),   # walkable cells
            float(tensor[9].sum()),   # safe cells
            float(tensor[2].sum()),   # bombs
            float(tensor[5].sum()),   # items
            float(tensor[11].mean()), # average local openness
        ],
        dtype=np.float32,
    )

