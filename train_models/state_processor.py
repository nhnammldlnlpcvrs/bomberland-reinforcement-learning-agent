"""
State Processor for Bomberland.

Converts raw game observations into:
  - (7, 13, 13) multi-channel tensor
  - (4,) scalar feature vector
  - (6,) legal action mask
  - Danger map for blast-zones with chain-reaction relaxation
"""

from collections import deque

import numpy as np
import torch

from train_models.config import (
    A_BOMB,
    A_DOWN,
    A_LEFT,
    A_RIGHT,
    A_STOP,
    A_UP,
    ALL_ACTIONS,
    BLAST_DIRS,
    BOARD_SIZE,
    BOMB_TIMER,
    DIR_DELTA,
    MAX_BOMB_CAPACITY,
    MAX_BOMB_RADIUS,
    MOVE_ACTIONS,
    TILE_BOX,
    TILE_CAPACITY,
    TILE_GRASS,
    TILE_RADIUS,
    TILE_WALL,
)


def _in_bounds(r: int, c: int) -> bool:
    return 0 < r < BOARD_SIZE - 1 and 0 < c < BOARD_SIZE - 1


def _bomb_set(bombs: np.ndarray) -> set:
    arr = np.asarray(bombs, dtype=np.int32)
    if arr.size == 0:
        return set()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return {(int(arr[i, 0]), int(arr[i, 1])) for i in range(arr.shape[0])}


def _is_passable(r: int, c: int, game_map: np.ndarray, bomb_positions: set) -> bool:
    if not _in_bounds(r, c):
        return False
    if game_map[r, c] in (TILE_WALL, TILE_BOX):
        return False
    if (r, c) in bomb_positions:
        return False
    return True


def _blast_cells(r: int, c: int, radius: int, game_map: np.ndarray) -> set:
    """All cells affected by a bomb at (r,c) with given blast radius."""
    cells = {(r, c)}
    for dr, dc in BLAST_DIRS:
        for step in range(1, radius + 1):
            nr, nc = r + dr * step, c + dc * step
            if not _in_bounds(nr, nc):
                break
            if game_map[nr, nc] == TILE_WALL:
                break
            cells.add((nr, nc))
            if game_map[nr, nc] == TILE_BOX:
                break
    return cells


def compute_danger_map(
    game_map: np.ndarray,
    players: np.ndarray,
    bombs: np.ndarray,
) -> np.ndarray:
    """
    danger[r][c] = steps until cell explodes (9999 = permanently safe).

    Propagates chain reactions: if bomb A's blast reaches bomb B, B inherits
    the earlier detonation time.
    """
    INF = 9999
    danger = np.full((BOARD_SIZE, BOARD_SIZE), INF, dtype=np.int32)

    bombs_arr = np.asarray(bombs, dtype=np.int32)
    if bombs_arr.size == 0:
        return danger
    if bombs_arr.ndim == 1:
        bombs_arr = bombs_arr.reshape(1, -1)

    n = bombs_arr.shape[0]
    blast_sets = []
    timers = []

    for i in range(n):
        br, bc, timer, owner = bombs_arr[i]
        t = int(timer)
        if t <= 0:
            blast_sets.append(set())
            timers.append(0)
            continue
        owner = int(owner)
        radius = 1 + int(players[owner][4]) if 0 <= owner < len(players) else 1
        blast_sets.append(_blast_cells(br, bc, radius, game_map))
        timers.append(t)

    effective = list(timers)

    # Relax chain reactions
    for _ in range(n):
        changed = False
        for i in range(n):
            if effective[i] <= 0:
                continue
            for j in range(n):
                if i == j or effective[j] <= 0:
                    continue
                bj_r, bj_c = int(bombs_arr[j, 0]), int(bombs_arr[j, 1])
                if (bj_r, bj_c) in blast_sets[i]:
                    if effective[i] < effective[j]:
                        effective[j] = effective[i]
                        changed = True
        if not changed:
            break

    for i in range(n):
        et = effective[i]
        if et <= 0:
            continue
        for r, c in blast_sets[i]:
            if et < danger[r, c]:
                danger[r, c] = et

    return danger


def _find_nearest_safe_bfs(
    start: tuple,
    game_map: np.ndarray,
    bomb_positions: set,
    danger: np.ndarray,
    max_depth: int = 10,
) -> bool:
    """True if a permanently-safe cell is reachable from start within max_depth."""
    q = deque()
    q.append((start[0], start[1], 0))
    visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    visited[start] = True

    while q:
        r, c, dist = q.popleft()
        if dist > 0 and danger[r, c] == 9999:
            return True
        if dist >= max_depth:
            continue
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if visited[nr, nc]:
                continue
            if not _is_passable(nr, nc, game_map, bomb_positions):
                continue
            if danger[nr, nc] <= dist + 2:
                continue
            visited[nr, nc] = True
            q.append((nr, nc, dist + 1))
    return False


def get_action_mask(
    obs: dict,
    agent_id: int,
) -> np.ndarray:
    """
    Returns boolean mask of shape (6,) where True = legal action.

    Masking rules:
      - Dead agent: only STOP is legal.
      - Movement into wall, box, active bomb, or out-of-bounds → illegal.
      - Movement into cell with danger_time <= 1 → illegal.
      - PLACE_BOMB: illegal if bombs_left=0, already standing on bomb,
        or no safe escape path exists after placing.
    """
    mask = np.zeros(6, dtype=bool)
    players = np.asarray(obs["players"], dtype=np.int32)
    game_map = np.asarray(obs["map"], dtype=np.int32)
    bombs = np.asarray(obs["bombs"], dtype=np.int32)

    r, c, alive = int(players[agent_id][0]), int(players[agent_id][1]), int(players[agent_id][2])
    bombs_left = int(players[agent_id][3])

    if not alive:
        mask[A_STOP] = True
        return mask

    bset = _bomb_set(bombs)
    danger = compute_danger_map(game_map, players, bombs)

    # STOP is always legal when alive
    mask[A_STOP] = True

    # Movement actions
    for action in MOVE_ACTIONS:
        dr, dc = DIR_DELTA[action]
        nr, nc = r + dr, c + dc
        if not _is_passable(nr, nc, game_map, bset):
            continue
        if danger[nr, nc] <= 1:
            continue
        mask[action] = True

    # Bomb placement
    if bombs_left > 0 and (r, c) not in bset:
        # Check escape path exists after placing bomb
        if _can_escape_after_bomb(obs, agent_id, game_map, bset, danger):
            mask[A_BOMB] = True

    # If no legal move and no bomb, at least allow STOP
    if not np.any(mask):
        mask[A_STOP] = True

    return mask


def _can_escape_after_bomb(
    obs: dict,
    agent_id: int,
    game_map: np.ndarray,
    bomb_positions: set,
    existing_danger: np.ndarray = None,
) -> bool:
    """Simulate placing a bomb; returns True if an escape path exists."""
    players = np.asarray(obs["players"], dtype=np.int32)
    bombs = np.asarray(obs["bombs"], dtype=np.int32)
    r, c = int(players[agent_id][0]), int(players[agent_id][1])

    if bombs.size == 0:
        new_bombs = np.array([[r, c, BOMB_TIMER, agent_id]], dtype=np.int32)
    else:
        if bombs.ndim == 1:
            bombs = bombs.reshape(1, -1)
        new_bomb = np.array([[r, c, BOMB_TIMER, agent_id]], dtype=np.int32)
        new_bombs = np.vstack([bombs, new_bomb])

    new_danger = compute_danger_map(game_map, players, new_bombs)
    return _find_nearest_safe_bfs((r, c), game_map, bomb_positions, new_danger)


def encode_observation(obs: dict, agent_id: int) -> tuple:
    """
    Encode a raw observation into:
      - state_tensor: (7, 13, 13) float32 tensor
      - scalars:       (4,) float32 tensor

    Channel layout:
      0: Wall
      1: Box
      2: Self position
      3: Opponent positions
      4: Bomb timers (normalized: (7 - timer) / 7)
      5: Danger zones (normalized: 1.0 - danger/7 clipped, 0=safe)
      6: Items (1=Radius, 2=Capacity, normalized to 0.5 and 1.0)
    """
    game_map = np.asarray(obs["map"], dtype=np.int32)
    players = np.asarray(obs["players"], dtype=np.int32)
    bombs_arr = np.asarray(obs["bombs"], dtype=np.int32)
    step = int(obs.get("step", obs.get("current_step", 0)) or 0)

    if bombs_arr.size > 0 and bombs_arr.ndim == 1:
        bombs_arr = bombs_arr.reshape(1, -1)

    channels = np.zeros((7, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Channel 0: Walls
    channels[0] = (game_map == TILE_WALL).astype(np.float32)

    # Channel 1: Boxes
    channels[1] = (game_map == TILE_BOX).astype(np.float32)

    # Channel 2: Self position
    my_r, my_c = int(players[agent_id][0]), int(players[agent_id][1])
    if int(players[agent_id][2]) and 0 <= my_r < BOARD_SIZE and 0 <= my_c < BOARD_SIZE:
        channels[2, my_r, my_c] = 1.0

    # Channel 3: Opponents
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1:
            continue
        er, ec = int(p[0]), int(p[1])
        if 0 <= er < BOARD_SIZE and 0 <= ec < BOARD_SIZE:
            channels[3, er, ec] = 1.0

    # Channel 4: Bomb timers (normalized)
    for bomb in bombs_arr:
        br, bc, timer, _owner = [int(x) for x in bomb[:4]]
        if 0 <= br < BOARD_SIZE and 0 <= bc < BOARD_SIZE and timer > 0:
            channels[4, br, bc] = (BOMB_TIMER - timer) / BOMB_TIMER

    # Channel 5: Danger zones
    danger = compute_danger_map(game_map, players, bombs_arr)
    for r_idx in range(BOARD_SIZE):
        for c_idx in range(BOARD_SIZE):
            d = danger[r_idx, c_idx]
            if d == 9999:
                channels[5, r_idx, c_idx] = 0.0
            else:
                channels[5, r_idx, c_idx] = max(0.0, 1.0 - d / BOMB_TIMER)

    # Channel 6: Items
    channels[6] = np.where(game_map == TILE_RADIUS, 0.5,
                  np.where(game_map == TILE_CAPACITY, 1.0, 0.0)).astype(np.float32)

    # Scalar features: [radius_bonus, capacity, bombs_left, step/500]
    radius_bonus = float(players[agent_id][4]) / (MAX_BOMB_RADIUS - 1) if int(players[agent_id][2]) else 0.0
    capacity = 0.2  # fixed proxy since capacity is not a direct field; bombs_left indicates capacity
    bombs_left_norm = float(players[agent_id][3]) / MAX_BOMB_CAPACITY if int(players[agent_id][2]) else 0.0
    step_norm = min(1.0, step / 500.0)

    scalars = np.array([radius_bonus, bombs_left_norm, capacity, step_norm], dtype=np.float32)

    state_tensor = torch.from_numpy(channels)
    scalar_tensor = torch.from_numpy(scalars)

    return state_tensor, scalar_tensor


def get_legal_action_mask_tensor(obs: dict, agent_id: int) -> torch.Tensor:
    """Convenience: returns (6,) bool tensor for action masking."""
    mask_np = get_action_mask(obs, agent_id)
    return torch.from_numpy(mask_np)
