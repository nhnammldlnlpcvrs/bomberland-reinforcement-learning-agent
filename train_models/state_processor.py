"""
State Processor for Bomberland.

Converts raw game observations into:
  - (7, 13, 13) multi-channel tensor (legacy v1)
  - (16, 13, 13) multi-channel tensor (v2, production)
  - (4,) scalar feature vector
  - (6,) legal action mask (production-grade)
  - Danger map for blast-zones with chain-reaction relaxation
  - TemporalStateProcessor: 4-frame stacking wrapper
"""

from collections import deque
from typing import Optional

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
    FRAME_STACK,
    MAX_BOMB_CAPACITY,
    MAX_BOMB_RADIUS,
    MOVE_ACTIONS,
    STATE_CHANNELS,
    STATE_CHANNELS_V2,
    TILE_BOX,
    TILE_CAPACITY,
    TILE_GRASS,
    TILE_RADIUS,
    TILE_WALL,
)

INF = 9999
CURRENT_DANGER_ESCAPE_THRESHOLD = 4
ACTION_ESCAPE_LOOKAHEAD = 8
MAX_BFS_ESCAPE = 8
MIN_SAFE_CELLS_AFTER_BOMB = 5


# ═══════════════════════════════════════════════════════════════════════════════
# Core geometry / map helpers
# ═══════════════════════════════════════════════════════════════════════════════

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


def _free_neighbors(r: int, c: int, game_map: np.ndarray, bomb_positions: set) -> int:
    total = 0
    for dr, dc in BLAST_DIRS:
        if _is_passable(r + dr, c + dc, game_map, bomb_positions):
            total += 1
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# Danger map computation (chain-reaction relaxation)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# BFS-based feature planes (v2 channels 12-14)
# ═══════════════════════════════════════════════════════════════════════════════

def _reachable_plane(
    start: tuple,
    game_map: np.ndarray,
    bomb_positions: set,
    danger: np.ndarray,
) -> np.ndarray:
    """Binary mask of cells reachable from start without entering danger."""
    reachable = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if not _is_passable(start[0], start[1], game_map, bomb_positions):
        return reachable
    q = deque([(start, 0)])
    visited = {start}
    while q:
        pos, dist = q.popleft()
        if danger[pos[0], pos[1]] <= dist + 1:
            continue
        reachable[pos[0], pos[1]] = 1.0
        for action in MOVE_ACTIONS:
            dr, dc = DIR_DELTA[action]
            nxt = (pos[0] + dr, pos[1] + dc)
            if nxt in visited:
                continue
            if not _is_passable(nxt[0], nxt[1], game_map, bomb_positions):
                continue
            visited.add(nxt)
            q.append((nxt, dist + 1))
    return reachable


def _dead_end_plane(
    game_map: np.ndarray,
    bomb_positions: set,
) -> np.ndarray:
    """Dead-end detection: 1.0 if 0-1 free neighbors, 0.45 if exactly 2."""
    dead_end = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for r in range(1, BOARD_SIZE - 1):
        for c in range(1, BOARD_SIZE - 1):
            if not _is_passable(r, c, game_map, bomb_positions):
                continue
            free = _free_neighbors(r, c, game_map, bomb_positions)
            if free <= 1:
                dead_end[r, c] = 1.0
            elif free == 2:
                dead_end[r, c] = 0.45
    return dead_end


def _frontier_plane(
    game_map: np.ndarray,
    bomb_positions: set,
) -> np.ndarray:
    """Cells adjacent to boxes: normalized by 3 (max 3 boxes adjacent)."""
    frontier = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for r in range(1, BOARD_SIZE - 1):
        for c in range(1, BOARD_SIZE - 1):
            if not _is_passable(r, c, game_map, bomb_positions):
                continue
            boxes = 0
            for dr, dc in BLAST_DIRS:
                nr, nc = r + dr, c + dc
                if _in_bounds(nr, nc) and int(game_map[nr, nc]) == TILE_BOX:
                    boxes += 1
            frontier[r, c] = min(1.0, boxes / 3.0)
    return frontier


def _center_plane() -> np.ndarray:
    """Manhattan-distance-based center bias (1.0 at center, decays outward)."""
    center = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    mid = BOARD_SIZE // 2
    max_dist = float((BOARD_SIZE - 1) * 2)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            center[r, c] = 1.0 - ((abs(r - mid) + abs(c - mid)) / max_dist)
    return center


# ═══════════════════════════════════════════════════════════════════════════════
# Escape / safety helpers (used by action mask v2)
# ═══════════════════════════════════════════════════════════════════════════════

def _bfs_nearest_safe(
    start: tuple,
    game_map: np.ndarray,
    bomb_positions: set,
    danger: np.ndarray,
    max_depth: int = MAX_BFS_ESCAPE,
) -> Optional[tuple]:
    """Find nearest permanently-safe cell. Returns (first_action, dist, pos) or None."""
    q = deque([(start, None, 0)])
    visited = {start}
    while q:
        pos, first, dist = q.popleft()
        if dist > 0 and danger[pos[0], pos[1]] == INF:
            return first, dist, pos
        if dist >= max_depth:
            continue
        for action in MOVE_ACTIONS:
            dr, dc = DIR_DELTA[action]
            nxt = (pos[0] + dr, pos[1] + dc)
            if nxt in visited:
                continue
            if not _is_passable(nxt[0], nxt[1], game_map, bomb_positions):
                continue
            if danger[nxt[0], nxt[1]] <= dist + 2:
                continue
            visited.add(nxt)
            q.append((nxt, action if first is None else first, dist + 1))
    return None


def can_reach_safe_from(
    start: tuple,
    game_map: np.ndarray,
    bomb_positions: set,
    danger: np.ndarray,
    max_depth: int = ACTION_ESCAPE_LOOKAHEAD,
) -> bool:
    """True if a permanently-safe cell is reachable from start within max_depth."""
    if danger[start[0], start[1]] == INF:
        return True
    return _bfs_nearest_safe(start, game_map, bomb_positions, danger, max_depth) is not None


def _reachable_safe_count(
    start: tuple,
    game_map: np.ndarray,
    bomb_positions: set,
    danger: np.ndarray,
    max_depth: int = 5,
) -> int:
    """Count safe cells reachable from start within max_depth."""
    q = deque([(start, 0)])
    visited = {start}
    count = 0
    while q:
        pos, dist = q.popleft()
        if danger[pos[0], pos[1]] > dist + 1:
            count += 1
        if dist >= max_depth:
            continue
        for action in MOVE_ACTIONS:
            dr, dc = DIR_DELTA[action]
            nxt = (pos[0] + dr, pos[1] + dc)
            if nxt in visited:
                continue
            if not _is_passable(nxt[0], nxt[1], game_map, bomb_positions):
                continue
            visited.add(nxt)
            q.append((nxt, dist + 1))
    return count


def _simulate_bomb_danger(
    obs: dict,
    agent_id: int,
    pos: Optional[tuple] = None,
) -> np.ndarray:
    """Compute danger map after hypothetically placing a bomb at agent or pos."""
    players = np.asarray(obs["players"], dtype=np.int32)
    game_map = np.asarray(obs["map"], dtype=np.int32)
    bombs = np.asarray(obs["bombs"], dtype=np.int32)

    r, c = int(players[agent_id][0]), int(players[agent_id][1])
    if pos is None:
        pos = (r, c)

    fake = np.array([[pos[0], pos[1], BOMB_TIMER, agent_id]], dtype=np.int32)
    if bombs.size == 0:
        new_bombs = fake
    else:
        if bombs.ndim == 1:
            bombs = bombs.reshape(1, -1)
        new_bombs = np.vstack([bombs, fake])

    return compute_danger_map(game_map, players, new_bombs)


def can_escape_after_bomb_v2(obs: dict, agent_id: int) -> bool:
    """Check if agent can survive placing a bomb at current position."""
    players = np.asarray(obs["players"], dtype=np.int32)
    game_map = np.asarray(obs["map"], dtype=np.int32)
    r, c = int(players[agent_id][0]), int(players[agent_id][1])
    bombs_left = int(players[agent_id][3])
    pos = (r, c)

    bset = _bomb_set(obs["bombs"])
    if bombs_left <= 0 or pos in bset:
        return False

    danger = _simulate_bomb_danger(obs, agent_id, pos)
    blocked = set(bset)
    blocked.add(pos)

    escape = _bfs_nearest_safe(pos, game_map, blocked, danger, MAX_BFS_ESCAPE)
    if escape is None:
        return False
    return _reachable_safe_count(escape[2], game_map, blocked, danger, 4) >= MIN_SAFE_CELLS_AFTER_BOMB


def _boxes_in_blast(pos: tuple, radius: int, game_map: np.ndarray) -> int:
    """Count boxes in blast radius from pos."""
    return sum(1 for r, c in _blast_cells(pos[0], pos[1], radius, game_map)
               if int(game_map[r, c]) == TILE_BOX)


def _enemies_in_blast(
    pos: tuple,
    radius: int,
    game_map: np.ndarray,
    players: np.ndarray,
    agent_id: int,
) -> list:
    """List enemy indices in blast radius from pos."""
    blast = _blast_cells(pos[0], pos[1], radius, game_map)
    hits = []
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1:
            continue
        if (int(p[0]), int(p[1])) in blast:
            hits.append(i)
    return hits


def enemy_escape_pressure(
    obs: dict,
    agent_id: int,
    bomb_pos: tuple,
) -> int:
    """Pressure exerted on enemies by a bomb at bomb_pos (0-1200)."""
    players = np.asarray(obs["players"], dtype=np.int32)
    game_map = np.asarray(obs["map"], dtype=np.int32)
    after = _simulate_bomb_danger(obs, agent_id, bomb_pos)
    my_radius = 1 + int(players[agent_id][4])
    blast = _blast_cells(bomb_pos[0], bomb_pos[1], my_radius, game_map)

    pressure = 0
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1:
            continue
        epos = (int(p[0]), int(p[1]))
        if epos in blast:
            pressure += 350
        escape = _bfs_nearest_safe(epos, game_map, _bomb_set(obs["bombs"]), after, 7)
        if escape is None:
            pressure += 500
        elif escape[1] >= 4:
            pressure += 180
    return min(pressure, 1200)


# ═══════════════════════════════════════════════════════════════════════════════
# Action mask (production-grade v2)
# ═══════════════════════════════════════════════════════════════════════════════

def get_action_mask(
    obs: dict,
    agent_id: int,
    danger: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Returns boolean mask of shape (6,) where True = legal action.

    Production-grade safety rules:
      - Dead agent: only STOP is legal.
      - Emergency escape: if current cell danger <= 4 and moves exist, STOP is illegal.
      - Movement: destination must be passable, danger > 1, and must have a path to
        permanent safety (can_reach_safe_from).
      - PLACE_BOMB: requires bombs_left > 0, not standing on bomb, can escape after
        placing, AND meaningful value (boxes > 0 OR enemy pressure >= 350).
    """
    mask = np.zeros(6, dtype=bool)
    players = np.asarray(obs["players"], dtype=np.int32)
    game_map = np.asarray(obs["map"], dtype=np.int32)
    bombs = np.asarray(obs["bombs"], dtype=np.int32)

    r, c, alive = int(players[agent_id][0]), int(players[agent_id][1]), int(players[agent_id][2])
    bombs_left = int(players[agent_id][3])
    bonus = int(players[agent_id][4])

    if not alive:
        mask[A_STOP] = True
        return mask

    if danger is None:
        danger = compute_danger_map(game_map, players, bombs)

    bset = _bomb_set(bombs)

    # STOP is legal by default (may be overridden by emergency escape)
    mask[A_STOP] = True

    # Movement actions
    move_candidates = []
    for action in MOVE_ACTIONS:
        dr, dc = DIR_DELTA[action]
        nr, nc = r + dr, c + dc
        if not _is_passable(nr, nc, game_map, bset):
            continue
        # Must not step into immediate explosion
        if danger[nr, nc] <= 1:
            continue
        # If current cell is in danger and destination is only marginally better
        if danger[nr, nc] <= 2 and not (danger[r, c] <= 1 and danger[nr, nc] > danger[r, c]):
            continue
        # Must be able to reach permanent safety from destination
        if not can_reach_safe_from((nr, nc), game_map, bset, danger):
            continue
        mask[action] = True
        move_candidates.append(action)

    # Emergency escape: force movement when in danger
    if danger[r, c] <= CURRENT_DANGER_ESCAPE_THRESHOLD and move_candidates:
        mask[A_STOP] = False

    # Bomb placement — meaningful gate
    if bombs_left > 0 and (r, c) not in bset and danger[r, c] == INF:
        if can_escape_after_bomb_v2(obs, agent_id):
            radius = 1 + bonus
            value = _boxes_in_blast((r, c), radius, game_map)
            value += len(_enemies_in_blast((r, c), radius, game_map, players, agent_id))
            if value > 0 or enemy_escape_pressure(obs, agent_id, (r, c)) >= 350:
                mask[A_BOMB] = True

    # Fallback
    if not np.any(mask):
        mask[A_STOP] = True

    return mask


# ═══════════════════════════════════════════════════════════════════════════════
# State encoders
# ═══════════════════════════════════════════════════════════════════════════════

def encode_observation(obs: dict, agent_id: int) -> tuple:
    """
    Legacy 7-channel encoder (v1). Kept for backward compatibility.

    Returns:
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

    channels = np.zeros((STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

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
    capacity = 0.2  # fixed proxy
    bombs_left_norm = float(players[agent_id][3]) / MAX_BOMB_CAPACITY if int(players[agent_id][2]) else 0.0
    step_norm = min(1.0, step / 500.0)

    scalars = np.array([radius_bonus, bombs_left_norm, capacity, step_norm], dtype=np.float32)

    state_tensor = torch.from_numpy(channels)
    scalar_tensor = torch.from_numpy(scalars)

    return state_tensor, scalar_tensor


def encode_observation_v2(obs: dict, agent_id: int) -> tuple:
    """
    Production 16-channel encoder (v2).

    Returns:
      - state_tensor: (16, 13, 13) float32 tensor
      - scalars:       (4,) float32 tensor

    Channel layout:
       0: Wall
       1: Box
       2: Self position
       3: Opponent positions
       4: Bomb timers (normalized: timer / 7)
       5: Danger zones (normalized: 1.0 - danger/7, 0=safe)
       6: Items (Radius=0.5, Capacity=1.0)
       7: Grass (walkable non-item tiles)
       8: Bomb owner self (bombs placed by training agent)
       9: Bomb owner enemy (bombs placed by opponents)
      10: Danger now (danger <= 1, immediate explosion)
      11: Danger future (normalized approaching danger, 0=safe)
      12: Reachable (BFS from agent position, binary)
      13: Dead end (0-1 free neighbors=1.0, 2=0.45)
      14: Frontier (cells adjacent to boxes, normalized /3)
      15: Center bias (Manhattan distance to center)
    """
    game_map = np.asarray(obs["map"], dtype=np.int32)
    players = np.asarray(obs["players"], dtype=np.int32)
    bombs_arr = np.asarray(obs["bombs"], dtype=np.int32)
    step = int(obs.get("step", obs.get("current_step", 0)) or 0)

    if bombs_arr.size > 0 and bombs_arr.ndim == 1:
        bombs_arr = bombs_arr.reshape(1, -1)

    channels = np.zeros((STATE_CHANNELS_V2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Channel 0: Walls
    channels[0] = (game_map == TILE_WALL).astype(np.float32)

    # Channel 1: Boxes
    channels[1] = (game_map == TILE_BOX).astype(np.float32)

    # Channel 2: Self position
    my_r, my_c = int(players[agent_id][0]), int(players[agent_id][1])
    is_alive = bool(int(players[agent_id][2]))
    if is_alive and 0 <= my_r < BOARD_SIZE and 0 <= my_c < BOARD_SIZE:
        channels[2, my_r, my_c] = 1.0

    # Channel 3: Opponent positions
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1:
            continue
        er, ec = int(p[0]), int(p[1])
        if 0 <= er < BOARD_SIZE and 0 <= ec < BOARD_SIZE:
            channels[3, er, ec] = 1.0

    # Channel 4: Bomb timers (normalized)
    # Channel 8: Bomb owner self
    # Channel 9: Bomb owner enemy
    for bomb in bombs_arr:
        br, bc, timer, owner = [int(x) for x in bomb[:4]]
        if 0 <= br < BOARD_SIZE and 0 <= bc < BOARD_SIZE and timer > 0:
            channels[4, br, bc] = max(0.0, min(float(timer) / 7.0, 1.0))
            if owner == agent_id:
                channels[8, br, bc] = 1.0
            else:
                channels[9, br, bc] = 1.0

    # Channel 5: Danger zones (smoothed)
    danger = compute_danger_map(game_map, players, bombs_arr)
    for r_idx in range(BOARD_SIZE):
        for c_idx in range(BOARD_SIZE):
            d = danger[r_idx, c_idx]
            if d == INF:
                channels[5, r_idx, c_idx] = 0.0
            else:
                channels[5, r_idx, c_idx] = max(0.0, 1.0 - d / BOMB_TIMER)

    # Channel 6: Items
    channels[6] = np.where(game_map == TILE_RADIUS, 0.5,
                  np.where(game_map == TILE_CAPACITY, 1.0, 0.0)).astype(np.float32)

    # Channel 7: Grass (walkable, non-item tiles)
    channels[7] = (game_map == TILE_GRASS).astype(np.float32)

    # Channel 10: Danger now (cell explodes this turn: danger <= 1)
    channels[10] = ((danger > 0) & (danger <= 1)).astype(np.float32)

    # Channel 11: Danger future (normalized approaching danger for cells with danger < INF)
    channels[11] = np.where(
        danger < INF,
        np.clip((8.0 - danger.astype(np.float32)) / 7.0, 0.0, 1.0),
        0.0,
    ).astype(np.float32)

    # BFS planes require bomb set
    bset = _bomb_set(bombs_arr)
    start = (my_r, my_c) if is_alive else (1, 1)

    # Channel 12: Reachable
    channels[12] = _reachable_plane(start, game_map, bset, danger)

    # Channel 13: Dead end
    channels[13] = _dead_end_plane(game_map, bset)

    # Channel 14: Frontier (cells adjacent to boxes)
    channels[14] = _frontier_plane(game_map, bset)

    # Channel 15: Center bias multiplied by reachable area
    center = _center_plane()
    channels[15] = center * channels[12]  # center * reachable

    # Scalar features: [radius_bonus, bombs_left_norm, capacity_proxy, step_norm]
    radius_bonus = float(players[agent_id][4]) / (MAX_BOMB_RADIUS - 1) if is_alive else 0.0
    bombs_left_norm = float(players[agent_id][3]) / MAX_BOMB_CAPACITY if is_alive else 0.0
    capacity = 0.2
    step_norm = min(1.0, step / 500.0)

    scalars = np.array([radius_bonus, bombs_left_norm, capacity, step_norm], dtype=np.float32)

    state_tensor = torch.from_numpy(channels)
    scalar_tensor = torch.from_numpy(scalars)

    return state_tensor, scalar_tensor


def get_legal_action_mask_tensor(obs: dict, agent_id: int) -> torch.Tensor:
    """Convenience: returns (6,) bool tensor for action masking."""
    mask_np = get_action_mask(obs, agent_id)
    return torch.from_numpy(mask_np)


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal frame-stacking (Phase 4A)
# ═══════════════════════════════════════════════════════════════════════════════

class FrameBuffer:
    """Fixed-size FIFO buffer for frame stacking."""

    def __init__(self, frame_stack: int = FRAME_STACK):
        self.frame_stack = max(1, int(frame_stack))
        self.frames: deque = deque(maxlen=self.frame_stack)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self.frames.clear()
        frame = np.asarray(frame, dtype=np.float32)
        for _ in range(self.frame_stack):
            self.frames.append(frame.copy())
        return self.stacked()

    def append(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if not self.frames:
            return self.reset(frame)
        self.frames.append(frame.copy())
        return self.stacked()

    def stacked(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)


class TemporalStateProcessor:
    """
    Wraps encode_observation_v2 with 4-frame temporal stacking.

    Produces (64, 13, 13) tensors by stacking 4 consecutive 16-channel frames.
    Automatically resets on episode boundaries (agent respawn detection).
    """

    def __init__(self, frame_stack: int = FRAME_STACK):
        self.frame_stack = frame_stack
        self._buffer: Optional[FrameBuffer] = None
        self._last_pos: Optional[tuple] = None

    def reset(self):
        self._buffer = None
        self._last_pos = None

    def process(self, obs: dict, agent_id: int) -> tuple:
        """
        Encode current observation with temporal history.

        Returns:
            state_tensor: (frame_stack * STATE_CHANNELS_V2, 13, 13)
            scalar_tensor: (4,)
        """
        state_tensor, scalar_tensor = encode_observation_v2(obs, agent_id)
        frame_np = state_tensor.numpy()  # (16, 13, 13)

        # Detect respawn (episode reset) by checking if position jumped
        players = np.asarray(obs["players"], dtype=np.int32)
        is_alive = bool(int(players[agent_id][2]))
        current_pos = (int(players[agent_id][0]), int(players[agent_id][1])) if is_alive else None

        if self._buffer is None or self._last_pos is None:
            self._buffer = FrameBuffer(self.frame_stack)
            stacked = self._buffer.reset(frame_np)
        elif not is_alive:
            # Dead agent: keep last frame but mark
            stacked = self._buffer.append(frame_np)
        else:
            # Check for abrupt position change (respawn)
            if self._last_pos is not None:
                dr = abs(current_pos[0] - self._last_pos[0])
                dc = abs(current_pos[1] - self._last_pos[1])
                if dr > 2 or dc > 2:
                    # Likely respawn or new episode
                    self._buffer.reset(frame_np)
            stacked = self._buffer.append(frame_np)

        self._last_pos = current_pos
        stacked_tensor = torch.from_numpy(stacked)
        return stacked_tensor, scalar_tensor
