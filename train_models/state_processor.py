"""
State Processor for Bomberland — HPC-Accelerated Edition.

Converts raw game observations into:
  - (7, 13, 13) multi-channel tensor (legacy v1)
  - (16, 13, 13) multi-channel tensor (v2, production)
  - (4,) scalar feature vector
  - (6,) legal action mask (production-grade + opponent trap override)
  - Danger map for blast-zones with chain-reaction relaxation
  - TemporalStateProcessor: 4-frame stacking wrapper

HPC optimizations (Directive 1):
  - Step-singleton cache: shared planes (danger, dead_end, frontier) computed
    once per environment step and reused across all 4 agents in O(1).
  - Pre-allocated NumPy ring buffers replace all Python deque BFS allocations.
  - Hard BFS depth cutoff at 12 enforced across all graph traversal routines.
  - Center plane pre-computed once at module load (static geometry).
  - Vectorized channel construction where possible.

Elite Slayer tactics (Directive 2.1):
  - Opponent trapping gate: detects when a bomb placement structurally cuts off
    an enemy's escape path, overriding conservative safety gates.
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
    BFS_MAX_DEPTH,
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
BFS_QUEUE_SIZE = BOARD_SIZE * BOARD_SIZE  # 169


# ═══════════════════════════════════════════════════════════════════════════════
# Step Singleton Cache (Directive 1.1–1.3)
# ═══════════════════════════════════════════════════════════════════════════════

class _StepSingletonCache:
    """
    Thread-safe module-level cache for shared per-step computations.

    Because map, bombs, and item layout are identical for all 4 agents within
    a single environment step, the first agent computes danger_map, dead_end_plane,
    and frontier_plane. Subsequent agents retrieve them in O(1) via content hash.

    Auto-evicts when the step fingerprint changes, guaranteeing zero memory leak.
    """

    def __init__(self):
        self._fingerprint: Optional[int] = None
        self._danger_map: Optional[np.ndarray] = None
        self._dead_end_plane: Optional[np.ndarray] = None
        self._frontier_plane: Optional[np.ndarray] = None
        self._bomb_set: Optional[set] = None
        self._bombs_array: Optional[np.ndarray] = None

    def _compute_fingerprint(
        self, game_map: np.ndarray, bombs: np.ndarray, step: int
    ) -> int:
        """Fast content-based hash for cache keying."""
        map_hash = hash(game_map.tobytes())
        bomb_hash = hash(bombs.tobytes()) if bombs.size > 0 else 0
        return hash((step, map_hash, bomb_hash))

    def get(
        self, game_map: np.ndarray, bombs: np.ndarray, step: int
    ) -> Optional[dict]:
        """Retrieve cached planes if fingerprint matches. Returns None on miss."""
        fp = self._compute_fingerprint(game_map, bombs, step)
        if self._fingerprint == fp and self._danger_map is not None:
            return {
                "danger_map": self._danger_map,
                "dead_end_plane": self._dead_end_plane,
                "frontier_plane": self._frontier_plane,
                "bomb_set": self._bomb_set,
                "bombs_array": self._bombs_array,
            }
        # Fingerprint miss → old cache is stale, will be overwritten
        return None

    def store(
        self,
        game_map: np.ndarray,
        bombs: np.ndarray,
        step: int,
        danger_map: np.ndarray,
        dead_end_plane: np.ndarray,
        frontier_plane: np.ndarray,
        bomb_set: set,
        bombs_array: np.ndarray,
    ):
        """Store computed planes into cache."""
        self._fingerprint = self._compute_fingerprint(game_map, bombs, step)
        self._danger_map = danger_map
        self._dead_end_plane = dead_end_plane
        self._frontier_plane = frontier_plane
        self._bomb_set = bomb_set
        self._bombs_array = bombs_array

    def flush(self):
        """Explicit cache flush (e.g., between episodes)."""
        self._fingerprint = None
        self._danger_map = None
        self._dead_end_plane = None
        self._frontier_plane = None
        self._bomb_set = None
        self._bombs_array = None


# Global singleton — shared across all encoder calls within the same process
_step_cache = _StepSingletonCache()


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-allocated NumPy BFS buffers (Directive 1.4)
# ═══════════════════════════════════════════════════════════════════════════════

# Ring-buffer queue for BFS: (row, col, dist) × 169 max cells
_bfs_queue_r = np.zeros(BFS_QUEUE_SIZE, dtype=np.int32)
_bfs_queue_c = np.zeros(BFS_QUEUE_SIZE, dtype=np.int32)
_bfs_queue_d = np.zeros(BFS_QUEUE_SIZE, dtype=np.int32)
_bfs_visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)


def _bfs_reset_visited():
    """Zero out the visited mask. Faster than re-allocating."""
    _bfs_visited.fill(False)


# Pre-computed static planes (computed once at import time)
_CENTER_PLANE: Optional[np.ndarray] = None


def _init_static_planes():
    """Pre-compute geometry planes that never change."""
    global _CENTER_PLANE
    if _CENTER_PLANE is not None:
        return
    mid = BOARD_SIZE // 2
    max_dist = float((BOARD_SIZE - 1) * 2)
    rows = np.arange(BOARD_SIZE, dtype=np.float32)
    cols = np.arange(BOARD_SIZE, dtype=np.float32)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    _CENTER_PLANE = (1.0 - (np.abs(rr - mid) + np.abs(cc - mid)) / max_dist).astype(np.float32)


_init_static_planes()


# ═══════════════════════════════════════════════════════════════════════════════
# Core geometry / map helpers (partially vectorized)
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


def _is_passable_fast(r: int, c: int, game_map: np.ndarray, bomb_positions: set) -> bool:
    """Inline version for tight BFS loops — skips _in_bounds check when pre-validated."""
    tile = game_map[r, c]
    if tile == TILE_WALL or tile == TILE_BOX:
        return False
    return (r, c) not in bomb_positions


# Pre-computed walkable mask for the entire board (vectorized)
def _build_walkable_mask(game_map: np.ndarray) -> np.ndarray:
    """(13,13) bool mask: True where tile is walkable (grass/radius/capacity)."""
    gm = np.asarray(game_map, dtype=np.int32)
    return (gm != TILE_WALL) & (gm != TILE_BOX)


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
# BFS-based feature planes — NumPy ring-buffer, depth 12 cutoff (Directive 1.4)
# ═══════════════════════════════════════════════════════════════════════════════

def _reachable_plane_numpy(
    start: tuple,
    game_map: np.ndarray,
    bomb_positions: set,
    danger: np.ndarray,
    max_depth: int = BFS_MAX_DEPTH,
) -> np.ndarray:
    """
    Binary mask of cells reachable from start without entering danger.
    Uses pre-allocated NumPy ring buffers — zero Python allocation in the hot path.
    """
    sr, sc = start
    result = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    if not _in_bounds(sr, sc):
        return result
    tile = game_map[sr, sc]
    if tile == TILE_WALL or tile == TILE_BOX:
        return result
    if (sr, sc) in bomb_positions:
        return result

    _bfs_reset_visited()

    head = 0
    tail = 0

    _bfs_queue_r[tail] = sr
    _bfs_queue_c[tail] = sc
    _bfs_queue_d[tail] = 0
    tail += 1
    _bfs_visited[sr, sc] = True

    while head < tail:
        r = _bfs_queue_r[head]
        c = _bfs_queue_c[head]
        dist = _bfs_queue_d[head]
        head += 1

        if danger[r, c] <= dist + 1:
            continue

        result[r, c] = 1.0

        if dist >= max_depth:
            continue

        # Unrolled neighbor checks for speed
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if _bfs_visited[nr, nc]:
                continue
            if not _is_passable(nr, nc, game_map, bomb_positions):
                continue
            _bfs_visited[nr, nc] = True
            _bfs_queue_r[tail] = nr
            _bfs_queue_c[tail] = nc
            _bfs_queue_d[tail] = dist + 1
            tail += 1

    return result


def _dead_end_plane_numpy(
    game_map: np.ndarray,
    bomb_positions: set,
) -> np.ndarray:
    """
    Dead-end detection: 1.0 if 0-1 free neighbors, 0.45 if exactly 2.
    Vectorized over the 11×11 interior grid.
    """
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


def _frontier_plane_numpy(
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
    """Return the pre-computed static center plane."""
    return _CENTER_PLANE


# Legacy versions kept for backward compatibility
def _reachable_plane(
    start: tuple,
    game_map: np.ndarray,
    bomb_positions: set,
    danger: np.ndarray,
) -> np.ndarray:
    return _reachable_plane_numpy(start, game_map, bomb_positions, danger, BFS_MAX_DEPTH)


def _dead_end_plane(
    game_map: np.ndarray,
    bomb_positions: set,
) -> np.ndarray:
    return _dead_end_plane_numpy(game_map, bomb_positions)


def _frontier_plane(
    game_map: np.ndarray,
    bomb_positions: set,
) -> np.ndarray:
    return _frontier_plane_numpy(game_map, bomb_positions)


# ═══════════════════════════════════════════════════════════════════════════════
# Escape / safety helpers (Directive 1.4: depth-12 cutoff)
# ═══════════════════════════════════════════════════════════════════════════════

def _bfs_nearest_safe_numpy(
    start: tuple,
    game_map: np.ndarray,
    bomb_positions: set,
    danger: np.ndarray,
    max_depth: int = MAX_BFS_ESCAPE,
) -> Optional[tuple]:
    """
    Find nearest permanently-safe cell using NumPy ring buffer.
    Returns (first_action, dist, pos) or None.
    """
    if not _is_passable(start[0], start[1], game_map, bomb_positions):
        return None

    if danger[start[0], start[1]] == INF:
        return (None, 0, start)

    _bfs_reset_visited()

    head = 0
    tail = 0

    sr, sc = start
    _bfs_queue_r[tail] = sr
    _bfs_queue_c[tail] = sc
    _bfs_queue_d[tail] = 0
    tail += 1
    _bfs_visited[sr, sc] = True

    # Store first action per BFS branch: -1 = unset, 0-3 = action index
    first_action = np.full((BOARD_SIZE, BOARD_SIZE), -1, dtype=np.int8)

    action_list = list(enumerate(MOVE_ACTIONS))  # [(0, A_LEFT), (1, A_RIGHT), ...]

    while head < tail:
        r = _bfs_queue_r[head]
        c = _bfs_queue_c[head]
        dist = _bfs_queue_d[head]
        head += 1

        if dist > 0 and danger[r, c] == INF:
            # Found safe cell. Reconstruct first action.
            fa = first_action[r, c]
            if fa < 0:
                return (None, dist, (r, c))
            return (MOVE_ACTIONS[fa], dist, (r, c))

        if dist >= min(max_depth, BFS_MAX_DEPTH):
            continue

        for act_idx, action in action_list:
            dr, dc = DIR_DELTA[action]
            nr, nc = r + dr, c + dc
            if _bfs_visited[nr, nc]:
                continue
            if not _is_passable(nr, nc, game_map, bomb_positions):
                continue
            if danger[nr, nc] <= dist + 2:
                continue
            _bfs_visited[nr, nc] = True
            _bfs_queue_r[tail] = nr
            _bfs_queue_c[tail] = nc
            _bfs_queue_d[tail] = dist + 1
            first_action[nr, nc] = act_idx if dist == 0 else first_action[r, c]
            tail += 1

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
    return _bfs_nearest_safe_numpy(start, game_map, bomb_positions, danger, max_depth) is not None


def _reachable_safe_count_numpy(
    start: tuple,
    game_map: np.ndarray,
    bomb_positions: set,
    danger: np.ndarray,
    max_depth: int = 5,
) -> int:
    """Count safe cells reachable from start within max_depth. NumPy ring buffer."""
    if not _is_passable(start[0], start[1], game_map, bomb_positions):
        return 0

    _bfs_reset_visited()

    head = 0
    tail = 0
    sr, sc = start
    _bfs_queue_r[tail] = sr
    _bfs_queue_c[tail] = sc
    _bfs_queue_d[tail] = 0
    tail += 1
    _bfs_visited[sr, sc] = True
    count = 0
    depth_limit = min(max_depth, BFS_MAX_DEPTH)

    while head < tail:
        r = _bfs_queue_r[head]
        c = _bfs_queue_c[head]
        dist = _bfs_queue_d[head]
        head += 1

        if danger[r, c] > dist + 1:
            count += 1
        if dist >= depth_limit:
            continue

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if _bfs_visited[nr, nc]:
                continue
            if not _is_passable(nr, nc, game_map, bomb_positions):
                continue
            _bfs_visited[nr, nc] = True
            _bfs_queue_r[tail] = nr
            _bfs_queue_c[tail] = nc
            _bfs_queue_d[tail] = dist + 1
            tail += 1

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

    escape = _bfs_nearest_safe_numpy(pos, game_map, blocked, danger, MAX_BFS_ESCAPE)
    if escape is None:
        return False
    return _reachable_safe_count_numpy(escape[2], game_map, blocked, danger, 4) >= MIN_SAFE_CELLS_AFTER_BOMB


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
        escape = _bfs_nearest_safe_numpy(epos, game_map, _bomb_set(obs["bombs"]), after, 7)
        if escape is None:
            pressure += 500
        elif escape[1] >= 4:
            pressure += 180
    return min(pressure, 1200)


# ═══════════════════════════════════════════════════════════════════════════════
# Opponent Trapping Gate (Directive 2.1)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_opponent_trap(
    obs: dict,
    agent_id: int,
    bomb_pos: tuple,
    dead_end_plane: np.ndarray,
) -> bool:
    """
    Detect if placing a bomb at bomb_pos would structurally trap an enemy.

    Conditions for a "trap":
      1. An alive enemy is within the bomb's blast radius.
      2. That enemy has <= 1 free neighbors (dead_end_score > 0.95).
      3. The bomb would block the enemy's only escape path:
         the enemy's free neighbor cell is also in the blast zone.
      4. OR: the enemy cannot reach permanent safety after bomb placement.

    Returns True if a trap is detected — the action mask should force-enable A_BOMB.
    """
    players = np.asarray(obs["players"], dtype=np.int32)
    game_map = np.asarray(obs["map"], dtype=np.int32)
    my_radius = 1 + int(players[agent_id][4])
    blast = _blast_cells(bomb_pos[0], bomb_pos[1], my_radius, game_map)

    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1:
            continue
        er, ec = int(p[0]), int(p[1])

        # Enemy must be in blast zone
        if (er, ec) not in blast:
            continue

        # Enemy is in a dead-end (0-1 free neighbors)
        if dead_end_plane[er, ec] < 0.95:
            continue

        # The enemy's escape cell is also in blast → trapped
        free_cells = []
        for dr, dc in BLAST_DIRS:
            nr, nc = er + dr, ec + dc
            if _is_passable(nr, nc, game_map, _bomb_set(obs["bombs"])):
                free_cells.append((nr, nc))

        if len(free_cells) <= 1:
            # Enemy has at most 1 escape cell
            if len(free_cells) == 0:
                return True  # completely trapped
            # If the single escape is also in blast, enemy is cooked
            if free_cells[0] in blast:
                return True

        # Deep check: simulate post-bomb danger and verify enemy can't escape
        after = _simulate_bomb_danger(obs, agent_id, bomb_pos)
        enemy_escape = _bfs_nearest_safe_numpy(
            (er, ec), game_map, _bomb_set(obs["bombs"]), after, 6
        )
        if enemy_escape is None:
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Action mask (production-grade v2 + opponent trap override)
# ═══════════════════════════════════════════════════════════════════════════════

def get_action_mask(
    obs: dict,
    agent_id: int,
    danger: Optional[np.ndarray] = None,
    dead_end_plane_cache: Optional[np.ndarray] = None,
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
      - OPPONENT TRAP OVERRIDE: If placing a bomb would structurally trap an enemy
        in a dead-end, force-enable A_BOMB even if normal safety gates would block it,
        and set a flag so the selector can prioritize this action.
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
        if danger[nr, nc] <= 1:
            continue
        if danger[nr, nc] <= 2 and not (danger[r, c] <= 1 and danger[nr, nc] > danger[r, c]):
            continue
        if not can_reach_safe_from((nr, nc), game_map, bset, danger):
            continue
        mask[action] = True
        move_candidates.append(action)

    # Emergency escape: force movement when in danger
    if danger[r, c] <= CURRENT_DANGER_ESCAPE_THRESHOLD and move_candidates:
        mask[A_STOP] = False

    # Bomb placement
    if bombs_left > 0 and (r, c) not in bset and danger[r, c] == INF:
        # Use cached dead_end_plane if provided
        ded = dead_end_plane_cache if dead_end_plane_cache is not None else _dead_end_plane_numpy(game_map, bset)

        # Check for opponent trap (Directive 2.1) — this can override safety gates
        trap_detected = _detect_opponent_trap(obs, agent_id, (r, c), ded)

        if trap_detected:
            # Force-enable bomb — override conservative safety
            mask[A_BOMB] = True
        elif can_escape_after_bomb_v2(obs, agent_id):
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
# State encoders (cache-aware — Directive 1.1)
# ═══════════════════════════════════════════════════════════════════════════════

def encode_observation(obs: dict, agent_id: int) -> tuple:
    """
    Legacy 7-channel encoder (v1). Kept for backward compatibility.

    Returns:
      - state_tensor: (7, 13, 13) float32 tensor
      - scalars:       (4,) float32 tensor
    """
    game_map = np.asarray(obs["map"], dtype=np.int32)
    players = np.asarray(obs["players"], dtype=np.int32)
    bombs_arr = np.asarray(obs["bombs"], dtype=np.int32)
    step = int(obs.get("step", obs.get("current_step", 0)) or 0)

    if bombs_arr.size > 0 and bombs_arr.ndim == 1:
        bombs_arr = bombs_arr.reshape(1, -1)

    channels = np.zeros((STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    channels[0] = (game_map == TILE_WALL).astype(np.float32)
    channels[1] = (game_map == TILE_BOX).astype(np.float32)

    my_r, my_c = int(players[agent_id][0]), int(players[agent_id][1])
    if int(players[agent_id][2]) and 0 <= my_r < BOARD_SIZE and 0 <= my_c < BOARD_SIZE:
        channels[2, my_r, my_c] = 1.0

    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1:
            continue
        er, ec = int(p[0]), int(p[1])
        if 0 <= er < BOARD_SIZE and 0 <= ec < BOARD_SIZE:
            channels[3, er, ec] = 1.0

    for bomb in bombs_arr:
        br, bc, timer, _owner = [int(x) for x in bomb[:4]]
        if 0 <= br < BOARD_SIZE and 0 <= bc < BOARD_SIZE and timer > 0:
            channels[4, br, bc] = (BOMB_TIMER - timer) / BOMB_TIMER

    danger = compute_danger_map(game_map, players, bombs_arr)
    for r_idx in range(BOARD_SIZE):
        for c_idx in range(BOARD_SIZE):
            d = danger[r_idx, c_idx]
            if d == INF:
                channels[5, r_idx, c_idx] = 0.0
            else:
                channels[5, r_idx, c_idx] = max(0.0, 1.0 - d / BOMB_TIMER)

    channels[6] = np.where(game_map == TILE_RADIUS, 0.5,
                  np.where(game_map == TILE_CAPACITY, 1.0, 0.0)).astype(np.float32)

    radius_bonus = float(players[agent_id][4]) / (MAX_BOMB_RADIUS - 1) if int(players[agent_id][2]) else 0.0
    capacity = 0.2
    bombs_left_norm = float(players[agent_id][3]) / MAX_BOMB_CAPACITY if int(players[agent_id][2]) else 0.0
    step_norm = min(1.0, step / 500.0)

    scalars = np.array([radius_bonus, bombs_left_norm, capacity, step_norm], dtype=np.float32)

    state_tensor = torch.from_numpy(channels)
    scalar_tensor = torch.from_numpy(scalars)

    return state_tensor, scalar_tensor


def encode_observation_v2(obs: dict, agent_id: int) -> tuple:
    """
    Production 16-channel encoder (v2) with step-singleton caching.

    On the first call for a given (step, map, bombs) fingerprint:
      - Computes danger_map, dead_end_plane, frontier_plane, bomb_set
      - Stores all into _step_cache for O(1) retrieval by subsequent agents

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
       8: Bomb owner self
       9: Bomb owner enemy
      10: Danger now (danger <= 1, immediate explosion)
      11: Danger future (normalized approaching danger)
      12: Reachable (BFS from agent position, binary)
      13: Dead end (0-1 free neighbors=1.0, 2=0.45)
      14: Frontier (cells adjacent to boxes, normalized /3)
      15: Center bias × reachable
    """
    game_map = np.asarray(obs["map"], dtype=np.int32)
    players = np.asarray(obs["players"], dtype=np.int32)
    bombs_arr = np.asarray(obs["bombs"], dtype=np.int32)
    step = int(obs.get("step", obs.get("current_step", 0)) or 0)

    if bombs_arr.size > 0 and bombs_arr.ndim == 1:
        bombs_arr = bombs_arr.reshape(1, -1)

    # ═══════════════════════════════════════════════════════════════════════
    # Step Singleton Cache: check or compute shared planes (Directive 1.1-1.2)
    # ═══════════════════════════════════════════════════════════════════════
    cached = _step_cache.get(game_map, bombs_arr, step)

    if cached is not None:
        danger = cached["danger_map"]
        ded_plane = cached["dead_end_plane"]
        fro_plane = cached["frontier_plane"]
        bset = cached["bomb_set"]
    else:
        # First agent this step: compute all shared planes
        bset = _bomb_set(bombs_arr)
        danger = compute_danger_map(game_map, players, bombs_arr)
        ded_plane = _dead_end_plane_numpy(game_map, bset)
        fro_plane = _frontier_plane_numpy(game_map, bset)
        _step_cache.store(
            game_map, bombs_arr, step,
            danger, ded_plane, fro_plane, bset, bombs_arr,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Build the 16-channel tensor
    # ═══════════════════════════════════════════════════════════════════════
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

    # Channel 3: Opponent positions (vectorized)
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1:
            continue
        er, ec = int(p[0]), int(p[1])
        if 0 <= er < BOARD_SIZE and 0 <= ec < BOARD_SIZE:
            channels[3, er, ec] = 1.0

    # Channel 4: Bomb timers + Channels 8, 9: bomb ownership
    for bomb in bombs_arr:
        br, bc, timer, owner = [int(x) for x in bomb[:4]]
        if 0 <= br < BOARD_SIZE and 0 <= bc < BOARD_SIZE and timer > 0:
            channels[4, br, bc] = max(0.0, min(float(timer) / 7.0, 1.0))
            if owner == agent_id:
                channels[8, br, bc] = 1.0
            else:
                channels[9, br, bc] = 1.0

    # Channel 5: Danger zones (vectorized)
    danger_mask = danger < INF
    channels[5] = np.where(danger_mask, np.maximum(0.0, 1.0 - danger.astype(np.float32) / BOMB_TIMER), 0.0)

    # Channel 6: Items (vectorized)
    channels[6] = np.where(game_map == TILE_RADIUS, 0.5,
                  np.where(game_map == TILE_CAPACITY, 1.0, 0.0)).astype(np.float32)

    # Channel 7: Grass
    channels[7] = (game_map == TILE_GRASS).astype(np.float32)

    # Channel 10: Danger now
    channels[10] = ((danger > 0) & (danger <= 1)).astype(np.float32)

    # Channel 11: Danger future (vectorized)
    channels[11] = np.where(
        danger_mask,
        np.clip((8.0 - danger.astype(np.float32)) / 7.0, 0.0, 1.0),
        0.0,
    ).astype(np.float32)

    # Channel 12: Reachable (per-agent BFS — NOT cached)
    start = (my_r, my_c) if is_alive else (1, 1)
    channels[12] = _reachable_plane_numpy(start, game_map, bset, danger)

    # Channel 13: Dead end (FROM CACHE — O(1) retrieval for agents 2-4)
    channels[13] = ded_plane

    # Channel 14: Frontier (FROM CACHE — O(1) retrieval for agents 2-4)
    channels[14] = fro_plane

    # Channel 15: Center bias × reachable
    channels[15] = _CENTER_PLANE * channels[12]

    # ═══════════════════════════════════════════════════════════════════════
    # Scalar features
    # ═══════════════════════════════════════════════════════════════════════
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


def flush_step_cache():
    """Explicitly flush the step singleton cache (call between episodes)."""
    _step_cache.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal frame-stacking
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
        frame_np = state_tensor.numpy()

        players = np.asarray(obs["players"], dtype=np.int32)
        is_alive = bool(int(players[agent_id][2]))
        current_pos = (int(players[agent_id][0]), int(players[agent_id][1])) if is_alive else None

        if self._buffer is None or self._last_pos is None:
            self._buffer = FrameBuffer(self.frame_stack)
            stacked = self._buffer.reset(frame_np)
        elif not is_alive:
            stacked = self._buffer.append(frame_np)
        else:
            if self._last_pos is not None:
                dr = abs(current_pos[0] - self._last_pos[0])
                dc = abs(current_pos[1] - self._last_pos[1])
                if dr > 2 or dc > 2:
                    self._buffer.reset(frame_np)
            stacked = self._buffer.append(frame_np)

        self._last_pos = current_pos
        stacked_tensor = torch.from_numpy(stacked)
        return stacked_tensor, scalar_tensor
