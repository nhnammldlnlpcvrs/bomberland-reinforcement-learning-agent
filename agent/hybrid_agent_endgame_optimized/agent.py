"""
Hybrid Bomberland Agent: Rule-based + BFS + Heuristic Scoring + optional
model-assisted movement tiebreaker.

The rule policy remains primary.  The optional model is disabled by default
and can only rerank already-safe movement actions after rule scoring.
Designed to run well under 100ms per act() call.

Architecture:
  1. State Parser       — extract self, enemies, bombs
  2. Danger Map         — compute threat grid with chain-reaction relaxation
  3. BFS Pathfinding    — safe-path search, escape verification
  4. Heuristic Scorer   — rank all 6 actions, pick best
"""

import os
import time
from pathlib import Path

import numpy as np
from collections import deque

# ==============================================================================
# Constants
# ==============================================================================

BOARD_SIZE = 13
INF = 9999
BOMB_TIMER = 7

MAX_BFS_SAFE   = 12
MAX_BFS_ESCAPE = 8
MAX_BFS_ITEM   = 10
MAX_BFS_TARGET = 10
MAX_BFS_ENEMY  = 8
MAX_ENEMY_ESCAPE_SIM = 7
MAX_FUTURE_SURVIVABILITY = 5
MAX_RADIUS     = 5

TILE_GRASS    = 0
TILE_WALL     = 1
TILE_BOX      = 2
TILE_RADIUS   = 3
TILE_CAPACITY = 4

A_STOP = 0
A_LEFT = 1
A_RIGHT = 2
A_UP = 3
A_DOWN = 4
A_BOMB = 5

# (drow, dcol) — matches engine convention:
#   LEFT=1 → row-1   RIGHT=2 → row+1
#   UP=3   → col-1   DOWN=4  → col+1
DIRS = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}

BLAST_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

MOVE_ACTIONS = [A_LEFT, A_RIGHT, A_UP, A_DOWN]
ALL_ACTIONS  = [A_STOP, A_LEFT, A_RIGHT, A_UP, A_DOWN, A_BOMB]

BOMB_HYSTERESIS_THRESHOLD = 80
BOMB_HYSTERESIS_TUNED_THRESHOLD = BOMB_HYSTERESIS_THRESHOLD + 50
MEANINGFUL_ESCAPE_PRESSURE = 500
ESCAPE_PRESSURE_SCORE_MULTIPLIER = 0.25

FUTURE_SAFE_CELL_WEIGHT = 18
FUTURE_BRANCH_WEIGHT = 35
FUTURE_DEPTH_WEIGHT = 20
FUTURE_DEAD_END_PENALTY = 25
FUTURE_SCORE_MULTIPLIER = 1.0
OPEN_ZONE_BONUS = 60
CORRIDOR_PENALTY = 40
DEAD_END_PENALTY = 120
NEAR_ENEMY_PRESSURE_2 = 120
NEAR_ENEMY_PRESSURE_4 = 60

MAX_POSITION_HISTORY = 12
EXPANSION_BFS_DEPTH = 5
EXPANSION_SCORE_MULTIPLIER = 0.45
LOOP_EXPANSION_SCORE_MULTIPLIER = 0.75
TERRITORY_PRESSURE_MULTIPLIER = 0.5
ACTION_ADVANTAGE_WEIGHT = 0.15

HYBRID_MODEL_ENABLE = os.environ.get("HYBRID_MODEL_ENABLE", "true").lower() in {
    "1", "true", "yes", "on"
}
HYBRID_MODEL_CHECKPOINT = os.environ.get(
    "HYBRID_MODEL_CHECKPOINT",
    str(Path(__file__).resolve().parent / "ml" / "checkpoints" / "action_ranker_bomb_fixed.pt"),
)
HYBRID_MODEL_MAX_LATENCY_MS = float(os.environ.get("HYBRID_MODEL_MAX_LATENCY_MS", "5"))
HYBRID_MODEL_ONLY_SAFE_ACTIONS = os.environ.get(
    "HYBRID_MODEL_ONLY_SAFE_ACTIONS", "true"
).lower() in {"1", "true", "yes", "on"}
HYBRID_MODEL_CONF_MARGIN = float(os.environ.get("HYBRID_MODEL_CONF_MARGIN", "0.18"))
HYBRID_MODEL_SKIP_AFTER_BOMB_STEPS = int(os.environ.get("HYBRID_MODEL_SKIP_AFTER_BOMB_STEPS", "10"))
HYBRID_MODEL_REQUIRE_SAFE_AREA_NONDECREASE = os.environ.get(
    "HYBRID_MODEL_REQUIRE_SAFE_AREA_NONDECREASE", "true"
).lower() in {"1", "true", "yes", "on"}
HYBRID_MODEL_SKIP_NEAR_BOMB = os.environ.get(
    "HYBRID_MODEL_SKIP_NEAR_BOMB", "true"
).lower() in {"1", "true", "yes", "on"}

MODEL_TIE_MARGIN = 5.0
MODEL_MIN_CONFIDENCE_GAP = HYBRID_MODEL_CONF_MARGIN
MODEL_CANDIDATE_ACTIONS = tuple(MOVE_ACTIONS)
MODEL_SAFE_AREA_DEPTH = 6
MODEL_NEAR_BOMB_RADIUS = 3
MODEL_BLAST_LOOKAHEAD = 4

HYBRID_ENDGAME_ENABLE = os.environ.get("HYBRID_ENDGAME_ENABLE", "false").lower() in {
    "1", "true", "yes", "on"
}
HYBRID_ENDGAME_START_STEP = int(os.environ.get("HYBRID_ENDGAME_START_STEP", "350"))
HYBRID_ENDGAME_SEARCH_ENABLE = os.environ.get(
    "HYBRID_ENDGAME_SEARCH_ENABLE", "true"
).lower() in {"1", "true", "yes", "on"}
HYBRID_ENDGAME_SEARCH_BUDGET_MS = float(os.environ.get("HYBRID_ENDGAME_SEARCH_BUDGET_MS", "30"))
HYBRID_ENDGAME_MAX_DEPTH = int(os.environ.get("HYBRID_ENDGAME_MAX_DEPTH", "3"))
HYBRID_ENDGAME_ONLY_SAFE_ACTIONS = os.environ.get(
    "HYBRID_ENDGAME_ONLY_SAFE_ACTIONS", "true"
).lower() in {"1", "true", "yes", "on"}

ENDGAME_MIN_ACTION_MARGIN = 8.0


# ==============================================================================
# Helpers — bounds / state
# ==============================================================================

def _in_bounds(r, c):
    """Check if (r, c) is within the playable area (excludes outer walls)."""
    return 0 < r < BOARD_SIZE - 1 and 0 < c < BOARD_SIZE - 1


def _my_state(obs, agent_id):
    """Return (row, col, alive, bombs_left, radius_bonus) for this agent."""
    p = obs["players"][agent_id]
    return int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4])


def _alive_enemies(obs, agent_id):
    """Return list of (row, col) for every alive enemy."""
    players = obs["players"]
    enemies = []
    for i, p in enumerate(players):
        if i != agent_id and int(p[2]) == 1:
            enemies.append((int(p[0]), int(p[1])))
    return enemies


def _bomb_set(bombs):
    """Return a set of (row, col) for every active bomb on the floor."""
    arr = np.asarray(bombs, dtype=np.int32)
    if arr.size == 0:
        return set()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return {(int(arr[i, 0]), int(arr[i, 1])) for i in range(arr.shape[0])}


# ==============================================================================
# Helpers — passability
# ==============================================================================

def _is_passable(r, c, game_map, bomb_positions):
    """A cell can be walked into if it is grass/item and has no bomb."""
    if not _in_bounds(r, c):
        return False
    if game_map[r, c] in (TILE_WALL, TILE_BOX):
        return False
    if (r, c) in bomb_positions:
        return False
    return True


def _free_neighbors(pos, game_map, bomb_positions):
    """Count of passable neighbour cells (excluding STOP)."""
    r, c = pos
    cnt = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if _is_passable(nr, nc, game_map, bomb_positions):
                cnt += 1
    return cnt


def _is_corridor_or_dead_end(pos, game_map, bomb_set):
    """Classify local mobility around pos."""
    free = _free_neighbors(pos, game_map, bomb_set)
    if free <= 1:
        return "dead_end"
    if free == 2:
        return "corridor"
    return "open"


def _nearby_enemy_pressure(pos, players, agent_id):
    """Small risk estimate for being close to alive enemies."""
    pressure = 0
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1:
            continue
        dist = abs(pos[0] - int(p[0])) + abs(pos[1] - int(p[1]))
        if dist <= 2:
            pressure += NEAR_ENEMY_PRESSURE_2
        elif dist <= 4:
            pressure += NEAR_ENEMY_PRESSURE_4
    return pressure


def _future_survivability_score(pos, obs, danger_time,
                                max_depth=MAX_FUTURE_SURVIVABILITY,
                                cache=None):
    """Score how much safe space remains reachable from pos soon."""
    key = (pos[0], pos[1])
    if cache is not None and key in cache:
        return cache[key]

    game_map = obs["map"]
    bomb_set = _bomb_set(obs["bombs"])
    q = deque()
    q.append((pos, 0))
    visited = {pos}
    safe_cells = 0
    branch_points = 0
    dead_end_cells = 0
    max_safe_depth = 0

    while q:
        cell, dist = q.popleft()
        r, c = cell

        if danger_time[r, c] <= dist + 1:
            continue

        safe_cells += 1
        if dist > max_safe_depth:
            max_safe_depth = dist

        mobility = _free_neighbors(cell, game_map, bomb_set)
        if mobility >= 3:
            branch_points += 1
        elif mobility <= 1:
            dead_end_cells += 1

        if dist >= max_depth:
            continue

        for action in MOVE_ACTIONS:
            dr, dc = DIRS[action]
            nr, nc = r + dr, c + dc
            npos = (nr, nc)

            if npos in visited:
                continue
            if not _is_passable(nr, nc, game_map, bomb_set):
                continue

            visited.add(npos)
            q.append((npos, dist + 1))

    score = 0
    score += safe_cells * FUTURE_SAFE_CELL_WEIGHT
    score += branch_points * FUTURE_BRANCH_WEIGHT
    score += max_safe_depth * FUTURE_DEPTH_WEIGHT
    score -= dead_end_cells * FUTURE_DEAD_END_PENALTY

    if safe_cells <= 2:
        score -= 250
    elif safe_cells <= 4:
        score -= 120

    if _free_neighbors(pos, game_map, bomb_set) <= 1:
        score -= 180

    if danger_time[pos[0], pos[1]] == INF:
        score += 40
    elif danger_time[pos[0], pos[1]] >= 5:
        score += 20

    score = max(-500, min(score, 500))
    if cache is not None:
        cache[key] = score
    return score


def _controlled_expansion_score(pos, game_map, bomb_set, danger_time,
                                max_depth=EXPANSION_BFS_DEPTH, cache=None):
    """Small long-term value proxy for safe territory and map opening."""
    key = (pos[0], pos[1])
    if cache is not None and key in cache:
        return cache[key]

    q = deque()
    q.append((pos, 0))
    visited = {pos}
    safe_reachable = 0
    branch_points = 0
    adjacent_boxes = set()
    item_cells = 0

    while q:
        cell, dist = q.popleft()
        r, c = cell

        if danger_time[r, c] <= dist + 1:
            continue

        safe_reachable += 1
        if _free_neighbors(cell, game_map, bomb_set) >= 3:
            branch_points += 1
        if game_map[r, c] in (TILE_RADIUS, TILE_CAPACITY):
            item_cells += 1

        for dr, dc in BLAST_DIRS:
            ar, ac = r + dr, c + dc
            if _in_bounds(ar, ac) and game_map[ar, ac] == TILE_BOX:
                adjacent_boxes.add((ar, ac))

        if dist >= max_depth:
            continue

        for action in MOVE_ACTIONS:
            dr, dc = DIRS[action]
            nr, nc = r + dr, c + dc
            npos = (nr, nc)
            if npos in visited:
                continue
            if not _is_passable(nr, nc, game_map, bomb_set):
                continue
            visited.add(npos)
            q.append((npos, dist + 1))

    score = 0
    score += min(safe_reachable, 25) * 5
    score += min(branch_points, 8) * 10
    score += min(len(adjacent_boxes), 8) * 14
    score += item_cells * 35

    center = (BOARD_SIZE // 2, BOARD_SIZE // 2)
    dist_center = abs(pos[0] - center[0]) + abs(pos[1] - center[1])
    score += max(0, 40 - 5 * dist_center)

    zone_type = _is_corridor_or_dead_end(pos, game_map, bomb_set)
    if zone_type == "dead_end":
        score -= 80
    elif zone_type == "corridor":
        score -= 25

    score = max(-150, min(score, 220))
    if cache is not None:
        cache[key] = score
    return score


def _enemy_territory_pressure(pos, obs, danger_time, agent_id):
    """Light positional pressure without forcing risky chases."""
    if danger_time[pos[0], pos[1]] <= 2:
        return 0

    game_map = obs["map"]
    players = obs["players"]
    bomb_set = _bomb_set(obs["bombs"])
    zone_type = _is_corridor_or_dead_end(pos, game_map, bomb_set)
    if zone_type != "open":
        return 0

    center = (BOARD_SIZE // 2, BOARD_SIZE // 2)
    my_center_dist = abs(pos[0] - center[0]) + abs(pos[1] - center[1])
    pressure = 0

    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1:
            continue

        enemy_pos = (int(p[0]), int(p[1]))
        enemy_dist = abs(pos[0] - enemy_pos[0]) + abs(pos[1] - enemy_pos[1])
        enemy_mobility = _free_neighbors(enemy_pos, game_map, bomb_set)
        enemy_center_dist = abs(enemy_pos[0] - center[0]) + abs(enemy_pos[1] - center[1])

        if enemy_dist <= 4:
            pressure += 40
        if enemy_mobility <= 2:
            pressure += 40
        if my_center_dist < enemy_center_dist:
            pressure += 25

    return min(pressure, 120)


# ==============================================================================
# Helpers — blast geometry
# ==============================================================================

def _blast_cells(r, c, radius, game_map):
    """Set of (row, col) affected by a bomb at (r,c) with given radius."""
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


def _boxes_in_blast(r, c, radius, game_map):
    """How many boxes would be destroyed by a bomb at (r,c)?"""
    cnt = 0
    for br, bc in _blast_cells(r, c, radius, game_map):
        if game_map[br, bc] == TILE_BOX:
            cnt += 1
    return cnt


def _enemy_in_blast_line(r, c, radius, game_map, players, agent_id):
    """True if any alive enemy sits on a cell hit by a bomb at (r,c)."""
    for dr, dc in BLAST_DIRS:
        for step in range(1, radius + 1):
            nr, nc = r + dr * step, c + dc * step
            if not _in_bounds(nr, nc):
                break
            if game_map[nr, nc] == TILE_WALL:
                break
            for i, p in enumerate(players):
                if i != agent_id and int(p[2]) == 1:
                    if int(p[0]) == nr and int(p[1]) == nc:
                        return True
            if game_map[nr, nc] == TILE_BOX:
                break
    return False


def _bomb_danger_near(pos, danger_time, radius=MODEL_NEAR_BOMB_RADIUS):
    """True if a soon-active blast threat exists near pos."""
    pr, pc = pos
    for r in range(max(1, pr - radius), min(BOARD_SIZE - 1, pr + radius + 1)):
        for c in range(max(1, pc - radius), min(BOARD_SIZE - 1, pc + radius + 1)):
            if abs(r - pr) + abs(c - pc) <= radius and danger_time[r, c] < INF:
                return True
    return False


def _reachable_safe_area(pos, game_map, bomb_set, danger_time, max_depth=MODEL_SAFE_AREA_DEPTH):
    """Count cells reachable without arriving into danger."""
    q = deque()
    q.append((pos, 0))
    visited = {pos}
    safe_count = 0

    while q:
        cell, dist = q.popleft()
        r, c = cell
        if danger_time[r, c] <= dist + 1:
            continue
        safe_count += 1
        if dist >= max_depth:
            continue
        for action in MOVE_ACTIONS:
            dr, dc = DIRS[action]
            nr, nc = r + dr, c + dc
            npos = (nr, nc)
            if npos in visited:
                continue
            if not _is_passable(nr, nc, game_map, bomb_set):
                continue
            if danger_time[nr, nc] <= dist + 2:
                continue
            visited.add(npos)
            q.append((npos, dist + 1))

    return safe_count


def _in_current_or_future_blast(pos, danger_time, lookahead=MODEL_BLAST_LOOKAHEAD):
    r, c = pos
    return danger_time[r, c] <= lookahead


def _endgame_enemy_pressure_score(pos, obs, danger_time, agent_id):
    """Conservative pressure score for safe endgame positioning."""
    game_map = obs["map"]
    players = obs["players"]
    bomb_set = _bomb_set(obs["bombs"])
    enemies = _alive_enemies(obs, agent_id)
    if not enemies or danger_time[pos[0], pos[1]] < INF:
        return -1e9
    if _is_corridor_or_dead_end(pos, game_map, bomb_set) == "dead_end":
        return -1e9

    best = -1e9
    for enemy in enemies:
        dist = abs(pos[0] - enemy[0]) + abs(pos[1] - enemy[1])
        enemy_area = _enemy_safe_reachable_cells(
            enemy, game_map, obs["bombs"], danger_time, max_depth=5
        )
        cutoff_bonus = max(0, 12 - enemy_area) * 9
        distance_bonus = max(0, 8 - dist) * 10
        adjacency_penalty = 60 if dist <= 1 else 0
        best = max(best, cutoff_bonus + distance_bonus - adjacency_penalty)
    return best


def _endgame_candidate_score(pos, my_pos, obs, game_map, bomb_set, danger_time,
                             agent_id, baseline_area):
    if danger_time[pos[0], pos[1]] < INF:
        return -1e9
    if _in_current_or_future_blast(pos, danger_time, lookahead=5):
        return -1e9

    area = _reachable_safe_area(pos, game_map, bomb_set, danger_time, max_depth=7)
    if area < baseline_area:
        return -1e9

    mobility = _free_neighbors(pos, game_map, bomb_set)
    if mobility <= 1:
        return -1e9

    pressure = _endgame_enemy_pressure_score(pos, obs, danger_time, agent_id)
    if pressure <= -1e8:
        return -1e9

    center = (BOARD_SIZE // 2, BOARD_SIZE // 2)
    center_bonus = -0.8 * (abs(pos[0] - center[0]) + abs(pos[1] - center[1]))
    area_bonus = min(area, 18) * 1.8
    drift_penalty = 2.0 * (abs(pos[0] - my_pos[0]) + abs(pos[1] - my_pos[1]))
    return pressure + area_bonus + center_bonus - drift_penalty


# ==============================================================================
# Danger Map with chain-reaction relaxation
# ==============================================================================

def _compute_danger_map(game_map, players, bombs):
    """
    danger[r][c] = min number of steps until cell explodes (INF = safe).

    Uses iterative relaxation to propagate chain reactions:
      effective_timer[i] = min(own_timer, min_{j can reach i} effective_timer[j])
    """
    danger = np.full((BOARD_SIZE, BOARD_SIZE), INF, dtype=np.int32)

    bombs_arr = np.asarray(bombs, dtype=np.int32)
    if bombs_arr.size == 0:
        return danger
    if bombs_arr.ndim == 1:
        bombs_arr = bombs_arr.reshape(1, -1)

    n = bombs_arr.shape[0]

    # Per-bomb data
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

    # Relax chain reactions — at most n iterations
    for _ in range(n):
        changed = False
        for i in range(n):
            if effective[i] <= 0:
                continue
            ci = blast_sets[i]
            for j in range(n):
                if i == j or effective[j] <= 0:
                    continue
                bj_r, bj_c = int(bombs_arr[j, 0]), int(bombs_arr[j, 1])
                if (bj_r, bj_c) in ci:
                    if effective[i] < effective[j]:
                        effective[j] = effective[i]
                        changed = True
        if not changed:
            break

    # Fill danger grid
    for i in range(n):
        et = effective[i]
        if et <= 0:
            continue
        for r, c in blast_sets[i]:
            if et < danger[r, c]:
                danger[r, c] = et

    return danger


# ==============================================================================
# BFS utilities
# ==============================================================================

def _bfs(start, targets, game_map, bomb_set, danger_time, max_depth):
    """
    Multi-target BFS from start.
    A cell is reachable only if danger_time[cell] > (dist + 1).
    Returns (first_action, distance, target_cell) or None.
    """
    if start in targets:
        return None  # already there → no useful action

    q = deque()
    q.append((start, None, 0))
    visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    visited[start] = True

    while q:
        pos, first_action, dist = q.popleft()

        if pos in targets and dist > 0:
            return first_action, dist, pos

        if dist >= max_depth:
            continue

        for a in MOVE_ACTIONS:
            dr, dc = DIRS[a]
            nr, nc = pos[0] + dr, pos[1] + dc
            npos = (nr, nc)

            if visited[npos]:
                continue
            if not _is_passable(nr, nc, game_map, bomb_set):
                continue
            # danger_time > arrival_dist + 1
            if danger_time[nr, nc] <= dist + 2:
                continue

            visited[npos] = True
            q.append((npos, a if first_action is None else first_action, dist + 1))

    return None


def _find_nearest_safe(start, game_map, bomb_set, danger_time, max_depth):
    """
    BFS to nearest cell that is permanently safe (danger_time == INF).
    Returns (first_action, distance, target) or None.
    """
    q = deque()
    q.append((start, None, 0))
    visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    visited[start] = True

    while q:
        pos, first_action, dist = q.popleft()

        if dist > 0 and danger_time[pos] == INF:
            return first_action, dist, pos

        if dist >= max_depth:
            continue

        for a in MOVE_ACTIONS:
            dr, dc = DIRS[a]
            nr, nc = pos[0] + dr, pos[1] + dc
            npos = (nr, nc)

            if visited[npos]:
                continue
            if not _is_passable(nr, nc, game_map, bomb_set):
                continue
            if danger_time[nr, nc] <= dist + 2:
                continue

            visited[npos] = True
            q.append((npos, a if first_action is None else first_action, dist + 1))

    return None


def _bfs_to_nearest(start, target_set, game_map, bomb_set, danger_time, max_depth):
    """Convenience wrapper around _bfs; returns (first_action, dist, target)."""
    return _bfs(start, target_set, game_map, bomb_set, danger_time, max_depth)


def _item_targets(game_map):
    """Set of cells that contain items."""
    targets = set()
    for r in range(1, BOARD_SIZE - 1):
        for c in range(1, BOARD_SIZE - 1):
            if game_map[r, c] in (TILE_RADIUS, TILE_CAPACITY):
                targets.add((r, c))
    return targets


def _box_bomb_spots(game_map, bomb_set):
    """Set of passable cells adjacent to at least one box."""
    spots = set()
    for r in range(1, BOARD_SIZE - 1):
        for c in range(1, BOARD_SIZE - 1):
            if game_map[r, c] != TILE_BOX:
                continue
            for dr, dc in BLAST_DIRS:
                nr, nc = r + dr, c + dc
                if _is_passable(nr, nc, game_map, bomb_set):
                    spots.add((nr, nc))
    return spots


# ==============================================================================
# Bomb escape check
# ==============================================================================

def _can_escape_after_bomb(obs, agent_id, game_map, bomb_set):
    """Simulate placing a bomb at current position; check escape path exists."""
    my_r, my_c, _, _, bonus = _my_state(obs, agent_id)
    radius = 1 + bonus
    players = obs["players"]

    bombs_arr = np.asarray(obs["bombs"], dtype=np.int32)
    if bombs_arr.size == 0:
        new_bombs = np.array([[my_r, my_c, BOMB_TIMER, agent_id]], dtype=np.int32)
    else:
        if bombs_arr.ndim == 1:
            bombs_arr = bombs_arr.reshape(1, -1)
        new_bomb_row = np.array([[my_r, my_c, BOMB_TIMER, agent_id]], dtype=np.int32)
        new_bombs = np.vstack([bombs_arr, new_bomb_row])

    new_danger = _compute_danger_map(game_map, players, new_bombs)

    result = _find_nearest_safe(
        (my_r, my_c), game_map, bomb_set, new_danger, MAX_BFS_ESCAPE
    )
    return result is not None


def _simulate_bomb_danger_map(obs, bomb_pos, owner_id, timer=BOMB_TIMER):
    """Return danger map after adding one hypothetical bomb without mutating obs."""
    game_map = obs["map"]
    players = obs["players"]
    br, bc = bomb_pos

    bombs_arr = np.asarray(obs["bombs"], dtype=np.int32)
    fake_bomb = np.array([[br, bc, timer, owner_id]], dtype=np.int32)
    if bombs_arr.size == 0:
        new_bombs = fake_bomb
    else:
        if bombs_arr.ndim == 1:
            bombs_arr = bombs_arr.reshape(1, -1)
        new_bombs = np.vstack([bombs_arr, fake_bomb])

    return _compute_danger_map(game_map, players, new_bombs)


def _enemy_safe_reachable_cells(enemy_pos, game_map, bombs, danger_time,
                                max_depth=MAX_ENEMY_ESCAPE_SIM):
    """Count safe cells an enemy can reach within max_depth."""
    bomb_positions = _bomb_set(bombs)
    q = deque()
    q.append((enemy_pos, 0))
    visited = {enemy_pos}
    safe_count = 0

    while q:
        pos, dist = q.popleft()
        r, c = pos

        if danger_time[r, c] > dist + 1:
            safe_count += 1

        if dist >= max_depth:
            continue

        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            npos = (nr, nc)

            if npos in visited:
                continue
            if not _is_passable(nr, nc, game_map, bomb_positions):
                continue
            if danger_time[nr, nc] <= dist:
                continue

            visited.add(npos)
            q.append((npos, dist + 1))

    return safe_count


def _estimate_enemy_escape_pressure(obs, bomb_pos, agent_id):
    """Estimate how much a hypothetical bomb restricts enemy escape options."""
    game_map = obs["map"]
    players = obs["players"]
    bombs = obs["bombs"]
    my_bonus = int(players[agent_id][4]) if 0 <= agent_id < len(players) else 0
    radius = 1 + my_bonus

    danger_before = _compute_danger_map(game_map, players, bombs)
    danger_after = _simulate_bomb_danger_map(obs, bomb_pos, agent_id, BOMB_TIMER)
    bombs_arr = np.asarray(bombs, dtype=np.int32)
    fake_bomb = np.array([[bomb_pos[0], bomb_pos[1], BOMB_TIMER, agent_id]],
                         dtype=np.int32)
    if bombs_arr.size == 0:
        bombs_after = fake_bomb
    else:
        if bombs_arr.ndim == 1:
            bombs_arr = bombs_arr.reshape(1, -1)
        bombs_after = np.vstack([bombs_arr, fake_bomb])
    pressure = 0

    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1:
            continue

        enemy_pos = (int(p[0]), int(p[1]))
        safe_before = _enemy_safe_reachable_cells(
            enemy_pos, game_map, bombs, danger_before, MAX_ENEMY_ESCAPE_SIM
        )
        safe_after = _enemy_safe_reachable_cells(
            enemy_pos, game_map, bombs_after, danger_after, MAX_ENEMY_ESCAPE_SIM
        )
        reduced = safe_before - safe_after
        direct_hit = enemy_pos in _blast_cells(
            bomb_pos[0], bomb_pos[1], radius, game_map
        )
        near_bomb = abs(enemy_pos[0] - bomb_pos[0]) + abs(enemy_pos[1] - bomb_pos[1]) <= 3

        if reduced > 0 or direct_hit:
            if safe_after == 0:
                pressure += 900
            elif safe_after <= 2:
                pressure += 500
            elif safe_after <= 4:
                pressure += 250

        if reduced > 0:
            pressure += reduced * 60

        if direct_hit:
            pressure += 350

        if near_bomb:
            pressure += 120

    return min(pressure, 1600)


def _is_meaningful_bomb(obs, bomb_pos, agent_id, escape_pressure=None):
    """True when a bomb has farming value or creates meaningful enemy pressure."""
    game_map = obs["map"]
    players = obs["players"]
    bonus = int(players[agent_id][4]) if 0 <= agent_id < len(players) else 0
    radius = 1 + bonus
    boxes = _boxes_in_blast(bomb_pos[0], bomb_pos[1], radius, game_map)
    threatens = _enemy_in_blast_line(
        bomb_pos[0], bomb_pos[1], radius, game_map, players, agent_id
    )
    if escape_pressure is None:
        escape_pressure = _estimate_enemy_escape_pressure(obs, bomb_pos, agent_id)
    return boxes > 0 or threatens or escape_pressure >= MEANINGFUL_ESCAPE_PRESSURE


# ==============================================================================
# Action scoring
# ==============================================================================

def _score_move(action, my_pos, my_state, game_map, players, bomb_set,
                danger_time, item_targets, box_spots, enemies, obs, agent_id,
                future_cache=None, expansion_cache=None, is_loop=False,
                position_history=None):
    """Score a movement action (0-4)."""
    my_r, my_c = my_pos
    _, _, _, bombs_left, _ = my_state

    dr, dc = DIRS[action]
    nr, nc = my_r + dr, my_c + dc
    npos = (nr, nc)

    # Invalid move → huge penalty
    if not _is_passable(nr, nc, game_map, bomb_set):
        return -1e9, None

    score = 0.0
    dt = danger_time[nr, nc]
    curr_dt = danger_time[my_r, my_c]

    # ----- Safety (invariant) -----
    if dt <= 1:
        return -1e9, None
    elif dt <= 3:
        score -= 800            # dangerous in near future
    elif dt == INF:
        score += 80             # perfectly safe

    # Escape: leaving a dangerous cell for a safer one
    if curr_dt <= 3 and dt > curr_dt:
        score += 500

    # ----- Mobility -----
    free_n = _free_neighbors(npos, game_map, bomb_set)
    score += free_n * 25
    if free_n <= 1:
        score -= 300 if free_n == 0 else 100

    # ----- Future survivability -----
    future_score = _future_survivability_score(
        npos, obs, danger_time, MAX_FUTURE_SURVIVABILITY, future_cache
    )
    score += future_score * FUTURE_SCORE_MULTIPLIER

    zone_type = _is_corridor_or_dead_end(npos, game_map, bomb_set)
    enemy_pressure = _nearby_enemy_pressure(npos, players, agent_id)
    if zone_type == "dead_end":
        score -= DEAD_END_PENALTY
        score -= enemy_pressure
    elif zone_type == "corridor":
        score -= CORRIDOR_PENALTY
        score -= enemy_pressure * 0.5
    else:
        score += OPEN_ZONE_BONUS

    # ----- Controlled expansion / long-term territory value -----
    expansion_score = _controlled_expansion_score(
        npos, game_map, bomb_set, danger_time, EXPANSION_BFS_DEPTH,
        expansion_cache
    )
    if is_loop:
        score += expansion_score * LOOP_EXPANSION_SCORE_MULTIPLIER
    else:
        score += expansion_score * EXPANSION_SCORE_MULTIPLIER

    territory_pressure = _enemy_territory_pressure(
        npos, obs, danger_time, agent_id
    )
    score += territory_pressure * TERRITORY_PRESSURE_MULTIPLIER

    # ----- Loop breaker -----
    if position_history:
        repeat_count = position_history.count(npos)
        if is_loop:
            if action == A_STOP:
                score -= 180
            if repeat_count > 0:
                score -= repeat_count * 50
            else:
                score += 100
            if zone_type == "open":
                score += 60

    # ----- Item pickup -----
    if game_map[nr, nc] in (TILE_RADIUS, TILE_CAPACITY):
        score += 250

    # ----- Item approach -----
    if item_targets:
        item_bfs = _bfs_to_nearest(npos, item_targets, game_map, bomb_set,
                                   danger_time, MAX_BFS_ITEM)
        if item_bfs is not None:
            _, item_dist, _ = item_bfs
            score += max(0, 120 - 10 * item_dist)

    # ----- Box-farm approach -----
    if box_spots:
        box_bfs = _bfs_to_nearest(npos, box_spots, game_map, bomb_set,
                                  danger_time, MAX_BFS_TARGET)
        if box_bfs is not None:
            _, box_dist, _ = box_bfs
            score += max(0, 100 - 8 * box_dist)

    # Near-box adjacency bonus (good position for future bombing)
    if bombs_left > 0:
        for dr2, dc2 in BLAST_DIRS:
            adj_r, adj_c = nr + dr2, nc + dc2
            if _in_bounds(adj_r, adj_c) and game_map[adj_r, adj_c] == TILE_BOX:
                score += 80
                break

    # ----- Enemy approach -----
    if enemies:
        enemy_set = set(enemies)
        enemy_bfs = _bfs_to_nearest(npos, enemy_set, game_map, bomb_set,
                                    danger_time, MAX_BFS_ENEMY)
        if enemy_bfs is not None:
            _, enemy_dist, _ = enemy_bfs
            score += max(0, 80 - 6 * enemy_dist)

    # ----- STOP-specific -----
    if action == A_STOP:
        if dt <= 1:
            score -= 2000
        score -= 10  # slight bias toward moving

    return score, npos


def _score_bomb(my_pos, my_state, game_map, players, bomb_set,
                danger_time, obs, agent_id, escape_pressure=None,
                is_loop=False):
    """Score the PLACE_BOMB action.  Returns -1e9 if invalid or unsafe."""
    my_r, my_c = my_pos
    _, _, _, bombs_left, bonus = my_state
    radius = 1 + bonus

    # Preconditions (invariant)
    if bombs_left <= 0:
        return -1e9
    if my_pos in bomb_set:
        return -1e9
    if not _can_escape_after_bomb(obs, agent_id, game_map, bomb_set):
        return -1e9
    if escape_pressure is None:
        escape_pressure = _estimate_enemy_escape_pressure(obs, my_pos, agent_id)
    if not _is_meaningful_bomb(obs, my_pos, agent_id, escape_pressure):
        return -1e9

    simulated_danger = _simulate_bomb_danger_map(obs, my_pos, agent_id, BOMB_TIMER)
    escape_future_score = _future_survivability_score(
        my_pos, obs, simulated_danger, MAX_FUTURE_SURVIVABILITY
    )
    if escape_future_score < -250:
        return -1e9

    score = 0.0
    boxes = _boxes_in_blast(my_r, my_c, radius, game_map)
    threatens = _enemy_in_blast_line(my_r, my_c, radius, game_map, players, agent_id)

    # ----- Box destruction -----
    if boxes > 0:
        score += 350 + 80 * boxes
        if is_loop:
            score += 100

    # ----- Direct enemy hit -----
    if threatens:
        score += 500

    # ----- Enemy escape pressure -----
    score += escape_pressure * ESCAPE_PRESSURE_SCORE_MULTIPLIER

    # ----- Future survivability after bomb -----
    if escape_future_score < -100:
        score -= 300

    # ----- Proximity threat -----
    enemies = _alive_enemies(obs, agent_id)
    if enemies:
        blast = _blast_cells(my_r, my_c, radius, game_map)
        for er, ec in enemies:
            dist = min(abs(er - br) + abs(ec - bc) for br, bc in blast)
            if dist <= 3:
                score += 150
                break

    # ----- Penalise valueless bombs -----
    if boxes == 0 and not threatens:
        score -= 200

    return score


# ==============================================================================
# Optional model tiebreaker
# ==============================================================================

class _OptionalActionRanker:
    """Lazy, failure-tolerant action ranker used only for safe movement ties."""

    def __init__(self, checkpoint_path):
        self.checkpoint_path = str(checkpoint_path)
        self.enabled = bool(HYBRID_MODEL_ENABLE)
        self.loaded = False
        self.load_attempted = False
        self.load_failed = False
        self.model = None
        self.torch = None
        self.encode_observation = None

    def _load(self):
        if not self.enabled or self.load_failed:
            return False
        if self.loaded:
            return True
        if self.load_attempted:
            return False
        self.load_attempted = True

        try:
            checkpoint = Path(self.checkpoint_path)
            if not checkpoint.exists():
                self.load_failed = True
                return False

            import torch
            from ml.features import encode_observation
            from ml.models.simple_cnn_policy import load_checkpoint

            model, _metadata = load_checkpoint(str(checkpoint), map_location="cpu")
            model.eval()
            self.torch = torch
            self.encode_observation = encode_observation
            self.model = model
            self.loaded = True
            return True
        except Exception:
            self.load_failed = True
            self.model = None
            self.torch = None
            self.encode_observation = None
            return False

    def choose(self, obs, agent_id, candidate_actions):
        """Return (action, gap, elapsed_ms), or (None, 0.0, elapsed_ms)."""
        started = time.perf_counter()
        if not self._load():
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return None, 0.0, elapsed_ms

        try:
            frame = dict(obs)
            frame["_agent_index"] = int(agent_id)
            encoded = self.encode_observation(frame)
            tensor = encoded["tensor"].astype(np.float32, copy=False)
            x = self.torch.from_numpy(tensor).unsqueeze(0)
            with self.torch.no_grad():
                logits = self.model(x)[0]
                probs = self.torch.softmax(logits, dim=0).cpu().numpy()

            ranked = sorted(
                ((int(action), float(probs[int(action)])) for action in candidate_actions),
                key=lambda item: item[1],
                reverse=True,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if not ranked:
                return None, 0.0, elapsed_ms
            gap = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
            return ranked[0][0], float(gap), elapsed_ms
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return None, 0.0, elapsed_ms


# ==============================================================================
# Agent class
# ==============================================================================

class Agent:
    """
    Hybrid Bomberland agent combining:
      - Danger-aware BFS pathfinding
      - Heuristic action scoring (safety, items, boxes, enemies)
      - Escape verification before every bomb placement

    Optional ML tiebreaker is disabled by default and may only rerank safe
    movement actions.  Under 100 ms per act().
    """
    team_id = "HybridAgent-ModelOptimized"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.position_history = []
        self.max_position_history = MAX_POSITION_HISTORY
        self.last_bomb_step = -10**9
        self.model_ranker = _OptionalActionRanker(HYBRID_MODEL_CHECKPOINT)
        self.counters = {
            "model_loaded": 0,
            "model_inference_errors": 0,
            "model_tiebreaker_used": 0,
            "model_action_accepted": 0,
            "model_action_rejected_by_safety": 0,
            "fallback_to_rule": 0,
        }
        self.reject_reason_counts = {}
        self.intervention_log = []
        self.endgame_counters = {
            "endgame_active_steps": 0,
            "endgame_search_used": 0,
            "endgame_action_changed": 0,
            "endgame_bomb_accepted": 0,
            "endgame_bomb_rejected_safety": 0,
            "endgame_fallback_to_production": 0,
            "endgame_timeout_fallback": 0,
        }
        self.endgame_reject_reason_counts = {}
        if HYBRID_MODEL_ENABLE and self.model_ranker._load():
            self.counters["model_loaded"] = 1

    def _maybe_reset_position_history(self, my_pos, bombs):
        spawn_positions = {
            0: (1, 1),
            1: (BOARD_SIZE - 2, BOARD_SIZE - 2),
            2: (1, BOARD_SIZE - 2),
            3: (BOARD_SIZE - 2, 1),
        }
        if (
            my_pos == spawn_positions.get(self.agent_id)
            and len(_bomb_set(bombs)) == 0
            and len(self.position_history) > 10
        ):
            self.position_history = []

    def _update_position_history(self, my_pos):
        self.position_history.append(my_pos)
        if len(self.position_history) > self.max_position_history:
            self.position_history = self.position_history[-self.max_position_history:]

    def _is_local_loop(self):
        if not self.position_history:
            return False
        recent = self.position_history[-8:]
        return (
            len(recent) >= 6
            and (len(set(recent)) <= 3
                 or self.position_history.count(self.position_history[-1]) >= 4)
        )

    def _reject_model(self, reason, rule_action):
        self.reject_reason_counts[reason] = self.reject_reason_counts.get(reason, 0) + 1
        self.counters["fallback_to_rule"] += 1
        return rule_action

    def _log_intervention(self, step, rule_action, model_action, candidates, gap):
        if len(self.intervention_log) >= 100:
            self.intervention_log.pop(0)
        self.intervention_log.append({
            "step": int(step),
            "rule_action": int(rule_action),
            "model_action": int(model_action),
            "candidates": [int(action) for action in candidates],
            "confidence_margin": float(gap),
        })

    def _reject_endgame(self, reason, production_action):
        self.endgame_reject_reason_counts[reason] = (
            self.endgame_reject_reason_counts.get(reason, 0) + 1
        )
        self.endgame_counters["endgame_fallback_to_production"] += 1
        return production_action

    def _maybe_endgame_optimize(self, obs, production_action, final_scores,
                                action_positions, my_pos, game_map, bomb_set,
                                danger_time, emergency_first_action):
        if not HYBRID_ENDGAME_ENABLE:
            return production_action

        step = int(obs["step"]) if "step" in obs else 0
        alive_players = sum(1 for player in obs["players"] if int(player[2]) == 1)
        if step <= HYBRID_ENDGAME_START_STEP and alive_players > 3:
            return self._reject_endgame("not_endgame", production_action)

        self.endgame_counters["endgame_active_steps"] += 1

        if emergency_first_action is not None and production_action == emergency_first_action:
            return self._reject_endgame("emergency_or_path_to_safety", production_action)
        if danger_time[my_pos[0], my_pos[1]] < INF:
            return self._reject_endgame("current_danger", production_action)
        if production_action == A_BOMB:
            # Bomb remains production-owned. Endgame layer may not introduce or
            # strengthen bomb placement without a separate safety validation.
            self.endgame_counters["endgame_bomb_accepted"] += 1
            return production_action
        if HYBRID_ENDGAME_ONLY_SAFE_ACTIONS and production_action not in MOVE_ACTIONS:
            return self._reject_endgame("production_not_movement", production_action)

        started = time.perf_counter()
        baseline_pos = action_positions.get(production_action, my_pos)
        baseline_area = _reachable_safe_area(
            baseline_pos, game_map, bomb_set, danger_time, max_depth=7
        )
        baseline_score = _endgame_candidate_score(
            baseline_pos, my_pos, obs, game_map, bomb_set, danger_time,
            self.agent_id, baseline_area
        )

        best_action = production_action
        best_score = baseline_score
        considered = 0
        for action in MOVE_ACTIONS:
            if (time.perf_counter() - started) * 1000.0 > HYBRID_ENDGAME_SEARCH_BUDGET_MS:
                self.endgame_counters["endgame_timeout_fallback"] += 1
                return self._reject_endgame("time_budget", production_action)
            if final_scores.get(action, -1e9) <= -1e8:
                continue
            pos = action_positions.get(action)
            if pos is None:
                continue
            considered += 1
            score = _endgame_candidate_score(
                pos, my_pos, obs, game_map, bomb_set, danger_time,
                self.agent_id, baseline_area
            )
            if score > best_score:
                best_score = score
                best_action = action

        if considered < 2:
            return self._reject_endgame("too_few_candidates", production_action)

        self.endgame_counters["endgame_search_used"] += 1
        if best_action != production_action and best_score >= baseline_score + ENDGAME_MIN_ACTION_MARGIN:
            self.endgame_counters["endgame_action_changed"] += 1
            return best_action
        return self._reject_endgame("no_clear_improvement", production_action)

    def _maybe_model_tiebreak(self, obs, rule_action, final_scores, action_positions,
                              my_pos, game_map, bomb_set, danger_time,
                              emergency_first_action):
        """Rerank a close safe movement tie, otherwise return rule_action."""
        if not HYBRID_MODEL_ENABLE:
            return self._reject_model("disabled", rule_action)

        if HYBRID_MODEL_ONLY_SAFE_ACTIONS and rule_action not in MODEL_CANDIDATE_ACTIONS:
            return self._reject_model("rule_action_not_movement", rule_action)

        current_step = int(obs["step"]) if "step" in obs else None
        if (
            current_step is not None
            and current_step - self.last_bomb_step <= HYBRID_MODEL_SKIP_AFTER_BOMB_STEPS
        ):
            return self._reject_model("recent_own_bomb", rule_action)

        if emergency_first_action is not None and rule_action == emergency_first_action:
            return self._reject_model("path_to_safety", rule_action)

        if danger_time[my_pos[0], my_pos[1]] < INF:
            return self._reject_model("current_cell_in_bomb_danger", rule_action)

        if HYBRID_MODEL_SKIP_NEAR_BOMB and _bomb_danger_near(my_pos, danger_time):
            return self._reject_model("near_bomb_danger", rule_action)

        rule_score = final_scores.get(rule_action, -1e9)
        if rule_score <= -1e8:
            return self._reject_model("invalid_rule_score", rule_action)

        baseline_area = _reachable_safe_area(
            action_positions.get(rule_action, my_pos),
            game_map,
            bomb_set,
            danger_time,
            MODEL_SAFE_AREA_DEPTH,
        )

        candidates = []
        for action in MODEL_CANDIDATE_ACTIONS:
            score = final_scores.get(action, -1e9)
            pos = action_positions.get(action)
            if score <= -1e8 or pos is None:
                continue
            if rule_score - score > MODEL_TIE_MARGIN:
                continue
            if _in_current_or_future_blast(pos, danger_time):
                self.counters["model_action_rejected_by_safety"] += 1
                self.reject_reason_counts["candidate_in_future_blast"] = (
                    self.reject_reason_counts.get("candidate_in_future_blast", 0) + 1
                )
                continue
            if HYBRID_MODEL_REQUIRE_SAFE_AREA_NONDECREASE:
                area = _reachable_safe_area(
                    pos, game_map, bomb_set, danger_time, MODEL_SAFE_AREA_DEPTH
                )
                if area < baseline_area:
                    self.counters["model_action_rejected_by_safety"] += 1
                    self.reject_reason_counts["safe_area_decrease"] = (
                        self.reject_reason_counts.get("safe_area_decrease", 0) + 1
                    )
                    continue
            candidates.append(action)

        if len(candidates) < 2:
            return self._reject_model("too_few_safe_candidates", rule_action)

        choice, gap, elapsed_ms = self.model_ranker.choose(obs, self.agent_id, candidates)
        self.counters["model_loaded"] = 1 if self.model_ranker.loaded else 0

        if choice is None:
            if self.model_ranker.loaded:
                self.counters["model_inference_errors"] += 1
            return self._reject_model("model_inference_failed", rule_action)

        self.counters["model_tiebreaker_used"] += 1

        if elapsed_ms > HYBRID_MODEL_MAX_LATENCY_MS:
            return self._reject_model("latency_gate", rule_action)

        if choice not in candidates or choice == A_BOMB:
            self.counters["model_action_rejected_by_safety"] += 1
            return self._reject_model("choice_rejected_by_safety", rule_action)

        if gap < MODEL_MIN_CONFIDENCE_GAP:
            return self._reject_model("confidence_margin", rule_action)

        if choice != rule_action:
            self.counters["model_action_accepted"] += 1
            self._log_intervention(-1 if current_step is None else current_step, rule_action, choice, candidates, gap)
        return choice

    def act(self, obs: dict) -> int:
        self._future_cache = {}
        self._expansion_cache = {}
        game_map = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        # ---- 1. State Parser ----
        my_r, my_c, alive, bombs_left, bonus = _my_state(obs, self.agent_id)
        if not alive:
            return A_STOP

        my_pos = (my_r, my_c)
        self._maybe_reset_position_history(my_pos, bombs)
        self._update_position_history(my_pos)
        is_loop = self._is_local_loop()

        my_state = (my_r, my_c, alive, bombs_left, bonus)
        bomb_set = _bomb_set(bombs)
        enemies = _alive_enemies(obs, self.agent_id)

        # ---- 2. Danger Map ----
        danger_time = _compute_danger_map(game_map, players, bombs)

        # ---- 3. Pre-compute shared data ----
        item_targets = _item_targets(game_map)
        box_spots = _box_bomb_spots(game_map, bomb_set)

        # ---- 4. Emergency escape ----
        curr_dt = danger_time[my_r, my_c]
        emergency_first_action = None
        if curr_dt <= 3:
            safe = _find_nearest_safe(my_pos, game_map, bomb_set,
                                      danger_time, MAX_BFS_SAFE)
            if safe is not None:
                emergency_first_action = safe[0]
                return safe[0]  # first_action
        elif curr_dt < INF:
            safe = _find_nearest_safe(my_pos, game_map, bomb_set,
                                      danger_time, MAX_BFS_SAFE)
            if safe is not None:
                emergency_first_action = safe[0]

        bomb_escape_pressure = None
        if bombs_left > 0 and my_pos not in bomb_set:
            if _can_escape_after_bomb(obs, self.agent_id, game_map, bomb_set):
                bomb_escape_pressure = _estimate_enemy_escape_pressure(
                    obs, my_pos, self.agent_id
                )

        # ---- 5. Score all actions ----
        best_action = A_STOP
        best_score = -1e9
        raw_scores = {}
        action_positions = {}

        for action in ALL_ACTIONS:
            if action == A_BOMB:
                score = _score_bomb(my_pos, my_state, game_map, players,
                                    bomb_set, danger_time, obs,
                                    self.agent_id, bomb_escape_pressure,
                                    is_loop)
                pos = my_pos
            else:
                score, pos = _score_move(action, my_pos, my_state, game_map,
                                         players, bomb_set, danger_time,
                                         item_targets, box_spots, enemies, obs,
                                         self.agent_id, self._future_cache,
                                         self._expansion_cache, is_loop,
                                         self.position_history)

            raw_scores[action] = score
            action_positions[action] = pos

        valid_scores = [score for score in raw_scores.values() if score > -1e8]
        if valid_scores:
            mean_score = sum(valid_scores) / len(valid_scores)
            final_scores = {
                action: (
                    score + ACTION_ADVANTAGE_WEIGHT * (score - mean_score)
                    if score > -1e8 else score
                )
                for action, score in raw_scores.items()
            }
        else:
            final_scores = raw_scores

        for action, score in final_scores.items():
            if score > best_score:
                best_score = score
                best_action = action
                best_pos = action_positions.get(action)

        # ---- 6. Bomb hysteresis ----
        if best_action == A_BOMB:
            best_move_score = -1e9
            for action in MOVE_ACTIONS:
                score, _ = _score_move(action, my_pos, my_state, game_map,
                                       players, bomb_set, danger_time,
                                       item_targets, box_spots, enemies, obs,
                                       self.agent_id, self._future_cache,
                                       self._expansion_cache, is_loop,
                                       self.position_history)
                if score > best_move_score:
                    best_move_score = score
            pressure = 0 if bomb_escape_pressure is None else bomb_escape_pressure
            threshold = BOMB_HYSTERESIS_TUNED_THRESHOLD
            if len(enemies) == 1 or pressure >= 900:
                threshold = 0
            elif pressure >= 500:
                threshold = 80
            if best_score < best_move_score + threshold:
                # Fall back to best move
                best_action = A_STOP
                best_score = -1e9
                for action in MOVE_ACTIONS:
                    score, pos = _score_move(action, my_pos, my_state, game_map,
                                             players, bomb_set, danger_time,
                                             item_targets, box_spots, enemies,
                                             obs, self.agent_id,
                                             self._future_cache,
                                             self._expansion_cache, is_loop,
                                             self.position_history)
                    if score > best_score:
                        best_score = score
                        best_action = action

        rule_action = best_action
        best_action = self._maybe_model_tiebreak(
            obs,
            rule_action,
            final_scores,
            action_positions,
            my_pos,
            game_map,
            bomb_set,
            danger_time,
            emergency_first_action,
        )
        production_action = best_action
        best_action = self._maybe_endgame_optimize(
            obs,
            production_action,
            final_scores,
            action_positions,
            my_pos,
            game_map,
            bomb_set,
            danger_time,
            emergency_first_action,
        )
        if best_action == A_BOMB:
            if "step" in obs:
                self.last_bomb_step = int(obs["step"])
        return best_action
