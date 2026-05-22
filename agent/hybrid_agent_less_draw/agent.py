"""
Hybrid Bomberland Agent: Rule-based + BFS + Heuristic Scoring.

No ML dependencies. Pure Python stdlib + numpy only.
Designed to run well under 100ms per act() call.

Architecture:
  1. State Parser       — extract self, enemies, bombs
  2. Danger Map         — compute threat grid with chain-reaction relaxation
  3. BFS Pathfinding    — safe-path search, escape verification
  4. Heuristic Scorer   — rank all 6 actions, pick best
"""

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
LATE_PRESSURE_MULTIPLIER = 1.15
ANTI_DRAW_PRESSURE_MULTIPLIER = 1.30
DUEL_PRESSURE_MULTIPLIER = 1.35
ANTI_DRAW_BOMB_PRESSURE_BONUS = 100
HIGH_BOMB_PRESSURE_BONUS = 180
LATE_STOP_PENALTY = 80
NEAR_ENEMY_STOP_PENALTY = 120
DUEL_CHASE_BONUS = 90
DUEL_RETREAT_PENALTY = 50

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


def _phase_pressure_multiplier(phase):
    if phase == "duel":
        return DUEL_PRESSURE_MULTIPLIER
    if phase == "anti_draw":
        return ANTI_DRAW_PRESSURE_MULTIPLIER
    if phase == "late_pressure":
        return LATE_PRESSURE_MULTIPLIER
    return 1.0


def _phase_item_multiplier(phase):
    if phase in ("anti_draw", "duel"):
        return 0.5
    if phase == "late_pressure":
        return 0.75
    return 1.0


def _phase_box_multiplier(phase):
    if phase in ("anti_draw", "duel"):
        return 0.4
    if phase == "late_pressure":
        return 0.75
    return 1.0


# ==============================================================================
# Action scoring
# ==============================================================================

def _score_move(action, my_pos, my_state, game_map, players, bomb_set,
                danger_time, item_targets, box_spots, enemies, obs, agent_id,
                future_cache=None, phase="normal"):
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

    # ----- Item pickup -----
    if game_map[nr, nc] in (TILE_RADIUS, TILE_CAPACITY):
        score += 250 * _phase_item_multiplier(phase)

    # ----- Item approach -----
    if item_targets:
        item_bfs = _bfs_to_nearest(npos, item_targets, game_map, bomb_set,
                                   danger_time, MAX_BFS_ITEM)
        if item_bfs is not None:
            _, item_dist, _ = item_bfs
            score += max(0, 120 - 10 * item_dist) * _phase_item_multiplier(phase)

    # ----- Box-farm approach -----
    if box_spots:
        box_bfs = _bfs_to_nearest(npos, box_spots, game_map, bomb_set,
                                  danger_time, MAX_BFS_TARGET)
        if box_bfs is not None:
            _, box_dist, _ = box_bfs
            score += max(0, 100 - 8 * box_dist) * _phase_box_multiplier(phase)

    # Near-box adjacency bonus (good position for future bombing)
    if bombs_left > 0:
        for dr2, dc2 in BLAST_DIRS:
            adj_r, adj_c = nr + dr2, nc + dc2
            if _in_bounds(adj_r, adj_c) and game_map[adj_r, adj_c] == TILE_BOX:
                score += 80 * _phase_box_multiplier(phase)
                break

    # ----- Enemy approach -----
    if enemies:
        enemy_set = set(enemies)
        enemy_bfs = _bfs_to_nearest(npos, enemy_set, game_map, bomb_set,
                                    danger_time, MAX_BFS_ENEMY)
        if enemy_bfs is not None:
            _, enemy_dist, _ = enemy_bfs
            score += max(0, 80 - 6 * enemy_dist) * _phase_pressure_multiplier(phase)

        if phase == "duel" and enemies and dt > 2 and future_score >= -100:
            enemy_dist_now = min(abs(my_r - er) + abs(my_c - ec) for er, ec in enemies)
            enemy_dist_next = min(abs(nr - er) + abs(nc - ec) for er, ec in enemies)
            if not (zone_type in ("dead_end", "corridor") and enemy_pressure >= 120):
                if enemy_dist_next < enemy_dist_now:
                    score += DUEL_CHASE_BONUS
                elif enemy_dist_next > enemy_dist_now and curr_dt > 3:
                    score -= DUEL_RETREAT_PENALTY

    # ----- STOP-specific -----
    if action == A_STOP:
        if dt <= 1:
            score -= 2000
        score -= 10  # slight bias toward moving
        if phase in ("late_pressure", "anti_draw", "duel") and curr_dt > 3:
            score -= LATE_STOP_PENALTY
            if enemies:
                nearest_enemy = min(abs(my_r - er) + abs(my_c - ec) for er, ec in enemies)
                if nearest_enemy <= 4:
                    score -= NEAR_ENEMY_STOP_PENALTY

    return score, npos


def _score_bomb(my_pos, my_state, game_map, players, bomb_set,
                danger_time, obs, agent_id, escape_pressure=None, phase="normal"):
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
        score += (350 + 80 * boxes) * _phase_box_multiplier(phase)

    # ----- Direct enemy hit -----
    if threatens:
        score += 500 * _phase_pressure_multiplier(phase)

    # ----- Enemy escape pressure -----
    score += (
        escape_pressure
        * ESCAPE_PRESSURE_SCORE_MULTIPLIER
        * _phase_pressure_multiplier(phase)
    )

    if phase in ("anti_draw", "duel") and escape_future_score >= -100:
        if escape_pressure >= 500:
            score += ANTI_DRAW_BOMB_PRESSURE_BONUS
        if escape_pressure >= 900:
            score += HIGH_BOMB_PRESSURE_BONUS

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
                score += 150 * _phase_pressure_multiplier(phase)
                break

    # ----- Penalise valueless bombs -----
    if boxes == 0 and not threatens:
        score -= 200

    return score


# ==============================================================================
# Agent class
# ==============================================================================

class Agent:
    """
    Hybrid Bomberland agent combining:
      - Danger-aware BFS pathfinding
      - Heuristic action scoring (safety, items, boxes, enemies)
      - Escape verification before every bomb placement

    No ML, no network, no disk I/O.  Under 100 ms per act().
    """
    team_id = "HybridAgent"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step_count = 0

    def _get_pressure_phase(self, obs):
        alive_enemies = len(_alive_enemies(obs, self.agent_id))
        if alive_enemies <= 1:
            return "duel"
        if self.step_count >= 430:
            return "anti_draw"
        if self.step_count >= 330:
            return "late_pressure"
        return "normal"

    def act(self, obs: dict) -> int:
        self._future_cache = {}
        self.step_count += 1
        game_map = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        # ---- 1. State Parser ----
        my_r, my_c, alive, bombs_left, bonus = _my_state(obs, self.agent_id)
        if not alive:
            return A_STOP

        my_pos = (my_r, my_c)
        my_state = (my_r, my_c, alive, bombs_left, bonus)
        bomb_set = _bomb_set(bombs)
        enemies = _alive_enemies(obs, self.agent_id)
        phase = self._get_pressure_phase(obs)

        # ---- 2. Danger Map ----
        danger_time = _compute_danger_map(game_map, players, bombs)

        # ---- 3. Pre-compute shared data ----
        item_targets = _item_targets(game_map)
        box_spots = _box_bomb_spots(game_map, bomb_set)

        # ---- 4. Emergency escape ----
        curr_dt = danger_time[my_r, my_c]
        if curr_dt <= 3:
            safe = _find_nearest_safe(my_pos, game_map, bomb_set,
                                      danger_time, MAX_BFS_SAFE)
            if safe is not None:
                return safe[0]  # first_action

        bomb_escape_pressure = None
        if bombs_left > 0 and my_pos not in bomb_set:
            if _can_escape_after_bomb(obs, self.agent_id, game_map, bomb_set):
                bomb_escape_pressure = _estimate_enemy_escape_pressure(
                    obs, my_pos, self.agent_id
                )

        # ---- 5. Score all actions ----
        best_action = A_STOP
        best_score = -1e9

        for action in ALL_ACTIONS:
            if action == A_BOMB:
                score = _score_bomb(my_pos, my_state, game_map, players,
                                    bomb_set, danger_time, obs,
                                    self.agent_id, bomb_escape_pressure, phase)
                pos = my_pos
            else:
                score, pos = _score_move(action, my_pos, my_state, game_map,
                                         players, bomb_set, danger_time,
                                         item_targets, box_spots, enemies, obs,
                                         self.agent_id, self._future_cache,
                                         phase)

            if score > best_score:
                best_score = score
                best_action = action
                best_pos = pos

        # ---- 6. Bomb hysteresis ----
        if best_action == A_BOMB:
            best_move_score = -1e9
            for action in MOVE_ACTIONS:
                score, _ = _score_move(action, my_pos, my_state, game_map,
                                       players, bomb_set, danger_time,
                                       item_targets, box_spots, enemies, obs,
                                       self.agent_id, self._future_cache,
                                       phase)
                if score > best_move_score:
                    best_move_score = score
            pressure = 0 if bomb_escape_pressure is None else bomb_escape_pressure
            threshold = BOMB_HYSTERESIS_TUNED_THRESHOLD
            if len(enemies) == 1 or pressure >= 900:
                threshold = 0
            elif pressure >= 500:
                threshold = 80
            if phase == "late_pressure":
                threshold -= 20
            elif phase == "anti_draw":
                threshold -= 40
            elif phase == "duel":
                threshold -= 50
            threshold = max(0, threshold)
            if best_score < best_move_score + threshold:
                # Fall back to best move
                best_action = A_STOP
                best_score = -1e9
                for action in MOVE_ACTIONS:
                    score, pos = _score_move(action, my_pos, my_state, game_map,
                                             players, bomb_set, danger_time,
                                             item_targets, box_spots, enemies,
                                             obs, self.agent_id,
                                             self._future_cache, phase)
                    if score > best_score:
                        best_score = score
                        best_action = action

        return best_action
