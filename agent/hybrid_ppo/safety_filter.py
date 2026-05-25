"""Safety filter for hybrid PPO agent.

Hard-gate safety invariants from online_robust. PPO action scores are
masked by this filter before final action selection — PPO never overrides
these rules.

Invariants enforced:
  1. danger_time[r][c] <= 1  → cell is lethal, action rejected
  2. Bomb placement requires verified escape path
  3. Invalid moves (walls, boxes, occupied-by-bomb) rejected
  4. Emergency escape overrides when agent is in danger (dt <= 3)
  5. Meaningless bombs (no boxes hit, no enemy threatened) rejected
"""

import numpy as np
from collections import deque

BOARD_SIZE = 13
INF = 9999
BOMB_TIMER = 7

TILE_WALL = 1
TILE_BOX = 2

MAX_BFS_ESCAPE = 8
MAX_BFS_SAFE = 12

DIRS = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
BLAST_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
MOVE_ACTIONS = [1, 2, 3, 4]
ALL_ACTIONS = [0, 1, 2, 3, 4, 5]


# ==============================================================================
# Helpers — bounds / passability
# ==============================================================================

def _in_bounds(r, c):
    return 0 < r < BOARD_SIZE - 1 and 0 < c < BOARD_SIZE - 1


def _bomb_set(bombs):
    arr = np.asarray(bombs, dtype=np.int32)
    if arr.size == 0:
        return set()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return {(int(arr[i, 0]), int(arr[i, 1])) for i in range(arr.shape[0])}


def _is_passable(r, c, game_map, bomb_positions):
    if not _in_bounds(r, c):
        return False
    if game_map[r, c] in (TILE_WALL, TILE_BOX):
        return False
    if (r, c) in bomb_positions:
        return False
    return True


# ==============================================================================
# Blast geometry
# ==============================================================================

def _blast_cells(r, c, radius, game_map):
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
    cnt = 0
    for br, bc in _blast_cells(r, c, radius, game_map):
        if game_map[br, bc] == TILE_BOX:
            cnt += 1
    return cnt


def _enemy_in_blast_line(r, c, radius, game_map, players, agent_id):
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

def compute_danger_map(game_map, players, bombs):
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

def _find_nearest_safe(start, game_map, bomb_set, danger_time, max_depth):
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


# ==============================================================================
# Bomb escape validation
# ==============================================================================

def can_escape_after_bomb(obs, agent_id, game_map, bomb_set):
    my_r = int(obs["players"][agent_id][0])
    my_c = int(obs["players"][agent_id][1])
    players = obs["players"]

    bombs_arr = np.asarray(obs["bombs"], dtype=np.int32)
    if bombs_arr.size == 0:
        new_bombs = np.array([[my_r, my_c, BOMB_TIMER, agent_id]], dtype=np.int32)
    else:
        if bombs_arr.ndim == 1:
            bombs_arr = bombs_arr.reshape(1, -1)
        new_bomb_row = np.array([[my_r, my_c, BOMB_TIMER, agent_id]], dtype=np.int32)
        new_bombs = np.vstack([bombs_arr, new_bomb_row])

    new_danger = compute_danger_map(game_map, players, new_bombs)
    result = _find_nearest_safe(
        (my_r, my_c), game_map, bomb_set, new_danger, MAX_BFS_ESCAPE
    )
    return result is not None


# ==============================================================================
# Public API — compute_safe_action_mask
# ==============================================================================

def compute_safe_action_mask(obs, agent_id):
    """Compute boolean mask of safe actions for PPO scoring.

    Returns bool[6] where True means the action is safe. PPO logits for
    unsafe actions are set to -inf before argmax — PPO NEVER selects them.

    Safety invariants (from online_robust):
      - Cells with danger_time <= 1 are lethal → movement rejected
      - Bomb requires verified escape path
      - Invalid moves (walls, boxes, bomb-occupied) rejected
      - Emergency escape overrides when danger_time <= 3
      - Meaningless bombs rejected (no boxes, no enemy threat)
    """
    game_map = obs["map"]
    players = obs["players"]
    bombs = obs["bombs"]

    my_r = int(players[agent_id][0])
    my_c = int(players[agent_id][1])
    alive = int(players[agent_id][2])

    if not alive:
        return np.array([True] + [False] * 5, dtype=bool)

    my_pos = (my_r, my_c)
    bomb_set = _bomb_set(bombs)
    danger_time = compute_danger_map(game_map, players, bombs)
    curr_dt = danger_time[my_r, my_c]

    mask = np.zeros(6, dtype=bool)

    # STOP: safe if current cell not about to explode
    if curr_dt > 1:
        mask[0] = True

    # Movement actions
    for a in MOVE_ACTIONS:
        dr, dc = DIRS[a]
        nr, nc = my_r + dr, my_c + dc
        if not _is_passable(nr, nc, game_map, bomb_set):
            continue
        if danger_time[nr, nc] <= 1:
            continue
        mask[a] = True

    # BOMB: validate escape + meaningful
    bombs_left = int(players[agent_id][3])
    bonus = int(players[agent_id][4])
    radius = 1 + bonus

    if bombs_left > 0 and my_pos not in bomb_set and curr_dt > 1:
        if can_escape_after_bomb(obs, agent_id, game_map, bomb_set):
            boxes = _boxes_in_blast(my_r, my_c, radius, game_map)
            threatens = _enemy_in_blast_line(
                my_r, my_c, radius, game_map, players, agent_id
            )
            if boxes > 0 or threatens:
                mask[5] = True

    # Emergency escape: if in danger, restrict to escape path only
    if curr_dt <= 3:
        safe_result = _find_nearest_safe(
            my_pos, game_map, bomb_set, danger_time, MAX_BFS_SAFE
        )
        if safe_result is not None:
            escape_action = safe_result[0]
            emergency_mask = np.zeros(6, dtype=bool)
            emergency_mask[escape_action] = True
            emergency_mask[0] = True  # STOP always fallback
            return emergency_mask

    # Guarantee at least STOP
    if not mask.any():
        mask[0] = True

    return mask
