from collections import deque

import numpy as np

from constants import (
    A_BOMB,
    A_STOP,
    ACTION_ESCAPE_LOOKAHEAD,
    ALL_ACTIONS,
    BOMB_TIMER,
    CURRENT_DANGER_ESCAPE_THRESHOLD,
    DIRS,
    INF,
    INVALID_SCORE,
    MAX_BFS_ESCAPE,
    MAX_BFS_SAFE,
    MIN_SAFE_CELLS_AFTER_BOMB,
    MOVE_ACTIONS,
)
from utils import (
    blast_cells,
    bomb_set,
    boxes_in_blast,
    dead_end_level,
    enemies_in_blast,
    free_neighbors,
    is_passable,
    my_state,
)


def compute_danger_map(game_map, players, bombs):
    danger = np.full(game_map.shape, INF, dtype=np.int32)
    arr = np.asarray(bombs, dtype=np.int32)
    if arr.size == 0:
        return danger
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    blast_sets = []
    timers = []
    for br, bc, timer, owner in arr:
        owner = int(owner)
        radius = 1 + int(players[owner][4]) if 0 <= owner < len(players) else 1
        blast_sets.append(blast_cells(int(br), int(bc), radius, game_map))
        timers.append(max(0, int(timer)))

    effective = list(timers)
    for _ in range(len(effective)):
        changed = False
        for i, cells in enumerate(blast_sets):
            if effective[i] <= 0:
                continue
            for j, bomb in enumerate(arr):
                if i == j or effective[j] <= 0:
                    continue
                if (int(bomb[0]), int(bomb[1])) in cells and effective[i] < effective[j]:
                    effective[j] = effective[i]
                    changed = True
        if not changed:
            break

    for cells, timer in zip(blast_sets, effective):
        if timer <= 0:
            continue
        for r, c in cells:
            if timer < danger[r, c]:
                danger[r, c] = timer
    return danger


def bfs_nearest_safe(start, game_map, bombs, danger, max_depth=MAX_BFS_SAFE):
    q = deque([(start, None, 0)])
    visited = {start}
    while q:
        pos, first, dist = q.popleft()
        if dist > 0 and danger[pos[0], pos[1]] == INF:
            return first, dist, pos
        if dist >= max_depth:
            continue
        for action in MOVE_ACTIONS:
            dr, dc = DIRS[action]
            nxt = (pos[0] + dr, pos[1] + dc)
            if nxt in visited:
                continue
            if not is_passable(nxt[0], nxt[1], game_map, bombs):
                continue
            if danger[nxt[0], nxt[1]] <= dist + 2:
                continue
            visited.add(nxt)
            q.append((nxt, action if first is None else first, dist + 1))
    return None


def can_reach_safe_from(start, game_map, bombs, danger, max_depth=ACTION_ESCAPE_LOOKAHEAD):
    if danger[start[0], start[1]] == INF:
        return True
    return bfs_nearest_safe(start, game_map, bombs, danger, max_depth) is not None


def reachable_safe_count(start, game_map, bombs, danger, max_depth=5):
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
            dr, dc = DIRS[action]
            nxt = (pos[0] + dr, pos[1] + dc)
            if nxt in visited:
                continue
            if not is_passable(nxt[0], nxt[1], game_map, bombs):
                continue
            visited.add(nxt)
            q.append((nxt, dist + 1))
    return count


def bfs_to_targets(start, targets, game_map, bombs, danger, max_depth):
    if not targets:
        return None
    q = deque([(start, None, 0)])
    visited = {start}
    while q:
        pos, first, dist = q.popleft()
        if dist > 0 and pos in targets:
            return first, dist, pos
        if dist >= max_depth:
            continue
        for action in MOVE_ACTIONS:
            dr, dc = DIRS[action]
            nxt = (pos[0] + dr, pos[1] + dc)
            if nxt in visited:
                continue
            if not is_passable(nxt[0], nxt[1], game_map, bombs):
                continue
            if danger[nxt[0], nxt[1]] <= dist + 2:
                continue
            visited.add(nxt)
            q.append((nxt, action if first is None else first, dist + 1))
    return None


def simulate_bomb_danger(obs, agent_id, pos=None):
    r, c, _alive, _bombs_left, _bonus = my_state(obs, agent_id)
    if pos is None:
        pos = (r, c)
    bombs = np.asarray(obs["bombs"], dtype=np.int32)
    fake = np.array([[pos[0], pos[1], BOMB_TIMER, agent_id]], dtype=np.int32)
    if bombs.size == 0:
        new_bombs = fake
    else:
        if bombs.ndim == 1:
            bombs = bombs.reshape(1, -1)
        new_bombs = np.vstack([bombs, fake])
    return compute_danger_map(obs["map"], obs["players"], new_bombs)


def can_escape_after_bomb(obs, agent_id):
    r, c, _alive, bombs_left, _bonus = my_state(obs, agent_id)
    pos = (r, c)
    bombs = bomb_set(obs["bombs"])
    if bombs_left <= 0 or pos in bombs:
        return False
    danger = simulate_bomb_danger(obs, agent_id, pos)
    blocked_after_placement = set(bombs)
    blocked_after_placement.add(pos)
    escape = bfs_nearest_safe(
        pos, obs["map"], blocked_after_placement, danger, MAX_BFS_ESCAPE
    )
    if escape is None:
        return False
    return reachable_safe_count(
        escape[2], obs["map"], blocked_after_placement, danger, 4
    ) >= MIN_SAFE_CELLS_AFTER_BOMB


def future_survivability(pos, obs, danger, max_depth=5):
    bombs = bomb_set(obs["bombs"])
    q = deque([(pos, 0)])
    visited = {pos}
    safe_cells = 0
    branch_cells = 0
    deepest = 0
    while q:
        cell, dist = q.popleft()
        if danger[cell[0], cell[1]] <= dist + 1:
            continue
        safe_cells += 1
        deepest = max(deepest, dist)
        if free_neighbors(cell, obs["map"], bombs) >= 3:
            branch_cells += 1
        if dist >= max_depth:
            continue
        for action in MOVE_ACTIONS:
            dr, dc = DIRS[action]
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt in visited:
                continue
            if is_passable(nxt[0], nxt[1], obs["map"], bombs):
                visited.add(nxt)
                q.append((nxt, dist + 1))
    score = safe_cells * 18 + branch_cells * 35 + deepest * 20
    if safe_cells <= 2:
        score -= 250
    elif safe_cells <= 4:
        score -= 100
    if dead_end_level(pos, obs["map"], bombs) == "dead_end":
        score -= 140
    return max(-500, min(500, score))


def enemy_escape_pressure(obs, agent_id, bomb_pos):
    after = simulate_bomb_danger(obs, agent_id, bomb_pos)
    game_map = obs["map"]
    players = obs["players"]
    my_radius = 1 + int(players[agent_id][4])
    blast = blast_cells(bomb_pos[0], bomb_pos[1], my_radius, game_map)
    pressure = 0
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1:
            continue
        epos = (int(p[0]), int(p[1]))
        if epos in blast:
            pressure += 350
        escape = bfs_nearest_safe(epos, game_map, bomb_set(obs["bombs"]), after, 7)
        if escape is None:
            pressure += 500
        elif escape[1] >= 4:
            pressure += 180
    return min(pressure, 1200)


def valid_action_mask(obs, agent_id, danger=None):
    game_map = obs["map"]
    players = obs["players"]
    r, c, alive, bombs_left, bonus = my_state(obs, agent_id)
    bombs = bomb_set(obs["bombs"])
    mask = np.zeros(6, dtype=bool)
    if not alive:
        mask[0] = True
        return mask
    if danger is None:
        danger = compute_danger_map(game_map, players, obs["bombs"])
    move_candidates = []
    for action in ALL_ACTIONS:
        if action == A_BOMB:
            if bombs_left <= 0 or (r, c) in bombs:
                continue
            if danger[r, c] != INF:
                continue
            if not can_escape_after_bomb(obs, agent_id):
                continue
            radius = 1 + bonus
            value = boxes_in_blast((r, c), radius, game_map)
            value += len(enemies_in_blast((r, c), radius, game_map, players, agent_id))
            if value <= 0 and enemy_escape_pressure(obs, agent_id, (r, c)) < 350:
                continue
            mask[action] = True
            continue

        dr, dc = DIRS[action]
        nr, nc = r + dr, c + dc
        if not is_passable(nr, nc, game_map, bombs):
            continue
        if danger[nr, nc] <= 1:
            continue
        if danger[nr, nc] <= 2 and not (
            danger[r, c] <= 1 and danger[nr, nc] > danger[r, c]
        ):
            continue
        if not can_reach_safe_from((nr, nc), game_map, bombs, danger):
            continue
        mask[action] = True
        if action != A_STOP:
            move_candidates.append(action)
    if danger[r, c] <= CURRENT_DANGER_ESCAPE_THRESHOLD and move_candidates:
        mask[A_STOP] = False
    if not mask.any():
        mask[0] = True
    return mask


def get_safe_actions(obs, agent_id, danger=None):
    mask = valid_action_mask(obs, agent_id, danger)
    return [action for action in ALL_ACTIONS if bool(mask[action])]


def current_position_is_danger(obs, agent_id, danger=None):
    r, c, alive, _bombs_left, _bonus = my_state(obs, agent_id)
    if not alive:
        return False
    if danger is None:
        danger = compute_danger_map(obs["map"], obs["players"], obs["bombs"])
    return danger[r, c] <= CURRENT_DANGER_ESCAPE_THRESHOLD


def escape_action(obs, agent_id, memory=None, danger=None):
    del memory
    r, c, alive, _bombs_left, _bonus = my_state(obs, agent_id)
    if not alive:
        return A_STOP
    if danger is None:
        danger = compute_danger_map(obs["map"], obs["players"], obs["bombs"])
    bombs = bomb_set(obs["bombs"])
    result = bfs_nearest_safe((r, c), obs["map"], bombs, danger, MAX_BFS_SAFE)
    if result is not None:
        return int(result[0])
    mask = valid_action_mask(obs, agent_id, danger)
    best_action = A_STOP
    best_timer = -1
    for action in MOVE_ACTIONS:
        if not bool(mask[action]):
            continue
        dr, dc = DIRS[action]
        nr, nc = r + dr, c + dc
        timer = int(danger[nr, nc])
        if timer > best_timer:
            best_timer = timer
            best_action = action
    if best_action != A_STOP:
        return int(best_action)
    return A_STOP


def anti_suicide_scores(scores, mask):
    result = dict(scores)
    for action in ALL_ACTIONS:
        if not bool(mask[action]):
            result[action] = INVALID_SCORE
    return result
