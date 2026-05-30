import numpy as np

from constants import (
    A_BOMB,
    A_STOP,
    ALL_ACTIONS,
    BLAST_DIRS,
    BOMB_BASE_BOX_BONUS,
    BOMB_BOX_WEIGHT,
    BOMB_DEAD_END_PENALTY,
    BOMB_ENEMY_HIT_WEIGHT,
    BOMB_ENEMY_PRESSURE_WEIGHT,
    BOMB_FRONTIER_WEIGHT,
    BOMB_LOW_SAFE_SPACE_PENALTY,
    BOMB_MULTI_BOX_WEIGHT,
    BOMB_NO_VALUE_PENALTY,
    BOMB_RECENT_AREA_PENALTY,
    BOMB_STUCK_BOX_BONUS,
    BOX_TARGET_BONUS,
    CENTER_CONTROL_BONUS,
    CORRIDOR_PENALTY,
    CURRENT_DANGER_ESCAPE_THRESHOLD,
    DEAD_END_PENALTY,
    DIRS,
    ENEMY_PRESSURE_TARGET_BONUS,
    ESCAPE_IMPROVEMENT_BONUS,
    FRONTIER_TARGET_BONUS,
    FUTURE_SURVIVAL_WEIGHT,
    INF,
    INVALID_SCORE,
    ITEM_TARGET_BONUS,
    MAX_BFS_ENEMY,
    MAX_BFS_ITEM,
    MAX_BFS_TARGET,
    MOBILITY_WEIGHT,
    NEAR_DANGER_PENALTY,
    OPEN_CELL_BONUS,
    RECENT_POSITION_PENALTY,
    SAFE_TILE_BONUS,
    STOP_BASE_PENALTY,
    STOP_IN_DANGER_PENALTY,
    STUCK_NEW_CELL_BONUS,
    STUCK_RECENT_POSITION_PENALTY,
    TENSOR_CENTER_WEIGHT,
    TENSOR_DANGER_WEIGHT,
    TENSOR_DEADEND_WEIGHT,
    TENSOR_FRONTIER_WEIGHT,
    TENSOR_ITEM_WEIGHT,
    TENSOR_MOBILITY_WEIGHT,
)
from encoder import encode
from safety import (
    bfs_to_targets,
    can_escape_after_bomb,
    compute_danger_map,
    enemy_escape_pressure,
    future_survivability,
    reachable_safe_count,
    simulate_bomb_danger,
)
from utils import (
    alive_enemies,
    blast_cells,
    bomb_set,
    box_bomb_spots,
    boxes_in_blast,
    dead_end_level,
    enemies_in_blast,
    free_neighbors,
    is_passable,
    item_targets,
    my_state,
)


def _frontier_value(pos, game_map):
    value = 0
    for dr, dc in BLAST_DIRS:
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < game_map.shape[0] and 0 <= nc < game_map.shape[1]:
            if int(game_map[nr, nc]) == 2:
                value += 1
    return value


def _frontier_targets(game_map, bombs):
    targets = set()
    for spot in box_bomb_spots(game_map, bombs):
        if _frontier_value(spot, game_map) > 0:
            targets.add(spot)
    return targets


def _center_targets(game_map, bombs, danger):
    targets = set()
    center = game_map.shape[0] // 2
    for r in range(1, game_map.shape[0] - 1):
        for c in range(1, game_map.shape[1] - 1):
            if not is_passable(r, c, game_map, bombs):
                continue
            if danger[r, c] != INF:
                continue
            if abs(r - center) + abs(c - center) <= 4:
                targets.add((r, c))
    return targets


def _enemy_pressure_targets(obs, agent_id, bombs, danger):
    game_map = obs["map"]
    players = obs["players"]
    targets = set()
    my_radius = 1 + int(players[agent_id][4])
    for er, ec, _eid in alive_enemies(obs, agent_id):
        for r in range(1, game_map.shape[0] - 1):
            for c in range(1, game_map.shape[1] - 1):
                if not is_passable(r, c, game_map, bombs):
                    continue
                if danger[r, c] != INF:
                    continue
                if (er, ec) in blast_cells(r, c, my_radius, game_map):
                    targets.add((r, c))
    return targets


def _target_bonus(start, targets, game_map, bombs, danger, max_depth, base_bonus):
    path = bfs_to_targets(start, targets, game_map, bombs, danger, max_depth)
    if path is None:
        return 0.0
    return max(0.0, base_bonus - 10.0 * path[1])


def _count_boxes_and_items(game_map):
    boxes = int((game_map == 2).sum())
    items = int(((game_map == 3) | (game_map == 4)).sum())
    return boxes, items


def _tensor_score_map(obs, agent_id):
    tensor = encode(obs, agent_id)
    item_map = np.maximum(tensor[2], tensor[3])
    danger_future = tensor[9]
    reachable = tensor[10]
    dead_end = tensor[11]
    frontier = tensor[12]
    safe_item = tensor[13]
    center_control = tensor[14]
    return (
        TENSOR_ITEM_WEIGHT * np.maximum(item_map, safe_item)
        + TENSOR_FRONTIER_WEIGHT * frontier
        + TENSOR_MOBILITY_WEIGHT * reachable
        + TENSOR_CENTER_WEIGHT * center_control
        - TENSOR_DANGER_WEIGHT * danger_future
        - TENSOR_DEADEND_WEIGHT * dead_end
    )


def score_actions(obs, agent_id, safe_actions=None, memory=None):
    game_map = obs["map"]
    players = obs["players"]
    r, c, alive, bombs_left, bonus = my_state(obs, agent_id)
    danger = compute_danger_map(game_map, players, obs["bombs"])
    mask = np.zeros(6, dtype=bool)
    if safe_actions is None:
        from safety import valid_action_mask
        mask = valid_action_mask(obs, agent_id, danger)
    else:
        for action in safe_actions:
            if 0 <= int(action) <= 5:
                mask[int(action)] = True
    scores = {a: INVALID_SCORE for a in ALL_ACTIONS}
    if not alive:
        scores[A_STOP] = 0.0
        return scores, mask, danger

    pos = (r, c)
    tensor_map = _tensor_score_map(obs, agent_id)
    bset = bomb_set(obs["bombs"])
    items = item_targets(game_map)
    box_spots = box_bomb_spots(game_map, bset)
    frontiers = _frontier_targets(game_map, bset)
    center_targets = _center_targets(game_map, bset, danger)
    pressure_targets = _enemy_pressure_targets(obs, agent_id, bset, danger)
    enemies = {(er, ec) for er, ec, _eid in alive_enemies(obs, agent_id)}
    stuck = memory.is_stuck() if memory is not None else False
    if memory is not None:
        box_count, item_count = _count_boxes_and_items(game_map)
        reach = reachable_safe_count(pos, game_map, bset, danger, 6)
        memory.observe_progress(box_count, item_count, reach)

    for action in ALL_ACTIONS:
        if not bool(mask[action]):
            continue
        if action == A_BOMB:
            if not can_escape_after_bomb(obs, agent_id):
                continue
            radius = 1 + bonus
            boxes = boxes_in_blast(pos, radius, game_map)
            hits = len(enemies_in_blast(pos, radius, game_map, players, agent_id))
            pressure = enemy_escape_pressure(obs, agent_id, pos)
            after_danger = simulate_bomb_danger(obs, agent_id, pos)
            blocked_after = set(bset)
            blocked_after.add(pos)
            safe_after = reachable_safe_count(pos, game_map, blocked_after, after_danger, 5)
            zone = dead_end_level(pos, game_map, bset)
            frontier = len({
                cell for cell in blast_cells(pos[0], pos[1], radius, game_map)
                if int(game_map[cell[0], cell[1]]) == 2
            })
            score = 0.0
            if boxes:
                score += BOMB_BASE_BOX_BONUS
            score += boxes * BOMB_BOX_WEIGHT
            score += max(0, boxes - 1) * BOMB_MULTI_BOX_WEIGHT
            score += frontier * BOMB_FRONTIER_WEIGHT
            score += hits * BOMB_ENEMY_HIT_WEIGHT
            score += pressure * BOMB_ENEMY_PRESSURE_WEIGHT
            if zone == "dead_end":
                score -= BOMB_DEAD_END_PENALTY
            if safe_after < 5:
                score -= BOMB_LOW_SAFE_SPACE_PENALTY
            if memory is not None and memory.recent_bomb_near(pos):
                score -= BOMB_RECENT_AREA_PENALTY
            if boxes == 0 and hits == 0 and pressure < 350:
                score -= BOMB_NO_VALUE_PENALTY
            if stuck and boxes > 0:
                score += BOMB_STUCK_BOX_BONUS
            scores[action] = score
            continue

        dr, dc = DIRS[action]
        npos = (r + dr, c + dc)
        if not is_passable(npos[0], npos[1], game_map, bset):
            continue
        dt = danger[npos[0], npos[1]]
        score = 0.0
        if dt == INF:
            score += SAFE_TILE_BONUS
        elif dt <= 3:
            score -= NEAR_DANGER_PENALTY
        if danger[r, c] <= CURRENT_DANGER_ESCAPE_THRESHOLD and dt > danger[r, c]:
            score += ESCAPE_IMPROVEMENT_BONUS

        mobility = free_neighbors(npos, game_map, bset)
        score += mobility * MOBILITY_WEIGHT
        zone = dead_end_level(npos, game_map, bset)
        if zone == "dead_end":
            score -= DEAD_END_PENALTY
        elif zone == "corridor":
            score -= CORRIDOR_PENALTY
        else:
            score += OPEN_CELL_BONUS
        score += future_survivability(npos, obs, danger) * FUTURE_SURVIVAL_WEIGHT
        score += float(tensor_map[npos[0], npos[1]])

        if game_map[npos[0], npos[1]] in (3, 4):
            score += 260
        score += _target_bonus(npos, items, game_map, bset, danger, MAX_BFS_ITEM, ITEM_TARGET_BONUS)
        score += _target_bonus(npos, box_spots, game_map, bset, danger, MAX_BFS_TARGET, BOX_TARGET_BONUS)
        score += _target_bonus(npos, frontiers, game_map, bset, danger, MAX_BFS_TARGET, FRONTIER_TARGET_BONUS)
        score += _target_bonus(npos, center_targets, game_map, bset, danger, MAX_BFS_TARGET, CENTER_CONTROL_BONUS)
        score += _target_bonus(npos, pressure_targets, game_map, bset, danger, MAX_BFS_ENEMY, ENEMY_PRESSURE_TARGET_BONUS)
        score += _target_bonus(npos, enemies, game_map, bset, danger, MAX_BFS_ENEMY, 70)

        if action == A_STOP:
            score -= STOP_BASE_PENALTY
            if danger[r, c] != INF:
                score -= STOP_IN_DANGER_PENALTY
        if memory is not None:
            repeats = memory.repeat_count(npos)
            if stuck:
                score += STUCK_NEW_CELL_BONUS if repeats == 0 else -STUCK_RECENT_POSITION_PENALTY * repeats
                score += mobility * 20
                score += _frontier_value(npos, game_map) * 120
            elif repeats:
                score -= RECENT_POSITION_PENALTY * repeats
        scores[action] = score

    return scores, mask, danger
