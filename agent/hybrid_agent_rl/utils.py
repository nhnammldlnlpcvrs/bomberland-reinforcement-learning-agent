import numpy as np

from constants import (
    BLAST_DIRS,
    BOARD_SIZE,
    TILE_BOX,
    TILE_CAPACITY,
    TILE_GRASS,
    TILE_RADIUS,
    TILE_WALL,
)


def in_bounds(r, c):
    return 0 < r < BOARD_SIZE - 1 and 0 < c < BOARD_SIZE - 1


def my_state(obs, agent_id):
    p = obs["players"][agent_id]
    return int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4])


def alive_enemies(obs, agent_id):
    enemies = []
    for i, p in enumerate(obs["players"]):
        if i != agent_id and int(p[2]) == 1:
            enemies.append((int(p[0]), int(p[1]), i))
    return enemies


def bomb_set(bombs):
    arr = np.asarray(bombs, dtype=np.int32)
    if arr.size == 0:
        return set()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return {(int(row[0]), int(row[1])) for row in arr}


def is_walkable_tile(tile):
    return int(tile) in (TILE_GRASS, TILE_RADIUS, TILE_CAPACITY)


def is_passable(r, c, game_map, bombs):
    if not in_bounds(r, c):
        return False
    if not is_walkable_tile(game_map[r, c]):
        return False
    return (r, c) not in bombs


def free_neighbors(pos, game_map, bombs):
    r, c = pos
    total = 0
    for dr, dc in BLAST_DIRS:
        if is_passable(r + dr, c + dc, game_map, bombs):
            total += 1
    return total


def dead_end_level(pos, game_map, bombs):
    free = free_neighbors(pos, game_map, bombs)
    if free <= 1:
        return "dead_end"
    if free == 2:
        return "corridor"
    return "open"


def blast_cells(r, c, radius, game_map):
    cells = {(r, c)}
    for dr, dc in BLAST_DIRS:
        for dist in range(1, radius + 1):
            nr, nc = r + dr * dist, c + dc * dist
            if not in_bounds(nr, nc):
                break
            if int(game_map[nr, nc]) == TILE_WALL:
                break
            cells.add((nr, nc))
            if int(game_map[nr, nc]) == TILE_BOX:
                break
    return cells


def boxes_in_blast(pos, radius, game_map):
    return sum(1 for r, c in blast_cells(pos[0], pos[1], radius, game_map)
               if int(game_map[r, c]) == TILE_BOX)


def enemies_in_blast(pos, radius, game_map, players, agent_id):
    blast = blast_cells(pos[0], pos[1], radius, game_map)
    hits = []
    for i, p in enumerate(players):
        if i == agent_id or int(p[2]) != 1:
            continue
        enemy_pos = (int(p[0]), int(p[1]))
        if enemy_pos in blast:
            hits.append(i)
    return hits


def item_targets(game_map):
    targets = set()
    for r in range(1, BOARD_SIZE - 1):
        for c in range(1, BOARD_SIZE - 1):
            if int(game_map[r, c]) in (TILE_RADIUS, TILE_CAPACITY):
                targets.add((r, c))
    return targets


def box_bomb_spots(game_map, bombs):
    spots = set()
    for r in range(1, BOARD_SIZE - 1):
        for c in range(1, BOARD_SIZE - 1):
            if int(game_map[r, c]) != TILE_BOX:
                continue
            for dr, dc in BLAST_DIRS:
                nr, nc = r + dr, c + dc
                if is_passable(nr, nc, game_map, bombs):
                    spots.add((nr, nc))
    return spots
