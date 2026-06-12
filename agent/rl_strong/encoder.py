from __future__ import annotations

import numpy as np

try:
    from .constants import (
        BOARD_SIZE,
        BOMB_TIMER,
        MAX_STEPS,
        N_CHANNELS,
        TILE_BOX,
        TILE_GRASS,
        TILE_ITEM_CAPACITY,
        TILE_ITEM_RADIUS,
        TILE_WALL,
    )
    from .action_mask import legal_action_mask
    from .utils import bomb_positions, compute_danger_map, normalize_obs, reachable_area
except ImportError:  # Loaded as a submission-local module.
    from constants import (
        BOARD_SIZE,
        BOMB_TIMER,
        MAX_STEPS,
        N_CHANNELS,
        TILE_BOX,
        TILE_GRASS,
        TILE_ITEM_CAPACITY,
        TILE_ITEM_RADIUS,
        TILE_WALL,
    )
    from action_mask import legal_action_mask
    from utils import bomb_positions, compute_danger_map, normalize_obs, reachable_area


def encode_observation(obs, agent_id):
    board, players, bombs, step = normalize_obs(obs)
    agent_id = max(0, min(int(agent_id), len(players) - 1))
    row, col = int(players[agent_id, 0]), int(players[agent_id, 1])
    bombs_set = bomb_positions(bombs)
    danger = compute_danger_map(board, players, bombs)

    planes = np.zeros((N_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    planes[0] = board == TILE_WALL
    planes[1] = board == TILE_BOX
    planes[2] = board == TILE_GRASS
    planes[3] = board == TILE_ITEM_RADIUS
    planes[4] = board == TILE_ITEM_CAPACITY

    if int(players[agent_id, 2]) and 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        planes[5, row, col] = 1.0

    alive_enemies = 0
    for idx, player in enumerate(players):
        prow, pcol, alive = int(player[0]), int(player[1]), int(player[2])
        if idx != agent_id and alive:
            alive_enemies += 1
            if 0 <= prow < BOARD_SIZE and 0 <= pcol < BOARD_SIZE:
                planes[6, prow, pcol] = 1.0
    planes[7].fill(alive_enemies / 3.0)

    for bomb in bombs:
        brow, bcol, timer, owner = [int(x) for x in bomb[:4]]
        if 0 <= brow < BOARD_SIZE and 0 <= bcol < BOARD_SIZE:
            planes[8, brow, bcol] = 1.0
            planes[9, brow, bcol] = max(0.0, min(1.0, timer / BOMB_TIMER))
            if owner == agent_id:
                planes[10, brow, bcol] = 1.0
            else:
                planes[11, brow, bcol] = 1.0

    planes[12] = danger <= 1
    planes[13] = danger <= 3
    planes[14] = danger <= BOMB_TIMER
    planes[15] = reachable_area(board, bombs_set, danger, (row, col), max_depth=12)

    mask = legal_action_mask(obs, agent_id)
    legal_cells = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    legal_cells[row, col] = 1.0 if mask[0] else 0.0
    for action, (dr, dc) in {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.items():
        if mask[action]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                legal_cells[nr, nc] = 1.0
    planes[16] = legal_cells

    center = (BOARD_SIZE - 1) / 2.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            dist = abs(r - center) + abs(c - center)
            planes[17, r, c] = 1.0 - dist / (BOARD_SIZE - 1)
    planes[18].fill(max(0.0, min(1.0, step / MAX_STEPS)))
    return planes.astype(np.float32)
