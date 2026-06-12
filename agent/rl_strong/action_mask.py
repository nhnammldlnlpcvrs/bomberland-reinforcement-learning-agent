from __future__ import annotations

import numpy as np

try:
    from .constants import MOVE_ACTIONS, MOVE_DELTAS, NUM_ACTIONS, PLACE_BOMB, STOP, TILE_WALL, TILE_BOX
    from .utils import (
        BLAST_DELTAS,
        bomb_positions,
        boxes_in_blast,
        compute_danger_map,
        has_escape_after_bomb,
        normalize_obs,
        passable,
    )
except ImportError:  # Loaded as a submission-local module.
    from constants import MOVE_ACTIONS, MOVE_DELTAS, NUM_ACTIONS, PLACE_BOMB, STOP, TILE_WALL, TILE_BOX
    from utils import (
        BLAST_DELTAS,
        bomb_positions,
        boxes_in_blast,
        compute_danger_map,
        has_escape_after_bomb,
        normalize_obs,
        passable,
    )


def base_legal_action_mask(obs, agent_id):
    try:
        board, players, bombs, _step = normalize_obs(obs)
        agent_id = int(agent_id)
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        if agent_id < 0 or agent_id >= len(players) or not int(players[agent_id, 2]):
            mask[STOP] = True
            return mask

        row, col = int(players[agent_id, 0]), int(players[agent_id, 1])
        bombs_set = bomb_positions(bombs)
        mask[STOP] = True
        for action in MOVE_ACTIONS:
            dr, dc = MOVE_DELTAS[action]
            mask[action] = passable(board, bombs_set, row + dr, col + dc)
        mask[PLACE_BOMB] = int(players[agent_id, 3]) > 0 and (row, col) not in bombs_set
        return mask
    except Exception:
        return np.array([True, False, False, False, False, False], dtype=bool)


def _enemy_in_blast(board, players, row, col, agent_id):
    radius = 1 + max(0, int(players[agent_id, 4]))
    for enemy_id, enemy in enumerate(players):
        if enemy_id == agent_id or not int(enemy[2]):
            continue
        target = (int(enemy[0]), int(enemy[1]))
        if target == (row, col):
            return True
        for dr, dc in BLAST_DELTAS:
            for dist in range(1, radius + 1):
                nr, nc = row + dr * dist, col + dc * dist
                if not (0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]) or int(board[nr, nc]) == TILE_WALL:
                    break
                if (nr, nc) == target:
                    return True
                if int(board[nr, nc]) == TILE_BOX:
                    break
    return False


def legal_action_mask(obs, agent_id):
    """Safety-first mask used at inference.

    It removes illegal moves, one-step blast exposure, and bomb placements that
    have no escape route. If every non-bomb action is filtered out, it falls
    back to the engine-legal mask so the agent always returns a valid action.
    """
    legal = base_legal_action_mask(obs, agent_id)
    try:
        board, players, bombs, _step = normalize_obs(obs)
        agent_id = int(agent_id)
        if agent_id < 0 or agent_id >= len(players) or not int(players[agent_id, 2]):
            return legal

        row, col = int(players[agent_id, 0]), int(players[agent_id, 1])
        danger = compute_danger_map(board, players, bombs)
        safe = legal.copy()

        for action in (STOP, *MOVE_ACTIONS):
            if not legal[action]:
                continue
            dr, dc = MOVE_DELTAS[action]
            nr, nc = row + dr, col + dc
            if not (0 <= nr < danger.shape[0] and 0 <= nc < danger.shape[1]):
                safe[action] = False
                continue
            safe[action] = bool(danger[nr, nc] > 1)

        useful_bomb = boxes_in_blast(board, players, row, col, agent_id) > 0 or _enemy_in_blast(
            board, players, row, col, agent_id
        )
        safe[PLACE_BOMB] = bool(legal[PLACE_BOMB] and useful_bomb and has_escape_after_bomb(board, players, bombs, agent_id))

        if not safe[:PLACE_BOMB].any():
            safe[:PLACE_BOMB] = legal[:PLACE_BOMB]
        return safe
    except Exception:
        return legal


def sanitize_action(action, obs, agent_id):
    mask = legal_action_mask(obs, agent_id)
    try:
        value = int(action)
    except Exception:
        value = STOP
    if 0 <= value < NUM_ACTIONS and mask[value]:
        return value, False
    valid = np.flatnonzero(mask)
    return (int(valid[0]) if valid.size else STOP), True


def highest_prob_valid(probabilities, obs, agent_id):
    mask = legal_action_mask(obs, agent_id)
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if probs.shape[0] < NUM_ACTIONS:
        valid = np.flatnonzero(mask)
        return int(valid[0]) if valid.size else STOP
    masked = np.where(mask, probs[:NUM_ACTIONS], -np.inf)
    if not np.isfinite(masked).any():
        return STOP
    return int(np.argmax(masked))
