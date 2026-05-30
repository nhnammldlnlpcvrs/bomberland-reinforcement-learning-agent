from __future__ import annotations

import numpy as np

try:
    from .constants import MOVE_ACTIONS, MOVE_DELTAS, NUM_ACTIONS, PLACE_BOMB, STOP
    from .utils import bomb_positions, normalize_obs, passable
except ImportError:  # Loaded as a submission-local module.
    from constants import MOVE_ACTIONS, MOVE_DELTAS, NUM_ACTIONS, PLACE_BOMB, STOP
    from utils import bomb_positions, normalize_obs, passable


def legal_action_mask(obs, agent_id):
    """Minimal legality mask only: invalid movement and unavailable bombs."""
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
