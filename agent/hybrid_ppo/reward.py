"""Reward shaping for hybrid PPO training.

Dense, shaped rewards addressing the credit-assignment problem that caused
prior PPO attempts to fail. Every step produces a signal, but survival reward
is kept small to avoid passive camping.

Design principles:
  - Survival reward: minimal, just enough to avoid early suicide
  - Conversion incentives: kills, box destruction, item pickup dominate
  - Safety penalties: own-bomb death, dead-end traps
  - Anti-passivity: loop penalty, STOP penalty, idle penalty
"""

import numpy as np


def compute_reward(obs, prev_obs, agent_id, action, done, step,
                   position_history=None, max_steps=500):
    """Compute shaped reward for one transition.

    Args:
        obs: Current observation dict with 'map', 'players', 'bombs'.
        prev_obs: Previous observation dict (None on first step).
        agent_id: Index of this agent in players array.
        action: The action taken this step (0-5).
        done: Whether the episode terminated this step.
        step: Current step count in the episode.
        position_history: Optional list of recent (row, col) positions
            for loop detection.
        max_steps: Maximum steps per episode for progress normalization.

    Returns:
        float reward value.
    """
    p = obs["players"][agent_id]
    alive = bool(int(p[2]))
    my_r, my_c = int(p[0]), int(p[1])

    # ---- Terminal rewards (highest magnitude) ----

    if not alive:
        # Detect own-bomb death
        if prev_obs is not None and _detect_own_bomb_death(obs, prev_obs, agent_id):
            return -25.0
        return -15.0  # death by enemy / environment

    if done:
        alive_players = [i for i in range(4) if int(obs["players"][i][2])]
        if len(alive_players) == 1 and alive_players[0] == agent_id:
            return 50.0  # win
        # Draw/timeout: neutral, survival reward accumulated along the way
        return 0.0

    reward = 0.0

    # ---- Survival (small — just enough to avoid early suicide) ----
    reward += 0.03

    # ---- Delta-based rewards (require prev_obs) ----
    if prev_obs is not None:
        prev_p = prev_obs["players"][agent_id]

        # Item collection
        if int(p[3]) > int(prev_p[3]):
            reward += 3.0  # bombs_left increased
        if int(p[4]) > int(prev_p[4]):
            reward += 3.0  # bomb_radius_bonus increased

        # Box destruction near agent (blast radius ~3)
        reward += _box_destruction_bonus(obs, prev_obs, my_r, my_c)

        # Enemy kills
        prev_alive = sum(1 for pp in prev_obs["players"] if int(pp[2]))
        curr_alive = sum(1 for pp in obs["players"] if int(pp[2]))
        killed = prev_alive - curr_alive
        if killed > 0:
            reward += 12.0 * killed

    # ---- Action-specific ----
    if action == 0:  # STOP
        reward -= 0.05  # slight bias toward movement
    elif action == 5:  # BOMB
        # Small bonus for using bomb (only if it was meaningful — will be
        # offset by negative if bomb was useless, detected via outcome)
        reward += 0.1

    # ---- Escape margin (safe cells near agent) ----
    safe_count = _count_safe_neighbors(obs, my_r, my_c)
    if safe_count <= 1:
        reward -= 0.5  # dangerous position
    elif safe_count >= 3:
        reward += 0.05  # comfortable position

    # ---- Loop / passivity detection ----
    if position_history is not None and len(position_history) >= 4:
        recent = position_history[-4:]
        unique = len(set(recent))
        if unique <= 2:
            reward -= 0.3  # looping
        if position_history.count((my_r, my_c)) >= 3:
            reward -= 0.2  # revisiting same cell

    # ---- Late-game pressure ----
    progress = step / max(max_steps, 1)
    if progress > 0.6:
        alive_enemies = sum(
            1 for i, pp in enumerate(obs["players"])
            if i != agent_id and int(pp[2])
        )
        if alive_enemies > 0:
            # Small bonus for staying alive + near enemies in endgame
            enemy_dist = _min_enemy_distance(obs, agent_id, my_r, my_c)
            if enemy_dist is not None and enemy_dist <= 5:
                reward += 0.05  # applying pressure
            reward += 0.02  # endgame survival

    # ---- Center control (very small) ----
    center = (6, 6)
    center_dist = abs(my_r - center[0]) + abs(my_c - center[1])
    reward += 0.01 * (1.0 - center_dist / 12.0)

    return float(reward)


# ==============================================================================
# Helper functions
# ==============================================================================

def _detect_own_bomb_death(obs, prev_obs, agent_id):
    """Heuristic: did agent die from their own bomb?"""
    prev_p = prev_obs["players"][agent_id]
    if not int(prev_p[2]):  # was already dead
        return False

    my_r, my_c = int(prev_p[0]), int(prev_p[1])

    bombs_arr = np.asarray(obs["bombs"], dtype=np.int32)
    if bombs_arr.size == 0:
        return False
    if bombs_arr.ndim == 1:
        bombs_arr = bombs_arr.reshape(1, -1)

    for i in range(bombs_arr.shape[0]):
        owner = int(bombs_arr[i, 3])
        if owner == agent_id:
            br, bc = int(bombs_arr[i, 0]), int(bombs_arr[i, 1])
            # Agent died within 2 cells of their own active bomb
            if abs(my_r - br) + abs(my_c - bc) <= 2:
                return True
    return False


def _box_destruction_bonus(obs, prev_obs, my_r, my_c):
    """Return bonus for boxes destroyed near agent's position."""
    prev_map = prev_obs["map"]
    curr_map = obs["map"]
    bonus = 0.0
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            nr, nc = my_r + dr, my_c + dc
            if 0 <= nr < 13 and 0 <= nc < 13:
                if prev_map[nr, nc] == 2 and curr_map[nr, nc] != 2:
                    bonus += 1.5
    # Cap to avoid over-rewarding lucky chain reactions
    return min(bonus, 6.0)


def _count_safe_neighbors(obs, my_r, my_c):
    """Count how many adjacent cells are passable and not in immediate danger."""
    from agent.hybrid_ppo.safety_filter import (
        compute_danger_map, _is_passable, _bomb_set,
    )
    game_map = obs["map"]
    bomb_set = _bomb_set(obs["bombs"])
    danger = compute_danger_map(game_map, obs["players"], obs["bombs"])
    count = 0
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = my_r + dr, my_c + dc
        if _is_passable(nr, nc, game_map, bomb_set) and danger[nr, nc] > 2:
            count += 1
    return count


def _min_enemy_distance(obs, agent_id, my_r, my_c):
    """Manhattan distance to nearest alive enemy, or None."""
    best = None
    for i, p in enumerate(obs["players"]):
        if i == agent_id or not int(p[2]):
            continue
        dist = abs(my_r - int(p[0])) + abs(my_c - int(p[1]))
        if best is None or dist < best:
            best = dist
    return best
