"""
Self-Play PPO Trainer for Bomberland Master Agent.

Runs multi-agent rollouts where the training agent plays against historical
checkpoints and rule-based baselines from the agent pool.
"""

import copy
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Add project root so we can import agent modules
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from train_models.config import (
    ACTION_SPACE,
    AGENT_POOL_DIR,
    A_BOMB,
    A_STOP,
    AUX_CHECKPOINT_PATH,
    BOARD_SIZE,
    CHECKPOINT_DIR,
    DEVICE,
    EVAL_INTERVAL,
    GAMMA,
    LOG_DIR,
    LOG_INTERVAL,
    MAX_STEPS,
    NUM_AGENTS,
    POOL_INITIAL_AGENTS,
    POOL_MAX_SIZE,
    REWARD_BOMB_BOX_HIT,
    REWARD_BOMB_ENEMY_THREAT,
    REWARD_BOMB_HOARDING,
    REWARD_BOMB_PLACED,
    REWARD_BOX_DESTROYED,
    REWARD_CENTER_CONTROL,
    REWARD_DANGER_ZONE,
    REWARD_DEATH,
    REWARD_ENTER_NEW_CELL,
    REWARD_ESCAPE_MARGIN_HIGH,
    REWARD_ESCAPE_MARGIN_LOW,
    REWARD_ITEM_COLLECTED,
    REWARD_KILL,
    REWARD_LATEGAME_PROXIMITY,
    REWARD_LATEGAME_SURVIVAL,
    REWARD_LIVING,
    REWARD_LOOP_PENALTY,
    REWARD_OWN_BOMB_DEATH,
    REWARD_REACHABLE_INCREASE,
    REWARD_REVISIT_PENALTY,
    REWARD_STOP_PENALTY,
    REWARD_WIN,
    ROLLOUT_STEPS,
    SAVE_INTERVAL,
    SELF_PLAY_UPDATE_INTERVAL,
    STATE_CHANNELS_V2,
    TILE_BOX,
    TILE_CAPACITY,
    TILE_GRASS,
    TILE_RADIUS,
    TILE_WALL,
    TOTAL_TIMESTEPS,
    UPDATE_EPOCHS,
    ensure_dirs,
)
from train_models.model import ActorCritic
from train_models.ppo_agent import PPOAgent, RolloutBuffer
from train_models.state_processor import (
    encode_observation,
    encode_observation_v2,
    get_action_mask,
)

# ── Opponent agent loading ────────────────────────────────────────────────────


# Mapping from POOL_INITIAL_AGENTS names to actual module paths
_RULE_AGENT_MODULE_MAP = {
    "TacticalRuleAgent": "agent.tactical_rule_agent",
    "SmarterRuleAgent": "agent.smarter_rule_agent",
    "GeniusRuleAgent": "agent.genius_rule_agent",
    "SimpleRuleAgent": "agent.simple_rule_agent",
    "BoxFarmerAgent": "agent.box_farmer_agent",
}


class _RuleAgentWrapper:
    """Wraps a rule-based agent for use as a self-play opponent."""

    def __init__(self, name: str, agent_id: int):
        self.agent_id = agent_id
        import importlib
        module_path = _RULE_AGENT_MODULE_MAP.get(name, f"agent.{name}")
        # Try directory-style import first (agent/name/agent.py)
        try:
            mod = importlib.import_module(f"{module_path}.agent")
        except ModuleNotFoundError:
            # Flat-file import (agent/name.py)
            mod = importlib.import_module(module_path)
        self._agent = mod.Agent(agent_id)

    def act(self, obs: dict) -> int:
        return self._agent.act(obs)


class _PPOOpponent:
    """Lightweight inference-only PPO opponent loaded from a checkpoint."""

    def __init__(self, checkpoint_path: str, agent_id: int):
        self.agent_id = agent_id
        self.device = DEVICE
        self.model = ActorCritic().to(self.device)
        self.model.eval()
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)

    @torch.no_grad()
    def act(self, obs: dict) -> int:
        state_t, scalar_t = encode_observation_v2(obs, self.agent_id)
        mask = torch.from_numpy(get_action_mask(obs, self.agent_id)).unsqueeze(0).to(self.device)
        obs_b = state_t.unsqueeze(0).to(self.device)
        scal_b = scalar_t.unsqueeze(0).to(self.device)
        action, _, _ = self.model.get_action(obs_b, scal_b, action_mask=mask, deterministic=False)
        return int(action)


# ── Agent pool ─────────────────────────────────────────────────────────────────


class AgentPool:
    """Manages historical checkpoints and rule-based agents for self-play."""

    def __init__(self, pool_dir: Path):
        self.pool_dir = pool_dir
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.entries: list = []  # each entry: {"type": "rule"|"checkpoint", "name": str, "path": str|None}

        # Seed with rule-based agents
        for name in POOL_INITIAL_AGENTS:
            self.entries.append({"type": "rule", "name": name, "path": None})

    def add_checkpoint(self, checkpoint_path: str):
        """Add a training checkpoint to the pool."""
        name = Path(checkpoint_path).stem
        # Avoid duplicates
        if not any(e.get("path") == checkpoint_path for e in self.entries):
            self.entries.append({"type": "checkpoint", "name": name, "path": checkpoint_path})
        # Trim pool if too large (keep the rule agents)
        while len(self.entries) > POOL_MAX_SIZE + len(POOL_INITIAL_AGENTS):
            # Remove oldest checkpoint entry
            for i, e in enumerate(self.entries):
                if e["type"] == "checkpoint":
                    self.entries.pop(i)
                    break

    def sample_opponents(self, n: int, exclude_agent_id: int) -> list:
        """Sample n opponent agents randomly from the pool."""
        if len(self.entries) == 0:
            # Fallback: use random actions
            return [_RandomAgent(i) for i in range(exclude_agent_id + 1, exclude_agent_id + 1 + n)]

        sampled = random.choices(self.entries, k=n)
        opponents = []
        for i, entry in enumerate(sampled):
            agent_id = (exclude_agent_id + 1 + i) % NUM_AGENTS
            if entry["type"] == "rule":
                try:
                    opponents.append(_RuleAgentWrapper(entry["name"], agent_id))
                except Exception:
                    opponents.append(_RandomAgent(agent_id))
            elif entry["type"] == "checkpoint" and entry["path"]:
                try:
                    opponents.append(_PPOOpponent(entry["path"], agent_id))
                except Exception:
                    opponents.append(_RandomAgent(agent_id))
            else:
                opponents.append(_RandomAgent(agent_id))
        return opponents

    def get_agent(self, entry: dict, agent_id: int):
        """Get a single agent instance from a pool entry."""
        if entry["type"] == "rule":
            return _RuleAgentWrapper(entry["name"], agent_id)
        elif entry["type"] == "checkpoint" and entry["path"]:
            return _PPOOpponent(entry["path"], agent_id)
        return _RandomAgent(agent_id)


class _RandomAgent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id

    def act(self, obs: dict) -> int:
        mask = get_action_mask(obs, self.agent_id)
        valid = np.flatnonzero(mask)
        return int(np.random.choice(valid)) if len(valid) > 0 else 0


# ── Reward computation ─────────────────────────────────────────────────────────


def _count_boxes(game_map: np.ndarray) -> int:
    return int(np.sum(game_map == TILE_BOX))


def _count_items(game_map: np.ndarray) -> int:
    return int(np.sum((game_map == TILE_RADIUS) | (game_map == TILE_CAPACITY)))


def _detect_own_bomb_death(obs_before: dict, obs_after: dict, agent_id: int) -> bool:
    """Heuristic: did agent die from their own bomb?"""
    prev_p = obs_before["players"][agent_id]
    if not int(prev_p[2]):
        return False
    my_r, my_c = int(prev_p[0]), int(prev_p[1])
    bombs_arr = np.asarray(obs_after["bombs"], dtype=np.int32)
    if bombs_arr.size == 0:
        return False
    if bombs_arr.ndim == 1:
        bombs_arr = bombs_arr.reshape(1, -1)
    for i in range(bombs_arr.shape[0]):
        owner = int(bombs_arr[i, 3])
        if owner == agent_id:
            br, bc = int(bombs_arr[i, 0]), int(bombs_arr[i, 1])
            if abs(my_r - br) + abs(my_c - bc) <= 2:
                return True
    return False


def _count_boxes_in_blast(r: int, c: int, radius: int, game_map: np.ndarray) -> int:
    """Count boxes in blast lines from (r, c) with given radius."""
    count = 0
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for step in range(1, radius + 1):
            nr, nc = r + dr * step, c + dc * step
            if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                break
            if game_map[nr, nc] == TILE_WALL:
                break
            if game_map[nr, nc] == TILE_BOX:
                count += 1
                break
    return count


def _count_enemies_in_blast(r: int, c: int, radius: int, game_map: np.ndarray,
                             players: np.ndarray, agent_id: int) -> int:
    """Count enemy agents in blast lines from (r, c)."""
    count = 0
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for step in range(1, radius + 1):
            nr, nc = r + dr * step, c + dc * step
            if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                break
            if game_map[nr, nc] == TILE_WALL:
                break
            for i, p in enumerate(players):
                if i != agent_id and int(p[2]) == 1:
                    if int(p[0]) == nr and int(p[1]) == nc:
                        count += 1
            if game_map[nr, nc] == TILE_BOX:
                break
    return count


def _count_safe_neighbors(obs: dict, my_r: int, my_c: int) -> int:
    """Count adjacent cells that are passable and not in immediate danger."""
    from train_models.state_processor import _is_passable, _bomb_set, compute_danger_map
    game_map = np.asarray(obs["map"], dtype=np.int32)
    bset = _bomb_set(np.asarray(obs["bombs"], dtype=np.int32))
    danger = compute_danger_map(game_map, np.asarray(obs["players"], dtype=np.int32), obs["bombs"])
    count = 0
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = my_r + dr, my_c + dc
        if _is_passable(nr, nc, game_map, bset) and danger[nr, nc] > 2:
            count += 1
    return count


def _min_enemy_distance(players: np.ndarray, agent_id: int, my_r: int, my_c: int) -> int:
    """Manhattan distance to nearest alive enemy, or None."""
    best = None
    for i, p in enumerate(players):
        if i == agent_id or not int(p[2]):
            continue
        dist = abs(my_r - int(p[0])) + abs(my_c - int(p[1]))
        if best is None or dist < best:
            best = dist
    return best


def compute_reward(
    obs_before: dict,
    obs_after: dict,
    agent_id: int,
    action: int,
    terminated: bool,
    truncated: bool,
    position_history: list = None,
    visited_cells: set = None,
) -> tuple:
    """
    Compute dense shaped reward for a single agent step.

    Returns:
        reward: float
        info: dict with component breakdown
    """
    players_before = np.asarray(obs_before["players"], dtype=np.int32)
    players_after = np.asarray(obs_after["players"], dtype=np.int32)
    game_map_before = np.asarray(obs_before["map"], dtype=np.int32)
    game_map_after = np.asarray(obs_after["map"], dtype=np.int32)

    p_before = players_before[agent_id]
    p_after = players_after[agent_id]
    alive_before = int(p_before[2])
    alive_after = int(p_after[2])
    my_r, my_c = int(p_after[0]), int(p_after[1])
    prev_r, prev_c = int(p_before[0]), int(p_before[1])

    reward = 0.0
    info = {}

    # ── Terminal rewards ───────────────────────────────────────────────────

    if alive_before == 1 and alive_after == 0:
        # Own-bomb death detection
        if _detect_own_bomb_death(obs_before, obs_after, agent_id):
            reward += REWARD_OWN_BOMB_DEATH
            info["own_bomb_death"] = REWARD_OWN_BOMB_DEATH
        else:
            reward += REWARD_DEATH
            info["death"] = REWARD_DEATH
        info["total"] = reward
        return reward, info

    if terminated:
        total_alive = sum(1 for p in players_after if int(p[2]) == 1)
        if alive_after and total_alive == 1:
            r_win = REWARD_WIN
            reward += r_win
            info["win"] = r_win
            info["total"] = reward
            return reward, info

    if not alive_after:
        info["total"] = reward
        return reward, info

    # ── Survival (small, anti-early-suicide) ────────────────────────────────
    reward += REWARD_LIVING
    info["living"] = REWARD_LIVING

    # ── Box destruction ─────────────────────────────────────────────────────
    boxes_before = _count_boxes(game_map_before)
    boxes_after = _count_boxes(game_map_after)
    boxes_destroyed = boxes_before - boxes_after
    if boxes_destroyed > 0:
        r_box = min(REWARD_BOX_DESTROYED * boxes_destroyed, 6.0)
        reward += r_box
        info["box"] = r_box
    else:
        info["box"] = 0.0

    # ── Item collection ─────────────────────────────────────────────────────
    bonus_before = int(p_before[4])
    bonus_after = int(p_after[4])
    bombs_cap_before = int(p_before[3])
    bombs_cap_after = int(p_after[3])
    if bonus_after > bonus_before or bombs_cap_after > bombs_cap_before:
        r_item = REWARD_ITEM_COLLECTED
        reward += r_item
        info["item"] = r_item
    else:
        info["item"] = 0.0

    # ── Kills ───────────────────────────────────────────────────────────────
    enemies_before = sum(1 for i, p in enumerate(players_before) if i != agent_id and int(p[2]) == 1)
    enemies_after = sum(1 for i, p in enumerate(players_after) if i != agent_id and int(p[2]) == 1)
    kills = enemies_before - enemies_after
    if kills > 0:
        r_kill = REWARD_KILL * kills
        reward += r_kill
        info["kill"] = r_kill
    else:
        info["kill"] = 0.0

    # ── Bomb placement ──────────────────────────────────────────────────────
    if action == A_BOMB:
        radius = 1 + int(p_before[4])
        boxes_hit = _count_boxes_in_blast(my_r, my_c, radius, game_map_before)
        enemies_threatened = _count_enemies_in_blast(
            my_r, my_c, radius, game_map_before, players_before, agent_id
        )
        r_bomb = REWARD_BOMB_PLACED + REWARD_BOMB_BOX_HIT * boxes_hit + REWARD_BOMB_ENEMY_THREAT * enemies_threatened
        reward += r_bomb
        info["bomb_placed"] = r_bomb
    else:
        info["bomb_placed"] = 0.0

    # ── Bomb hoarding ───────────────────────────────────────────────────────
    if int(p_after[3]) > 0 and action != A_BOMB:
        safe_mask = get_action_mask(obs_after, agent_id)
        if safe_mask[A_BOMB]:
            reward += REWARD_BOMB_HOARDING
            info["bomb_hoarding"] = REWARD_BOMB_HOARDING

    # ── STOP penalty ────────────────────────────────────────────────────────
    if action == A_STOP:
        reward += REWARD_STOP_PENALTY
        info["stop_penalty"] = REWARD_STOP_PENALTY

    # ── Escape margin ───────────────────────────────────────────────────────
    safe_count = _count_safe_neighbors(obs_after, my_r, my_c)
    if safe_count <= 1:
        reward += REWARD_ESCAPE_MARGIN_LOW
        info["escape_margin"] = REWARD_ESCAPE_MARGIN_LOW
    elif safe_count >= 3:
        reward += REWARD_ESCAPE_MARGIN_HIGH
        info["escape_margin"] = REWARD_ESCAPE_MARGIN_HIGH
    else:
        info["escape_margin"] = 0.0

    # ── Loop / revisit detection ────────────────────────────────────────────
    current_pos = (my_r, my_c)
    if position_history is not None:
        position_history.append(current_pos)
        if len(position_history) >= 5:
            recent = position_history[-5:]
            unique = len(set(recent))
            if unique <= 2:
                reward += REWARD_LOOP_PENALTY
                info["loop_penalty"] = REWARD_LOOP_PENALTY
            if position_history.count(current_pos) >= 3:
                reward += REWARD_REVISIT_PENALTY
                info["revisit_penalty"] = REWARD_REVISIT_PENALTY

    # ── Exploration: enter new cell ─────────────────────────────────────────
    if visited_cells is not None and current_pos not in visited_cells:
        reward += REWARD_ENTER_NEW_CELL
        info["new_cell"] = REWARD_ENTER_NEW_CELL
        visited_cells.add(current_pos)

    # ── Late-game pressure ──────────────────────────────────────────────────
    step = int(obs_after.get("step", obs_after.get("current_step", 0)) or 0)
    progress = step / max(MAX_STEPS, 1)
    if progress > 0.6:
        alive_enemies = sum(
            1 for i, pp in enumerate(players_after)
            if i != agent_id and int(pp[2])
        )
        if alive_enemies > 0:
            enemy_dist = _min_enemy_distance(players_after, agent_id, my_r, my_c)
            reward += REWARD_LATEGAME_SURVIVAL
            info["lategame_survival"] = REWARD_LATEGAME_SURVIVAL
            if enemy_dist is not None and enemy_dist <= 5:
                reward += REWARD_LATEGAME_PROXIMITY
                info["lategame_proximity"] = REWARD_LATEGAME_PROXIMITY

    # ── Center control ──────────────────────────────────────────────────────
    center = (6, 6)
    center_dist = abs(my_r - center[0]) + abs(my_c - center[1])
    reward += REWARD_CENTER_CONTROL * (1.0 - center_dist / 12.0)
    info["center"] = REWARD_CENTER_CONTROL * (1.0 - center_dist / 12.0)

    info["total"] = reward
    return reward, info


# ── Game runner ────────────────────────────────────────────────────────────────


class GameRunner:
    """Runs a single BomberEnv instance with 4 agents."""

    def __init__(self, seed: int):
        # Lazy import to avoid circular dependencies
        from engine.game import BomberEnv
        self.env = BomberEnv(width=13, height=13, max_steps=MAX_STEPS, seed=seed)
        self.seed = seed

    def reset(self, seed: int = None):
        return self.env.reset(seed=seed)


# ── Trainer ────────────────────────────────────────────────────────────────────


class Trainer:
    """
    Self-play PPO trainer.

    Manages:
      - Multiple parallel BomberEnv instances
      - Rollout collection with opponent sampling
      - PPO policy updates
      - Checkpointing and agent-pool maintenance
      - TensorBoard logging
    """

    def __init__(
        self,
        num_envs: int = 4,
        rollout_steps: int = ROLLOUT_STEPS,
        learning_rate: float = 3e-4,
        seed: int = 42,
    ):
        ensure_dirs()

        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.seed = seed
        self.device = DEVICE

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Model & optimizer
        self.model = ActorCritic().to(self.device)
        self._warmstart_cnn_from_aux()
        self.agent = PPOAgent(
            model=self.model,
            lr=learning_rate,
            device=self.device,
        )

        # Agent pool for self-play
        self.pool = AgentPool(AGENT_POOL_DIR)

        # Logging
        self.writer = SummaryWriter(log_dir=str(LOG_DIR))
        self.global_step = 0
        self.total_env_steps = 0
        self.best_eval_winrate = 0.0

        # Environment runners
        self.runners = [GameRunner(seed=seed + i) for i in range(num_envs)]

        # Metrics tracking (must come before _reset_all_envs)
        self.episode_rewards = [[] for _ in range(num_envs)]
        self.episode_lengths = [0] * num_envs
        self.episode_wins = [0] * num_envs
        self.episode_count = 0

        # Initialize environments and opponent rosters
        self.env_obs = [None] * num_envs
        self.training_slots = [i % NUM_AGENTS for i in range(num_envs)]  # cycle through all 4 slots
        self.opponents = [None] * num_envs
        self._reset_all_envs()

    def _reset_all_envs(self):
        """Reset all environments and assign new opponents."""
        for i in range(self.num_envs):
            seed_i = self.seed + self.total_env_steps + i * 1000
            self.env_obs[i] = self.runners[i].reset(seed=seed_i)
            agent_id = self.training_slots[i]
            self.opponents[i] = self.pool.sample_opponents(3, agent_id)
            self.episode_rewards[i] = []
            self.episode_lengths[i] = 0
            self.episode_wins[i] = 0

    def _get_all_actions(self, env_idx: int, obs: dict) -> list:
        """Collect actions for all 4 agents in one environment."""
        training_slot = self.training_slots[env_idx]
        actions = [0] * NUM_AGENTS

        # Training agent
        train_action, train_logp, train_val, mask, obs_np, scal_np = self.agent.select_action(
            obs, training_slot, deterministic=False
        )
        actions[training_slot] = train_action

        # Opponent agents
        opps = self.opponents[env_idx]
        opp_idx = 0
        for i in range(NUM_AGENTS):
            if i == training_slot:
                continue
            if opp_idx < len(opps):
                try:
                    actions[i] = opps[opp_idx].act(obs)
                except Exception:
                    actions[i] = 0
            else:
                actions[i] = 0
            opp_idx += 1

        return actions, train_action, train_logp, train_val, mask, obs_np, scal_np

    def collect_rollout(self) -> bool:
        """
        Collect one full rollout (rollout_steps transitions across all envs).

        Returns True if any data was collected (buffer is ready for update).
        """
        buffer = RolloutBuffer(self.rollout_steps, num_envs=self.num_envs)

        # Track previous observations for reward computation
        prev_obs = [None] * self.num_envs
        # Per-env exploration state (reset on episode boundary)
        position_history = [[] for _ in range(self.num_envs)]
        visited_cells = [set() for _ in range(self.num_envs)]

        for step_idx in range(self.rollout_steps):
            actions_arr = np.zeros(self.num_envs, dtype=np.int64)
            values_arr = np.zeros(self.num_envs, dtype=np.float32)
            logps_arr = np.zeros(self.num_envs, dtype=np.float32)
            masks_arr = np.zeros((self.num_envs, ACTION_SPACE), dtype=bool)
            obs_arr = np.zeros((self.num_envs, STATE_CHANNELS_V2, 13, 13), dtype=np.float32)
            scal_arr = np.zeros((self.num_envs, 4), dtype=np.float32)
            rewards_arr = np.zeros(self.num_envs, dtype=np.float32)
            dones_arr = np.zeros(self.num_envs, dtype=np.float32)
            prev_obs_snapshots = [None] * self.num_envs

            for i in range(self.num_envs):
                obs = self.env_obs[i]
                prev_obs_snapshots[i] = copy.deepcopy(obs)

                all_actions, train_action, train_logp, train_val, mask, obs_np, scal_np = \
                    self._get_all_actions(i, obs)

                actions_arr[i] = train_action
                values_arr[i] = train_val
                logps_arr[i] = train_logp
                masks_arr[i] = mask
                obs_arr[i] = obs_np
                scal_arr[i] = scal_np

                # Step the environment
                next_obs, terminated, truncated = self.runners[i].env.step(all_actions)
                self.env_obs[i] = next_obs
                self.episode_lengths[i] += 1

                # Compute reward
                reward, r_info = compute_reward(
                    prev_obs_snapshots[i], next_obs,
                    self.training_slots[i], train_action,
                    terminated, truncated,
                    position_history[i],
                    visited_cells[i],
                )
                rewards_arr[i] = reward
                dones_arr[i] = float(terminated or truncated)

                self.episode_rewards[i].append(reward)

                # Handle episode end
                if terminated or truncated:
                    # Record win
                    players = np.asarray(next_obs["players"], dtype=np.int32)
                    my_alive = int(players[self.training_slots[i]][2])
                    total_alive = sum(1 for p in players if int(p[2]) == 1)
                    if my_alive and total_alive == 1:
                        self.episode_wins[i] += 1

                    # Reset env
                    seed_i = self.seed + self.total_env_steps + i * 1000
                    self.env_obs[i] = self.runners[i].reset(seed=seed_i)
                    self.opponents[i] = self.pool.sample_opponents(3, self.training_slots[i])

                    # Reset exploration state
                    position_history[i] = []
                    visited_cells[i] = set()

                    # Log episode
                    ep_reward = sum(self.episode_rewards[i])
                    ep_len = self.episode_lengths[i]
                    self.writer.add_scalar("episode/reward", ep_reward, self.episode_count)
                    self.writer.add_scalar("episode/length", ep_len, self.episode_count)
                    self.writer.add_scalar("episode/win", int(self.episode_wins[i] > 0), self.episode_count)
                    self.episode_count += 1
                    self.episode_rewards[i] = []
                    self.episode_lengths[i] = 0
                    self.episode_wins[i] = 0

            # Store in buffer
            buffer.add(obs_arr, scal_arr, actions_arr, rewards_arr, dones_arr, values_arr, logps_arr, masks_arr)
            self.total_env_steps += self.num_envs
            self.global_step += self.num_envs

            # Periodic logging
            if self.global_step % LOG_INTERVAL == 0:
                self.writer.add_scalar("train/global_step", self.global_step, self.global_step)

        # Compute last values for GAE bootstrapping
        next_values = np.zeros(self.num_envs, dtype=np.float32)
        next_dones = np.zeros(self.num_envs, dtype=np.float32)
        for i in range(self.num_envs):
            state_t, scalar_t = encode_observation_v2(self.env_obs[i], self.training_slots[i])
            obs_b = state_t.unsqueeze(0).to(self.device)
            scal_b = scalar_t.unsqueeze(0).to(self.device)
            next_values[i] = float(self.model.get_value(obs_b, scal_b).item())

        # PPO update
        metrics = self.agent.update(buffer, next_values, next_dones)

        self.writer.add_scalar("train/policy_loss", metrics["policy_loss"], self.global_step)
        self.writer.add_scalar("train/value_loss", metrics["value_loss"], self.global_step)
        self.writer.add_scalar("train/entropy", metrics["entropy"], self.global_step)
        self.writer.add_scalar("train/approx_kl", metrics["approx_kl"], self.global_step)

        self.writer.add_scalar("train/total_steps", self.total_env_steps, self.global_step)

        return True

    def _warmstart_cnn_from_aux(self):
        """
        Warm-start the ActorCritic CNN backbone from a pretrained auxiliary model.

        The aux model (BomberAuxModel) was trained on 19-channel observations with
        11 auxiliary prediction tasks (death, escape, box value, etc.). Its CNN
        layers learned useful spatial feature detectors.

        Since the aux model has different input channels (19 vs 16), we only copy
        layers 2 and 3 (32→64 and 64→64 Conv2d weights). The first layer is
        initialized randomly as usual. BatchNorm layers are left at their default
        (gamma=1, beta=0) since the aux model has none.
        """
        aux_path = AUX_CHECKPOINT_PATH
        if not aux_path.exists():
            print(f"[Trainer] Aux checkpoint not found at {aux_path} — skipping warm-start")
            return

        print(f"[Trainer] Warm-starting CNN from {aux_path}")
        try:
            aux_ckpt = torch.load(aux_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"[Trainer] Failed to load aux checkpoint: {e} — skipping warm-start")
            return

        # Extract aux encoder Conv2d weights
        # Aux encoder: Conv2d(19,32)→ReLU→Conv2d(32,64)→ReLU→Conv2d(64,64)→ReLU
        aux_state = aux_ckpt.get("model_state_dict", aux_ckpt)
        aux_conv_keys = [k for k in aux_state if "encoder" in k and "weight" in k and "conv" not in k]
        # Map: encoder.0.weight → Conv2d(19,32), encoder.2.weight → Conv2d(32,64), encoder.4.weight → Conv2d(64,64)

        # Our CNN: cnn.conv[0]=Conv2d(16,32), cnn.conv[3]=Conv2d(32,64), cnn.conv[6]=Conv2d(64,64)
        # Only copy layers 2 and 3 (32→64 and 64→64) — same dimensions

        mapping = [
            # (aux_key_pattern, our_param_name)
            # Skip encoder.0 (19→32 vs 16→32)
            ("encoder.2.weight", "cnn.conv.3.weight"),
            ("encoder.2.bias", "cnn.conv.3.bias"),
            ("encoder.4.weight", "cnn.conv.6.weight"),
            ("encoder.4.bias", "cnn.conv.6.bias"),
        ]

        copied = 0
        model_state = self.model.state_dict()
        for aux_key, our_key in mapping:
            if aux_key in aux_state and our_key in model_state:
                if aux_state[aux_key].shape == model_state[our_key].shape:
                    model_state[our_key] = aux_state[aux_key].clone()
                    copied += 1
                else:
                    print(f"[Trainer] Shape mismatch for {aux_key}: "
                          f"{aux_state[aux_key].shape} vs {model_state[our_key].shape}")
            else:
                print(f"[Trainer] Key not found: aux={aux_key} in aux={aux_key in aux_state}, "
                      f"our={our_key} in model={our_key in model_state}")

        self.model.load_state_dict(model_state)
        print(f"[Trainer] Warm-started {copied}/{len(mapping)} CNN parameters from aux checkpoint")

    def evaluate(self, num_episodes: int = 20) -> dict:
        """Evaluate the current policy against rule-based opponents."""
        from engine.game import BomberEnv

        wins = 0
        draws = 0
        losses = 0
        total_reward = 0.0
        steps_list = []

        for ep in range(num_episodes):
            env = BomberEnv(width=13, height=13, max_steps=MAX_STEPS, seed=self.seed + 10000 + ep)
            obs = env.reset()
            training_slot = ep % NUM_AGENTS
            opps = self.pool.sample_opponents(3, training_slot)
            ep_reward = 0.0

            for step_count in range(MAX_STEPS):
                prev_obs = copy.deepcopy(obs)
                actions = [0] * NUM_AGENTS

                # Training agent
                train_action, _, _, _, _, _ = self.agent.select_action(
                    obs, training_slot, deterministic=True
                )
                actions[training_slot] = train_action

                # Opponents
                opp_idx = 0
                for i in range(NUM_AGENTS):
                    if i == training_slot:
                        continue
                    if opp_idx < len(opps):
                        try:
                            actions[i] = opps[opp_idx].act(obs)
                        except Exception:
                            actions[i] = 0
                    opp_idx += 1

                next_obs, terminated, truncated = env.step(actions)
                reward, _ = compute_reward(prev_obs, next_obs, training_slot, train_action, terminated, truncated, None, None)
                ep_reward += reward
                obs = next_obs

                if terminated or truncated:
                    break

            total_reward += ep_reward
            steps_list.append(step_count + 1)

            players = np.asarray(obs["players"], dtype=np.int32)
            my_alive = int(players[training_slot][2])
            total_alive = sum(1 for p in players if int(p[2]) == 1)

            if my_alive and total_alive == 1:
                wins += 1
            elif my_alive and total_alive > 1:
                draws += 1
            else:
                losses += 1

        n = num_episodes
        return {
            "win_rate": wins / n,
            "draw_rate": draws / n,
            "loss_rate": losses / n,
            "avg_reward": total_reward / n,
            "avg_steps": np.mean(steps_list),
            "wins": wins,
            "draws": draws,
            "losses": losses,
        }

    def train(self, total_timesteps: int = TOTAL_TIMESTEPS):
        """Main training loop."""
        print(f"[Trainer] Starting training on {self.device}")
        print(f"[Trainer] Total timesteps: {total_timesteps:,}")
        print(f"[Trainer] Rollout steps per update: {self.rollout_steps}")
        print(f"[Trainer] Number of parallel envs: {self.num_envs}")
        print(f"[Trainer] Agent pool size: {len(self.pool.entries)}")

        start_time = time.time()
        last_save_step = 0
        last_eval_step = 0
        last_pool_step = 0

        try:
            while self.total_env_steps < total_timesteps:
                self.collect_rollout()

                # Periodic checkpoint
                if self.total_env_steps - last_save_step >= SAVE_INTERVAL:
                    ckpt_path = CHECKPOINT_DIR / f"model_step_{self.total_env_steps:08d}.pth"
                    self.agent.save(str(ckpt_path))
                    # Also save as latest
                    latest_path = CHECKPOINT_DIR / "latest.pth"
                    self.agent.save(str(latest_path))
                    last_save_step = self.total_env_steps
                    print(f"[Trainer] Saved checkpoint at step {self.total_env_steps:,} "
                          f"→ {ckpt_path.name}")

                # Periodic evaluation
                if self.total_env_steps - last_eval_step >= EVAL_INTERVAL:
                    print(f"[Trainer] Running evaluation at step {self.total_env_steps:,}...")
                    eval_metrics = self.evaluate(num_episodes=20)
                    last_eval_step = self.total_env_steps

                    self.writer.add_scalar("eval/win_rate", eval_metrics["win_rate"], self.global_step)
                    self.writer.add_scalar("eval/draw_rate", eval_metrics["draw_rate"], self.global_step)
                    self.writer.add_scalar("eval/avg_reward", eval_metrics["avg_reward"], self.global_step)
                    self.writer.add_scalar("eval/avg_steps", eval_metrics["avg_steps"], self.global_step)

                    print(f"[Trainer] Eval @ {self.total_env_steps:,}: "
                          f"win={eval_metrics['win_rate']:.2%} "
                          f"draw={eval_metrics['draw_rate']:.2%} "
                          f"loss={eval_metrics['loss_rate']:.2%} "
                          f"avg_reward={eval_metrics['avg_reward']:.2f} "
                          f"avg_steps={eval_metrics['avg_steps']:.0f}")

                    # Save best model
                    if eval_metrics["win_rate"] > self.best_eval_winrate:
                        self.best_eval_winrate = eval_metrics["win_rate"]
                        best_path = CHECKPOINT_DIR / "best_model.pth"
                        self.agent.save_weights_only(str(best_path))
                        print(f"[Trainer] New best model! win_rate={self.best_eval_winrate:.2%}")

                # Periodic agent-pool snapshot
                if self.total_env_steps - last_pool_step >= SELF_PLAY_UPDATE_INTERVAL:
                    pool_path = AGENT_POOL_DIR / f"pool_step_{self.total_env_steps:08d}.pth"
                    self.agent.save_weights_only(str(pool_path))
                    self.pool.add_checkpoint(str(pool_path))
                    last_pool_step = self.total_env_steps
                    print(f"[Trainer] Added checkpoint to pool (size={len(self.pool.entries)})")

                # Progress indicator
                if self.total_env_steps % (self.rollout_steps * 10) == 0:
                    elapsed = time.time() - start_time
                    fps = self.total_env_steps / max(elapsed, 1)
                    eta = (total_timesteps - self.total_env_steps) / max(fps, 1)
                    print(f"[Trainer] Step {self.total_env_steps:>10,} / {total_timesteps:,} "
                          f"({self.total_env_steps / total_timesteps * 100:.1f}%) "
                          f"FPS: {fps:.0f}  ETA: {eta:.0f}s")

        except KeyboardInterrupt:
            print("\n[Trainer] Interrupted. Saving final checkpoint...")

        # Final save
        final_path = CHECKPOINT_DIR / f"model_final_{self.total_env_steps:08d}.pth"
        self.agent.save(str(final_path))
        latest_path = CHECKPOINT_DIR / "latest.pth"
        self.agent.save(str(latest_path))
        self.agent.save_weights_only(str(CHECKPOINT_DIR / "best_model.pth"))

        self.writer.close()
        elapsed = time.time() - start_time
        print(f"[Trainer] Training complete. {self.total_env_steps:,} steps in {elapsed:.0f}s")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = Trainer(
        num_envs=4,
        rollout_steps=ROLLOUT_STEPS,
        learning_rate=3e-4,
        seed=42,
    )
    trainer.train(total_timesteps=TOTAL_TIMESTEPS)
