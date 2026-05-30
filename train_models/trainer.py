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
    REWARD_BOMB_PLACED,
    REWARD_BOX_DESTROYED,
    REWARD_DANGER_ZONE,
    REWARD_DEATH,
    REWARD_ITEM_COLLECTED,
    REWARD_KILL,
    REWARD_LIVING,
    REWARD_WIN,
    ROLLOUT_STEPS,
    SAVE_INTERVAL,
    SELF_PLAY_UPDATE_INTERVAL,
    TILE_BOX,
    TILE_CAPACITY,
    TILE_RADIUS,
    TOTAL_TIMESTEPS,
    UPDATE_EPOCHS,
    ensure_dirs,
)
from train_models.model import ActorCritic
from train_models.ppo_agent import PPOAgent, RolloutBuffer
from train_models.state_processor import (
    encode_observation,
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
        state_t, scalar_t = encode_observation(obs, self.agent_id)
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


def compute_reward(
    obs_before: dict,
    obs_after: dict,
    agent_id: int,
    action: int,
    terminated: bool,
    truncated: bool,
) -> tuple:
    """
    Compute shaped reward for a single agent step.

    Returns:
        reward: float
        info: dict with component breakdown
    """
    players_before = np.asarray(obs_before["players"], dtype=np.int32)
    players_after = np.asarray(obs_after["players"], dtype=np.int32)
    game_map_before = np.asarray(obs_before["map"], dtype=np.int32)
    game_map_after = np.asarray(obs_after["map"], dtype=np.int32)

    reward = 0.0
    info = {}

    # Living reward
    reward += REWARD_LIVING
    info["living"] = REWARD_LIVING

    # Box destroyed by anyone (rough proxy; training agent gets credit)
    boxes_before = _count_boxes(game_map_before)
    boxes_after = _count_boxes(game_map_after)
    boxes_destroyed = boxes_before - boxes_after
    if boxes_destroyed > 0:
        r_box = REWARD_BOX_DESTROYED * boxes_destroyed
        reward += r_box
        info["box"] = r_box

    # Item collected by training agent
    bonus_before = int(players_before[agent_id][4])
    bonus_after = int(players_after[agent_id][4])
    bombs_before = int(players_before[agent_id][3])
    bombs_after = int(players_after[agent_id][3])
    if bonus_after > bonus_before or bombs_after > bombs_before:
        r_item = REWARD_ITEM_COLLECTED
        reward += r_item
        info["item"] = r_item
    else:
        info["item"] = 0.0

    # Death
    alive_before = int(players_before[agent_id][2])
    alive_after = int(players_after[agent_id][2])
    if alive_before == 1 and alive_after == 0:
        reward += REWARD_DEATH
        info["death"] = REWARD_DEATH

    # Kill
    enemies_before = sum(1 for i, p in enumerate(players_before) if i != agent_id and int(p[2]) == 1)
    enemies_after = sum(1 for i, p in enumerate(players_after) if i != agent_id and int(p[2]) == 1)
    kills = enemies_before - enemies_after
    if kills > 0:
        r_kill = REWARD_KILL * kills
        reward += r_kill
        info["kill"] = r_kill
    else:
        info["kill"] = 0.0

    # Win: training agent is sole survivor
    if terminated:
        total_alive = sum(1 for p in players_after if int(p[2]) == 1)
        if alive_after and total_alive == 1:
            reward += REWARD_WIN
            info["win"] = REWARD_WIN
        else:
            info["win"] = 0.0

    # Bomb placed
    if action == A_BOMB:
        reward += REWARD_BOMB_PLACED
        info["bomb_placed"] = REWARD_BOMB_PLACED
    else:
        info["bomb_placed"] = 0.0

    # Danger zone penalty is applied in the agent step, not here

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

        for step_idx in range(self.rollout_steps):
            actions_arr = np.zeros(self.num_envs, dtype=np.int64)
            values_arr = np.zeros(self.num_envs, dtype=np.float32)
            logps_arr = np.zeros(self.num_envs, dtype=np.float32)
            masks_arr = np.zeros((self.num_envs, ACTION_SPACE), dtype=bool)
            obs_arr = np.zeros((self.num_envs, 7, 13, 13), dtype=np.float32)
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
            state_t, scalar_t = encode_observation(self.env_obs[i], self.training_slots[i])
            obs_b = state_t.unsqueeze(0).to(self.device)
            scal_b = scalar_t.unsqueeze(0).to(self.device)
            next_values[i] = float(self.model.get_value(obs_b, scal_b).item())
            # Check if current episode is done (env was just reset mid-rollout)
            next_dones[i] = 0.0  # We handle this in GAE via the dones we stored

        # PPO update
        metrics = self.agent.update(buffer, next_values, next_dones)

        self.writer.add_scalar("train/policy_loss", metrics["policy_loss"], self.global_step)
        self.writer.add_scalar("train/value_loss", metrics["value_loss"], self.global_step)
        self.writer.add_scalar("train/entropy", metrics["entropy"], self.global_step)
        self.writer.add_scalar("train/approx_kl", metrics["approx_kl"], self.global_step)

        self.writer.add_scalar("train/total_steps", self.total_env_steps, self.global_step)

        return True

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
                reward, _ = compute_reward(prev_obs, next_obs, training_slot, train_action, terminated, truncated)
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
