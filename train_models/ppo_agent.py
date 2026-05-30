"""
PPO Agent with GAE advantage estimation and clipped policy updates.

Provides:
  - RolloutBuffer: stores (obs, scalars, action, reward, done, value, log_prob, mask)
  - PPOAgent: orchestrates experience collection and policy updates
"""

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from train_models.config import (
    ACTION_SPACE,
    BATCH_SIZE,
    DEVICE,
    ENTROPY_COEF,
    GAE_LAMBDA,
    GAMMA,
    MAX_GRAD_NORM,
    PPO_CLIP,
    SCALAR_FEATURES,
    STATE_CHANNELS,
    UPDATE_EPOCHS,
    VALUE_COEF,
)
from train_models.state_processor import encode_observation, get_action_mask


class RolloutBuffer:
    """Fixed-size buffer for on-policy PPO rollouts."""

    def __init__(self, capacity: int, num_envs: int = 1):
        self.capacity = capacity
        self.num_envs = num_envs

        self.obs = np.zeros((capacity, num_envs, STATE_CHANNELS, 13, 13), dtype=np.float32)
        self.scalars = np.zeros((capacity, num_envs, SCALAR_FEATURES), dtype=np.float32)
        self.actions = np.zeros((capacity, num_envs), dtype=np.int64)
        self.rewards = np.zeros((capacity, num_envs), dtype=np.float32)
        self.dones = np.zeros((capacity, num_envs), dtype=np.float32)
        self.values = np.zeros((capacity, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((capacity, num_envs), dtype=np.float32)
        self.masks = np.zeros((capacity, num_envs, ACTION_SPACE), dtype=bool)

        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        scalars: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
        mask: np.ndarray,
    ):
        idx = self.ptr
        self.obs[idx] = obs
        self.scalars[idx] = scalars
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.masks[idx] = mask
        self.ptr += 1

    def __len__(self) -> int:
        return self.ptr

    def ready(self) -> bool:
        return self.ptr >= self.capacity

    def compute_gae(
        self,
        next_value: np.ndarray,
        next_done: np.ndarray,
    ) -> tuple:
        """
        Compute Generalized Advantage Estimation.

        Returns:
            returns: (T, N) discounted returns
            advantages: (T, N) GAE advantages
        """
        T, N = self.ptr, self.num_envs
        advantages = np.zeros((T, N), dtype=np.float32)
        gae = np.zeros(N, dtype=np.float32)

        for t in reversed(range(T)):
            done = self.dones[t]
            reward = self.rewards[t]
            mask = 1.0 - done

            if t == T - 1:
                next_val = next_value
                next_d = next_done
            else:
                next_val = self.values[t + 1]
                next_d = self.dones[t + 1]

            delta = reward + GAMMA * next_val * (1.0 - next_d) - self.values[t]
            gae = delta + GAMMA * GAE_LAMBDA * mask * gae
            advantages[t] = gae

        returns = advantages + self.values
        return returns, advantages

    def get_training_data(self) -> dict:
        """Flatten (T, N, ...) into (T*N, ...) for minibatch sampling."""
        T = self.ptr
        N = self.num_envs
        total = T * N
        return {
            "obs": self.obs[:T].reshape(total, STATE_CHANNELS, 13, 13),
            "scalars": self.scalars[:T].reshape(total, SCALAR_FEATURES),
            "actions": self.actions[:T].reshape(total),
            "log_probs": self.log_probs[:T].reshape(total),
            "masks": self.masks[:T].reshape(total, ACTION_SPACE),
        }


class PPOAgent:
    """
    Proximal Policy Optimization agent for Bomberland.

    Handles:
      - Experience collection from environment
      - GAE advantage computation
      - Clipped PPO policy/value updates
      - Model checkpoint save/load
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        clip_range: float = PPO_CLIP,
        ent_coef: float = ENTROPY_COEF,
        vf_coef: float = VALUE_COEF,
        max_grad_norm: float = MAX_GRAD_NORM,
        device: str = DEVICE,
    ):
        self.model = model.to(device)
        self.device = device
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

    @torch.no_grad()
    def select_action(
        self,
        obs: dict,
        agent_id: int,
        deterministic: bool = False,
    ) -> tuple:
        """
        Select an action for a single agent.

        Returns:
            action:  int 0–5
            log_prob: float
            value:   float
            mask:    (6,) bool array
            obs_tensor: (1, 7, 13, 13) for storing in buffer
            scalars_tensor: (1, 4)
        """
        state_tensor, scalar_tensor = encode_observation(obs, agent_id)
        mask_np = get_action_mask(obs, agent_id)
        mask = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)  # (1, 6)

        obs_batch = state_tensor.unsqueeze(0).to(self.device)   # (1, 7, 13, 13)
        scalars_batch = scalar_tensor.unsqueeze(0).to(self.device)  # (1, 4)

        action, log_prob, value = self.model.get_action(
            obs_batch, scalars_batch, action_mask=mask, deterministic=deterministic
        )

        return (
            int(action),
            float(log_prob.item()),
            float(value.item()),
            mask_np,
            state_tensor.numpy(),
            scalar_tensor.numpy(),
        )

    def update(self, buffer: RolloutBuffer, next_value: np.ndarray, next_done: np.ndarray) -> dict:
        """
        Perform one PPO update epoch over the rollout buffer.

        Returns:
            metrics dict with policy_loss, value_loss, entropy, approx_kl
        """
        returns, advantages = buffer.compute_gae(next_value, next_done)
        data = buffer.get_training_data()

        obs_tensor = torch.as_tensor(data["obs"], device=self.device)
        scalars_tensor = torch.as_tensor(data["scalars"], device=self.device)
        actions_tensor = torch.as_tensor(data["actions"], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(data["log_probs"], device=self.device)
        action_masks = torch.as_tensor(data["masks"], device=self.device)
        returns_tensor = torch.as_tensor(returns.flatten(), device=self.device)
        advantages_tensor = torch.as_tensor(advantages.flatten(), device=self.device)

        # Normalize advantages
        adv_mean = advantages_tensor.mean()
        adv_std = advantages_tensor.std()
        advantages_tensor = (advantages_tensor - adv_mean) / (adv_std + 1e-8)

        total_samples = obs_tensor.shape[0]
        indices = np.arange(total_samples)

        epoch_metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}

        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(indices)

            for start in range(0, total_samples, BATCH_SIZE):
                batch_idx = indices[start:start + BATCH_SIZE]

                batch_obs = obs_tensor[batch_idx]
                batch_scalars = scalars_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_masks = action_masks[batch_idx]
                batch_returns = returns_tensor[batch_idx]
                batch_advantages = advantages_tensor[batch_idx]

                # Evaluate current policy on old actions
                new_log_probs, values, entropy = self.model.evaluate_actions(
                    batch_obs, batch_scalars, batch_actions, action_mask=batch_masks
                )

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss — clipped PPO style
                # Reconstruct old values from returns and GAE advantages
                # returns = advantages + old_values → old_values = returns - advantages
                old_values = batch_returns - batch_advantages
                value_pred = values.squeeze(-1)
                value_pred_clipped = old_values + torch.clamp(
                    value_pred - old_values, -self.clip_range, self.clip_range
                )
                value_loss_unclipped = (value_pred - batch_returns).pow(2)
                value_loss_clipped = (value_pred_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Entropy bonus
                entropy_mean = entropy.mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Approximate KL
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (new_log_probs - batch_old_log_probs)).mean().item()

                epoch_metrics["policy_loss"].append(policy_loss.item())
                epoch_metrics["value_loss"].append(value_loss.item())
                epoch_metrics["entropy"].append(entropy_mean.item())
                epoch_metrics["approx_kl"].append(approx_kl)

        return {k: float(np.mean(v)) for k, v in epoch_metrics.items()}

    def save(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def save_weights_only(self, path: str):
        """Export only model weights for inference (smaller file)."""
        torch.save(self.model.state_dict(), path)
