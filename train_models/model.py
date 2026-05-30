"""
Lightweight CNN Actor-Critic with Action Masking for Bomberland.

Architecture:
  CNN backbone: 3 × Conv2d(3×3, stride=1, pad=1) + BatchNorm + ReLU
                channels 32 → 64 → 64, 13×13 spatial preserved
                input channels: 16 (v2) or 64 (temporal 4-frame stack)
  FC:           Flatten(64×13×13) + ScalarFeats(4) → Linear(128) → ReLU
  Actor head:   Linear(128 → 6) for action logits
  Critic head:  Linear(128 → 1) for state value
"""

from typing import Optional

import torch
import torch.nn as nn

from train_models.config import (
    ACTION_SPACE,
    CNN_CHANNELS,
    FC_HIDDEN,
    SCALAR_FEATURES,
    STATE_CHANNELS,
    STATE_CHANNELS_V2,
)


class BomberCNN(nn.Module):
    """Convolutional feature extractor backbone."""

    def __init__(
        self,
        in_channels: int = STATE_CHANNELS,
        channels: list = None,
    ):
        super().__init__()
        if channels is None:
            channels = CNN_CHANNELS

        layers = []
        prev = in_channels
        for c in channels:
            layers.extend([
                nn.Conv2d(prev, c, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ])
            prev = c
        self.conv = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ActorCritic(nn.Module):
    """
    Actor-Critic network for Bomberland PPO agent.

    Forward returns:
      - action_logits: (batch, 6) raw logits
      - value:         (batch, 1) state value estimate
    """

    def __init__(
        self,
        obs_channels: int = STATE_CHANNELS_V2,
        scalar_dim: int = SCALAR_FEATURES,
        cnn_channels: list = None,
        fc_hidden: int = FC_HIDDEN,
        action_space: int = ACTION_SPACE,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = CNN_CHANNELS

        self.cnn = BomberCNN(in_channels=obs_channels, channels=cnn_channels)

        cnn_out = cnn_channels[-1]  # 64
        # 13×13 spatial → flattened
        self.cnn_flat_dim = cnn_out * 13 * 13  # 64 * 169 = 10816

        self.shared_fc = nn.Sequential(
            nn.Linear(self.cnn_flat_dim + scalar_dim, fc_hidden),
            nn.ReLU(inplace=True),
        )

        self.actor = nn.Linear(fc_hidden, action_space)
        self.critic = nn.Linear(fc_hidden, 1)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for PPO stability."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)

        # Actor and critic heads get smaller init scale
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(
        self,
        obs: torch.Tensor,
        scalars: torch.Tensor,
    ) -> tuple:
        """
        Args:
            obs:     (B, C, 13, 13) spatial channels (16 v2, 64 temporal)
            scalars: (B, 4) scalar features

        Returns:
            action_logits: (B, 6)
            value:         (B, 1)
        """
        cnn_out = self.cnn(obs)                       # (B, 64, 13, 13)
        cnn_flat = cnn_out.flatten(start_dim=1)        # (B, 64*13*13)
        fused = torch.cat([cnn_flat, scalars], dim=1)  # (B, 10816+4)
        shared = self.shared_fc(fused)                  # (B, 128)
        logits = self.actor(shared)                     # (B, 6)
        value = self.critic(shared)                     # (B, 1)
        return logits, value

    def get_action(
        self,
        obs: torch.Tensor,
        scalars: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> tuple:
        """
        Sample an action with optional masking.

        Args:
            obs:         (1, C, 13, 13) or (B, C, 13, 13)
            scalars:     (1, 4) or (B, 4)
            action_mask: (1, 6) or (B, 6) bool tensor, True = legal
            deterministic: if True, argmax over legal actions

        Returns:
            action:  int or (B,) tensor of actions
            log_prob: (1,) or (B,) log probability of selected action
            value:    (1, 1) or (B, 1)
        """
        logits, value = self.forward(obs, scalars)

        if action_mask is not None:
            # Mask illegal actions to -inf before softmax
            masked_logits = logits.clone()
            masked_logits[~action_mask] = -1e9
        else:
            masked_logits = logits

        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        if action.numel() == 1:
            action = int(action.item())

        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        scalars: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Evaluate actions for PPO update.

        Returns:
            log_probs: (B,) log probs of taken actions
            values:    (B, 1) state values
            entropy:   (B,) policy entropy
        """
        logits, values = self.forward(obs, scalars)

        if action_mask is not None:
            masked_logits = logits.clone()
            masked_logits[~action_mask] = -1e9
        else:
            masked_logits = logits

        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy

    def get_value(self, obs: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """Return state-value estimate only."""
        _, value = self.forward(obs, scalars)
        return value
