"""PPO Actor-Critic network for hybrid Bomberland agent.

Shared CNN backbone + policy head (6 actions) + value head (1).
CPU-friendly design: ~1.4M params, targets < 1ms inference.

Supports:
  - Safety action masking (logits set to -inf for unsafe actions)
  - Stochastic sampling (training rollouts)
  - Deterministic argmax (production inference)
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

from agent.hybrid_ppo.state_encoder import NUM_CHANNELS

BOARD_SIZE = 13
NUM_ACTIONS = 6

if TORCH_AVAILABLE:

    class PPOPolicy(nn.Module):
        """Actor-Critic with shared CNN backbone for 13x13 grid inputs."""

        def __init__(self, input_channels=NUM_CHANNELS, num_actions=NUM_ACTIONS,
                     hidden_dim=128):
            super().__init__()
            self.input_channels = int(input_channels)
            self.num_actions = int(num_actions)

            self.backbone = nn.Sequential(
                nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, hidden_dim),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(hidden_dim, num_actions)
            self.value_head = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            """Forward pass through backbone + both heads.

            Args:
                x: (B, C, 13, 13) float32 tensor.

            Returns:
                logits: (B, 6) action logits.
                value: (B, 1) state-value estimate.
            """
            features = self.backbone(x)
            logits = self.policy_head(features)
            value = self.value_head(features)
            return logits, value

        def act(self, x, mask=None):
            """Sample action from policy (for training rollouts).

            Args:
                x: (1, C, 13, 13) or (C, 13, 13) tensor.
                mask: (6,) bool tensor, True = safe action.

            Returns:
                action: int in [0, 5].
                log_prob: float (log probability of sampled action).
                value: float (state-value estimate).
                entropy: float (policy entropy in nats).
            """
            if x.dim() == 3:
                x = x.unsqueeze(0)
            logits, value = self.forward(x)
            if mask is not None:
                mask = mask.to(logits.device)
                logits = logits.clone()
                logits[:, ~mask] = -1e9
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return (
                int(action.item()),
                float(dist.log_prob(action).item()),
                float(value.squeeze(-1).item()),
                float(dist.entropy().item()),
            )

        def get_action_logits(self, x, mask=None):
            """Deterministic argmax for production inference.

            Args:
                x: (1, C, 13, 13) or (C, 13, 13) tensor.
                mask: (6,) bool tensor.

            Returns:
                action: int (argmax over safe actions).
                logits: (6,) float32 numpy array of action scores.
            """
            if x.dim() == 3:
                x = x.unsqueeze(0)
            logits, _ = self.forward(x)
            logits = logits.squeeze(0)
            if mask is not None:
                mask = mask.to(logits.device)
                logits = logits.clone()
                logits[~mask] = -1e9
            action = int(torch.argmax(logits).item())
            return action, logits.detach().cpu().numpy()

        def evaluate(self, x, action, mask=None):
            """Evaluate log-prob + entropy of given actions (for PPO update).

            Args:
                x: (B, C, 13, 13) tensor.
                action: (B,) long tensor of actions taken.
                mask: (B, 6) bool tensor of safe actions per sample.

            Returns:
                log_prob: (B,) tensor.
                entropy: (B,) tensor.
                value: (B, 1) tensor.
            """
            logits, value = self.forward(x)
            if mask is not None:
                mask = mask.to(logits.device)
                logits = logits.clone()
                logits[~mask] = -1e9
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return log_prob, entropy, value


def build_policy(input_channels=NUM_CHANNELS, num_actions=NUM_ACTIONS,
                 checkpoint_path=None):
    """Build PPOPolicy, optionally loading pretrained weights.

    Args:
        input_channels: Number of input feature channels.
        num_actions: Number of action outputs (always 6).
        checkpoint_path: Optional path to a .pth checkpoint.

    Returns:
        PPOPolicy model in eval mode.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed.")
    model = PPOPolicy(input_channels=input_channels, num_actions=num_actions)
    if checkpoint_path is not None:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def count_parameters(model):
    """Return total trainable parameters."""
    return sum(p.numel() for p in model.parameters())
