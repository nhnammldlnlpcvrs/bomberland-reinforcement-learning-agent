from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .constants import BOARD_SIZE, N_CHANNELS
except ImportError:
    from constants import BOARD_SIZE, N_CHANNELS


class BomberAuxModel(nn.Module):
    """Standalone auxiliary model for curriculum understanding.

    This model is intentionally separate from the PPO actor. It can be trained
    on curriculum rollouts without changing production or research policies.
    """

    def __init__(self, in_channels: int = N_CHANNELS, features_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, features_dim),
            nn.ReLU(),
        )
        self.death_head = nn.Linear(features_dim, 1)
        self.escape_head = nn.Linear(features_dim, 1)
        self.escape_available_head = nn.Linear(features_dim, 1)
        self.bomb_escape_available_head = nn.Linear(features_dim, 1)
        self.trapped_if_bomb_head = nn.Linear(features_dim, 1)
        self.future_blast_head = nn.Linear(features_dim, 1)
        self.box_value_head = nn.Linear(features_dim, 1)
        self.reachable_delta_head = nn.Linear(features_dim, 1)
        self.safe_tiles_after_bomb_head = nn.Linear(features_dim, 1)
        self.blast_corridor_distance_head = nn.Linear(features_dim, 1)
        self.return_head = nn.Linear(features_dim, 1)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(obs.float())
        return {
            "death_logit": self.death_head(features).squeeze(-1),
            "escape_logit": self.escape_head(features).squeeze(-1),
            "escape_available_logit": self.escape_available_head(features).squeeze(-1),
            "bomb_escape_available_logit": self.bomb_escape_available_head(features).squeeze(-1),
            "trapped_if_bomb_logit": self.trapped_if_bomb_head(features).squeeze(-1),
            "future_blast_logit": self.future_blast_head(features).squeeze(-1),
            "box_value": self.box_value_head(features).squeeze(-1),
            "reachable_delta": self.reachable_delta_head(features).squeeze(-1),
            "safe_tiles_after_bomb": self.safe_tiles_after_bomb_head(features).squeeze(-1),
            "blast_corridor_distance": self.blast_corridor_distance_head(features).squeeze(-1),
            "return": self.return_head(features).squeeze(-1),
        }
