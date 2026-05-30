from __future__ import annotations

import torch
import torch.nn as nn


class StandaloneBomberCnnLstm(nn.Module):
    """Small PyTorch-only CNN-LSTM action classifier for recurrent BC diagnostics."""

    def __init__(
        self,
        in_channels: int = 19,
        board_size: int = 13,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        num_actions: int = 6,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.board_size = int(board_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_size = int(hidden_size)
        self.num_actions = int(num_actions)
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * self.board_size * self.board_size, self.embedding_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, batch_first=True)
        self.action_head = nn.Linear(self.hidden_size, self.num_actions)

    def forward(self, obs: torch.Tensor, hidden=None):
        """Return per-timestep logits for obs shaped [B, T, C, 13, 13]."""
        if obs.ndim != 5:
            raise ValueError(f"Expected obs [B,T,C,H,W], got {tuple(obs.shape)}")
        batch, seq_len, channels, height, width = obs.shape
        x = obs.reshape(batch * seq_len, channels, height, width).float()
        embeddings = self.cnn(x).reshape(batch, seq_len, self.embedding_dim)
        output, hidden = self.lstm(embeddings, hidden)
        logits = self.action_head(output)
        return logits, hidden
