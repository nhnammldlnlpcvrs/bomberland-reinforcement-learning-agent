from __future__ import annotations

import torch
import torch.nn as nn


class BombOutcomeCnnLstm(nn.Module):
    """CNN-LSTM outcome predictor for offline bomb decision analysis.

    This intentionally mirrors the modular recurrent BC backbone so existing
    movement/bomb/escape checkpoints can initialize the representation, while
    the outcome heads stay training-only.
    """

    def __init__(
        self,
        in_channels: int = 19,
        board_size: int = 13,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        dropout: float = 0.0,
        layer_norm: bool = False,
        include_action_head: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.board_size = int(board_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_size = int(hidden_size)
        self.num_lstm_layers = int(num_lstm_layers)
        self.dropout = float(dropout)
        self.layer_norm_enabled = bool(layer_norm)
        self.include_action_head = bool(include_action_head)

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
        self.layer_norm = nn.LayerNorm(self.embedding_dim) if self.layer_norm_enabled else nn.Identity()
        lstm_dropout = self.dropout if self.num_lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_size,
            num_layers=self.num_lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

        # Existing modular heads are kept only for checkpoint compatibility.
        self.movement_head = nn.Linear(self.hidden_size, 5)
        self.bomb_head = nn.Linear(self.hidden_size, 1)
        self.bomb_value_head = nn.Linear(self.hidden_size, 1)
        self.escape_head = nn.Linear(self.hidden_size, 5)
        self.action_head = nn.Linear(self.hidden_size, 6) if self.include_action_head else None

        self.box_value_head = nn.Linear(self.hidden_size, 1)
        self.death_risk_head = nn.Linear(self.hidden_size, 1)
        self.zero_value_head = nn.Linear(self.hidden_size, 1)
        self.escape_success_head = nn.Linear(self.hidden_size, 1)
        self.reachable_delta_head = nn.Linear(self.hidden_size, 1)

    def encode(self, obs: torch.Tensor, hidden=None):
        if obs.ndim != 5:
            raise ValueError(f"Expected obs [B,T,C,H,W], got {tuple(obs.shape)}")
        batch, seq_len, channels, height, width = obs.shape
        x = obs.reshape(batch * seq_len, channels, height, width).float()
        embeddings = self.cnn(x).reshape(batch, seq_len, self.embedding_dim)
        embeddings = self.layer_norm(embeddings)
        features, hidden = self.lstm(embeddings, hidden)
        return features, hidden

    def forward(self, obs: torch.Tensor, hidden=None):
        features, hidden = self.encode(obs, hidden)
        out = {
            "movement_logits": self.movement_head(features),
            "bomb_logit": self.bomb_head(features).squeeze(-1),
            "bomb_value_logit": self.bomb_value_head(features).squeeze(-1),
            "escape_logits": self.escape_head(features),
            "box_value": self.box_value_head(features).squeeze(-1),
            "death_risk_logit": self.death_risk_head(features).squeeze(-1),
            "zero_value_logit": self.zero_value_head(features).squeeze(-1),
            "escape_success_logit": self.escape_success_head(features).squeeze(-1),
            "reachable_delta": self.reachable_delta_head(features).squeeze(-1),
        }
        if self.action_head is not None:
            out["action_logits"] = self.action_head(features)
        return out, hidden
