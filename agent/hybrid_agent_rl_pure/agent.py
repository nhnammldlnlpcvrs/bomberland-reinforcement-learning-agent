"""Submission-ready pure RL-style Bomberland agent.

The policy is a compact Dueling DQN when `rl_pure_model.pth` is present.
If loading fails, the agent falls back to the same BFS/safety policy used as
the final action guard, so malformed observations or missing weights do not
crash the match.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from .model import DuelingDQN, encode_observation, fallback_policy, safe_action_mask
except Exception:  # Evaluators often import agent.py as a loose file.
    from model import DuelingDQN, encode_observation, fallback_policy, safe_action_mask


class Agent:
    team_id = "hybrid_agent_rl_pure"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.base_dir = Path(__file__).resolve().parent
        self.device = "cpu"
        self.model = None
        self.max_inference_s = 0.095
        self._load_model()

    def _load_model(self):
        if torch is None or DuelingDQN is None:
            return
        try:
            config_path = self.base_dir / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
            ckpt_path = self.base_dir / "rl_pure_model.pth"
            if not ckpt_path.exists():
                return
            checkpoint = torch.load(str(ckpt_path), map_location="cpu")
            model_cfg = checkpoint.get("model_config", config)
            self.model = DuelingDQN(
                spatial_channels=int(model_cfg.get("spatial_channels", 18)),
                scalar_dim=int(model_cfg.get("scalar_dim", 8)),
                num_actions=int(model_cfg.get("num_actions", 6)),
                hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            )
            state = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state)
            self.model.eval()
        except Exception:
            self.model = None

    def act(self, obs: dict) -> int:
        start = time.perf_counter()
        try:
            q_values = None
            if self.model is not None and torch is not None:
                spatial, scalar = encode_observation(obs, self.agent_id)
                with torch.no_grad():
                    spatial_t = torch.from_numpy(spatial).unsqueeze(0)
                    scalar_t = torch.from_numpy(scalar).unsqueeze(0)
                    q_values = self.model(spatial_t, scalar_t).squeeze(0).cpu().numpy()
                if time.perf_counter() - start > self.max_inference_s:
                    return int(fallback_policy(obs, self.agent_id))
            action = fallback_policy(obs, self.agent_id, q_values)
            if not safe_action_mask(obs, self.agent_id)[action]:
                action = fallback_policy(obs, self.agent_id)
            if 0 <= int(action) <= 5:
                return int(action)
        except Exception:
            pass
        return 0
