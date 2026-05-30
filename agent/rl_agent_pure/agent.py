from __future__ import annotations

import random
from pathlib import Path

import numpy as np

try:
    from .action_mask import legal_action_mask
    from .policy import predict_with_mask
except ImportError:  # Loaded by competition loader as top-level agent.py.
    from action_mask import legal_action_mask
    from policy import predict_with_mask


class Agent:
    team_id = "rl_agent_pure"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.model = None
        self.load_error = None
        self.rng = random.Random(2026 + self.agent_id)
        self._load_model()

    def _load_model(self):
        root = Path(__file__).resolve().parent
        candidates = [root / "policy.zip", root / "model.zip"]
        model_path = next((p for p in candidates if p.exists()), None)
        if model_path is None:
            self.load_error = "missing_policy_zip"
            return
        try:
            from stable_baselines3 import PPO

            try:
                from .model import BomberFeaturesExtractor
            except ImportError:
                from model import BomberFeaturesExtractor

            self.model = PPO.load(
                str(model_path),
                device="cpu",
                custom_objects={
                    "policy_kwargs": {
                        "features_extractor_class": BomberFeaturesExtractor,
                        "features_extractor_kwargs": {"features_dim": 256},
                        "normalize_images": False,
                    },
                },
            )
        except Exception as exc:
            self.model = None
            self.load_error = str(exc)

    def _random_valid_action(self, obs):
        mask = legal_action_mask(obs, self.agent_id)
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return 0
        return int(self.rng.choice(valid.tolist()))

    def act(self, obs: dict) -> int:
        if self.model is None:
            return self._random_valid_action(obs)
        try:
            return int(predict_with_mask(self.model, obs, self.agent_id, deterministic=True))
        except Exception:
            return self._random_valid_action(obs)
