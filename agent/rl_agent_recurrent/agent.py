from __future__ import annotations

import random
from pathlib import Path

import numpy as np

try:
    from .action_mask import highest_prob_valid, legal_action_mask, sanitize_action
    from .encoder import encode_observation
except ImportError:  # Loaded by competition loader as top-level agent.py.
    from action_mask import highest_prob_valid, legal_action_mask, sanitize_action
    from encoder import encode_observation


class Agent:
    team_id = "rl_agent_recurrent"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.model = None
        self.lstm_state = None
        self.episode_start = True
        self._last_step = None
        self.load_error = None
        self.rng = random.Random(10_000 + self.agent_id)
        self._load_model()

    def _load_model(self):
        root = Path(__file__).resolve().parent
        model_path = next((path for path in (root / "policy.zip", root / "model.zip") if path.exists()), None)
        if model_path is None:
            self.load_error = "missing_policy_zip"
            return
        try:
            from sb3_contrib import RecurrentPPO

            try:
                from .model import BomberFeaturesExtractor
            except ImportError:
                from model import BomberFeaturesExtractor

            self.model = RecurrentPPO.load(
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

    def _reset_recurrent_state(self):
        self.lstm_state = None
        self.episode_start = True

    def _random_valid_action(self, obs):
        valid = np.flatnonzero(legal_action_mask(obs, self.agent_id))
        return int(self.rng.choice(valid.tolist())) if valid.size else 0

    def _encoded_obs(self, obs: dict) -> np.ndarray:
        step = int(obs.get("step", obs.get("_step", 0)) or 0)
        if self._last_step is None or step <= self._last_step:
            self._reset_recurrent_state()
        self._last_step = step
        return encode_observation(obs, self.agent_id)

    def _predict_with_mask(self, obs: dict) -> int:
        tensor_obs = self._encoded_obs(obs)
        action, self.lstm_state = self.model.predict(
            tensor_obs,
            state=self.lstm_state,
            episode_start=np.array([self.episode_start], dtype=bool),
            deterministic=True,
        )
        self.episode_start = False
        action, invalid = sanitize_action(action, obs, self.agent_id)
        if not invalid:
            return int(action)

        # Safety remains a minimal legality gate. If the recurrent policy proposes
        # an illegal action, prefer its next valid probability when available.
        try:
            import torch

            with torch.no_grad():
                obs_tensor = torch.as_tensor(tensor_obs[None], dtype=torch.float32, device=self.model.policy.device)
                episode_starts = torch.as_tensor([False], dtype=torch.float32, device=self.model.policy.device)
                dist = self.model.policy.get_distribution(obs_tensor, self.lstm_state, episode_starts)
                probs = dist.distribution.probs.detach().cpu().numpy()[0]
            return highest_prob_valid(probs, obs, self.agent_id)
        except Exception:
            return self._random_valid_action(obs)

    def act(self, obs: dict) -> int:
        if self.model is None:
            return self._random_valid_action(obs)
        try:
            return self._predict_with_mask(obs)
        except Exception:
            self._reset_recurrent_state()
            return self._random_valid_action(obs)
