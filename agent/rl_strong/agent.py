from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

try:
    from .action_mask import highest_prob_valid, legal_action_mask, sanitize_action
    from .encoder import encode_observation
    from .frame_buffer import FrameBuffer
except ImportError:  # Loaded by competition loader as top-level agent.py.
    from action_mask import highest_prob_valid, legal_action_mask, sanitize_action
    from encoder import encode_observation
    from frame_buffer import FrameBuffer


class Agent:
    team_id = "rl_strong"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.model = None
        self.load_error = None
        self.rng = random.Random(9000 + self.agent_id)
        self.frame_stack = 4
        self.buffer = FrameBuffer(self.frame_stack)
        self._last_step = None
        self._load_metadata()
        self._load_model()

    def _load_metadata(self):
        meta_path = Path(__file__).resolve().parent / "metadata.json"
        if not meta_path.exists():
            return
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            self.frame_stack = max(1, int(metadata.get("frame_stack", self.frame_stack)))
            self.buffer = FrameBuffer(self.frame_stack)
        except Exception:
            self.frame_stack = 4
            self.buffer = FrameBuffer(self.frame_stack)

    def _load_model(self):
        root = Path(__file__).resolve().parent
        candidates = [root / "policy.zip", root / "model.zip"]
        model_path = next((path for path in candidates if path.exists()), None)
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
        valid = np.flatnonzero(legal_action_mask(obs, self.agent_id))
        return int(self.rng.choice(valid.tolist())) if valid.size else 0

    def _stacked_obs(self, obs: dict) -> np.ndarray:
        frame = encode_observation(obs, self.agent_id)
        step = int(obs.get("step", obs.get("_step", 0)) or 0)
        if self._last_step is None or step <= self._last_step:
            self._last_step = step
            return self.buffer.reset(frame)
        self._last_step = step
        return self.buffer.append(frame)

    def _predict_with_mask(self, obs: dict) -> int:
        tensor_obs = self._stacked_obs(obs)
        action, _state = self.model.predict(tensor_obs, deterministic=True)
        action, invalid = sanitize_action(action, obs, self.agent_id)
        if not invalid:
            return int(action)
        try:
            import torch

            with torch.no_grad():
                obs_tensor = torch.as_tensor(tensor_obs[None], dtype=torch.float32, device=self.model.policy.device)
                dist = self.model.policy.get_distribution(obs_tensor)
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
            return self._random_valid_action(obs)
