from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

try:
    from .action_mask import highest_prob_valid, legal_action_mask, sanitize_action
    from .constants import PLACE_BOMB, STOP
    from .encoder import encode_observation
    from .modular_model import ModularBomberCnnLstm
except ImportError:  # Loaded by competition loader as top-level agent.py.
    from action_mask import highest_prob_valid, legal_action_mask, sanitize_action
    from constants import PLACE_BOMB, STOP
    from encoder import encode_observation
    from modular_model import ModularBomberCnnLstm


class Agent:
    team_id = "rl_agent_recurrent_modular"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.model = None
        self.hidden = None
        self._last_step = None
        self._last_bomb_step = None
        self.load_error = None
        self.rng = random.Random(30_000 + self.agent_id)
        self.debug_counters = {
            "rl_loaded": 0,
            "load_errors": 0,
            "inference_errors": 0,
            "total_states": 0,
            "movement_head_used": 0,
            "bomb_head_used": 0,
            "escape_head_used": 0,
            "bomb_head_activated": 0,
            "bomb_action_accepted": 0,
            "bomb_action_rejected_illegal": 0,
            "fallback_random": 0,
            "invalid_after_mask": 0,
        }
        self._load_model()

    def _metadata(self) -> dict:
        root = Path(__file__).resolve().parent
        path = root / "metadata.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _load_model(self) -> None:
        root = Path(__file__).resolve().parent
        meta = self._metadata()
        self.bomb_threshold = float(meta.get("bomb_threshold", 0.5))
        self.bomb_value_threshold = float(meta.get("bomb_value_threshold", 0.6))
        self.escape_context_steps = int(meta.get("escape_context_steps", 7))
        checkpoint_path = root / str(meta.get("checkpoint", "modular_policy.pt"))
        if not checkpoint_path.exists():
            # Also support direct copying of the research checkpoint under its
            # original filename for quick smoke tests.
            candidates = sorted(root.glob("*.pt"))
            checkpoint_path = candidates[0] if candidates else checkpoint_path
        if not checkpoint_path.exists():
            self.load_error = f"missing_checkpoint:{checkpoint_path.name}"
            self.debug_counters["load_errors"] += 1
            return
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            config = checkpoint.get("config", {})
            self.model = ModularBomberCnnLstm(
                in_channels=int(config.get("in_channels", 19)),
                embedding_dim=int(config.get("embedding_dim", 128)),
                hidden_size=int(config.get("hidden_size", 128)),
                num_lstm_layers=int(config.get("num_lstm_layers", 1)),
                dropout=float(config.get("dropout", 0.0)),
                layer_norm=bool(config.get("layer_norm", False)),
            )
            self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
            self.model.eval()
            self.debug_counters["rl_loaded"] = 1
        except Exception as exc:
            self.model = None
            self.load_error = str(exc)
            self.debug_counters["load_errors"] += 1

    def _reset_recurrent_state(self) -> None:
        self.hidden = None
        self._last_bomb_step = None

    def _random_valid_action(self, obs) -> int:
        self.debug_counters["fallback_random"] += 1
        valid = np.flatnonzero(legal_action_mask(obs, self.agent_id))
        return int(self.rng.choice(valid.tolist())) if valid.size else STOP

    def _prepare_obs(self, obs: dict) -> tuple[np.ndarray, int]:
        step = int(obs.get("step", obs.get("_step", 0)) or 0)
        if self._last_step is None or step <= self._last_step:
            self._reset_recurrent_state()
        self._last_step = step
        return encode_observation(obs, self.agent_id), step

    def _in_post_bomb_escape_context(self, step: int) -> bool:
        if self._last_bomb_step is None:
            return False
        return 0 < step - self._last_bomb_step <= self.escape_context_steps

    def _predict(self, obs: dict) -> int:
        encoded, step = self._prepare_obs(obs)
        mask = legal_action_mask(obs, self.agent_id)
        tensor = torch.as_tensor(encoded[None, None], dtype=torch.float32)
        with torch.no_grad():
            out, self.hidden = self.model(tensor, self.hidden)
            movement_probs = torch.softmax(out["movement_logits"][0, -1], dim=-1).cpu().numpy()
            escape_probs = torch.softmax(out["escape_logits"][0, -1], dim=-1).cpu().numpy()
            bomb_prob = float(torch.sigmoid(out["bomb_logit"][0, -1]).item())
            value_prob = float(torch.sigmoid(out.get("bomb_value_logit", out["bomb_logit"])[0, -1]).item())

        self.debug_counters["total_states"] += 1
        if bomb_prob >= self.bomb_threshold:
            self.debug_counters["bomb_head_activated"] += 1
        if bomb_prob >= self.bomb_threshold and value_prob >= self.bomb_value_threshold:
            if mask[PLACE_BOMB]:
                self.debug_counters["bomb_head_used"] += 1
                self.debug_counters["bomb_action_accepted"] += 1
                self._last_bomb_step = step
                return PLACE_BOMB
            self.debug_counters["bomb_action_rejected_illegal"] += 1

        if self._in_post_bomb_escape_context(step):
            self.debug_counters["escape_head_used"] += 1
            action = highest_prob_valid(np.r_[escape_probs, -np.inf], obs, self.agent_id)
        else:
            self.debug_counters["movement_head_used"] += 1
            action = highest_prob_valid(np.r_[movement_probs, -np.inf], obs, self.agent_id)

        action, invalid = sanitize_action(action, obs, self.agent_id)
        if invalid:
            self.debug_counters["invalid_after_mask"] += 1
        return int(action)

    def act(self, obs: dict) -> int:
        if self.model is None:
            return self._random_valid_action(obs)
        try:
            return self._predict(obs)
        except Exception:
            self.debug_counters["inference_errors"] += 1
            self._reset_recurrent_state()
            return self._random_valid_action(obs)
