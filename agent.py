"""
Bomberland Master Agent — PPO Inference Bridge.

Loads the trained PPO model from train_models/checkpoints/best_model.pth
and performs lightweight forward-pass inference under the 100ms/step budget.

Architecture:
  - __init__: load model weights once (counts toward 20s startup budget)
  - act():    single forward pass, no I/O, no disk writes, fully deterministic
"""

from pathlib import Path

import numpy as np
import torch

# ── Import from train_models package ─────────────────────────────────────────
# These modules are pure Python + numpy + torch; no banned imports.
from train_models.model import ActorCritic
from train_models.state_processor import encode_observation_v2, get_action_mask
from train_models.config import DEVICE, BEST_MODEL_PATH


class Agent:
    """
    PPO-trained Bomberland agent.

    Competition-compliant:
      - Imports: numpy, torch, standard library only
      - act() returns in < 100ms (single CPU forward pass, 16-channel v2 encoder)
      - No I/O, network, subprocess, or disk writes inside act()
    """
    team_id = "MasterAgentPPO"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.device = DEVICE

        # ── Build model architecture ──────────────────────────────────────────
        self.model = ActorCritic().to(self.device)
        self.model.eval()

        # ── Load trained weights ──────────────────────────────────────────────
        self._load_weights()

        # ── Warm-up forward pass (JIT trace / CUDA warmup) ────────────────────
        self._warmup()

    def _load_weights(self):
        """Load the best checkpoint. Supports both full checkpoints and weights-only."""
        # Search order: best_model.pth, latest.pth, model_final_*.pth
        candidates = [
            BEST_MODEL_PATH,
            BEST_MODEL_PATH.parent / "latest.pth",
        ]
        # Also search for any model_final_* file
        final_models = sorted(
            BEST_MODEL_PATH.parent.glob("model_final_*.pth"),
            reverse=True,
        )
        candidates.extend(final_models)
        # And any model_step_* file
        step_models = sorted(
            BEST_MODEL_PATH.parent.glob("model_step_*.pth"),
            reverse=True,
        )
        candidates.extend(step_models)

        loaded = False
        for ckpt_path in candidates:
            if not ckpt_path.exists():
                continue
            try:
                checkpoint = torch.load(
                    str(ckpt_path),
                    map_location=self.device,
                    weights_only=False,
                )
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                loaded = True
                break
            except Exception:
                continue

        if not loaded:
            # No checkpoint found — agent will use random policy with action masking
            pass

    def _warmup(self):
        """Run a single dummy forward pass to warm up the model."""
        dummy_obs = torch.zeros(1, 16, 13, 13, device=self.device)
        dummy_scalars = torch.zeros(1, 4, device=self.device)
        dummy_mask = torch.ones(1, 6, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            self.model.get_action(dummy_obs, dummy_scalars, action_mask=dummy_mask)

    @torch.no_grad()
    def act(self, obs: dict) -> int:
        """
        Select action given the current observation.

        Uses deterministic (argmax) action selection with full safety masking
        to prevent illegal moves. Falls back to random valid action on error.

        Returns:
            int: action 0–5 (STOP, LEFT, RIGHT, UP, DOWN, PLACE_BOMB)
        """
        try:
            # ── Encode state ──────────────────────────────────────────────────
            state_tensor, scalar_tensor = encode_observation_v2(obs, self.agent_id)
            action_mask = get_action_mask(obs, self.agent_id)

            # ── Prepare tensors ───────────────────────────────────────────────
            obs_batch = state_tensor.unsqueeze(0).to(self.device)   # (1, 16, 13, 13)
            scal_batch = scalar_tensor.unsqueeze(0).to(self.device)  # (1, 4)
            mask_batch = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)  # (1, 6)

            # ── Forward pass ──────────────────────────────────────────────────
            action, _, _ = self.model.get_action(
                obs_batch,
                scal_batch,
                action_mask=mask_batch,
                deterministic=True,
            )

            return int(action)

        except Exception:
            # Safety fallback: pick any legal action
            mask = get_action_mask(obs, self.agent_id)
            valid = np.flatnonzero(mask)
            if len(valid) > 0:
                return int(np.random.choice(valid))
            return 0  # STOP
