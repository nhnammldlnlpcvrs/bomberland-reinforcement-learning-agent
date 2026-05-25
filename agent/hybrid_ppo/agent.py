"""Hybrid PPO Imitation Agent — deterministic argmax with safety mask.

Loads the imitation-pretrained CNN policy and delegates action selection
to PPO logits hard-gated by the safety filter. Falls back to online_robust
heuristic if the model fails to load or confidence is low.

All internal imports use importlib.util to avoid namespace clashes with
the competition runtime_guard which registers agent modules as "agent".
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
_HERE = Path(__file__).resolve().parent


def _load_module(name, path):
    """Load a Python module from an absolute file path.

    Pre-registers the flat "agent" module (set by the competition runtime_guard)
    as a pseudo-package so that intra-package imports like
    ``from agent.hybrid_ppo.xxx import Y`` resolve correctly.
    """
    if "agent" in sys.modules and not hasattr(sys.modules["agent"], "__path__"):
        sys.modules["agent"].__path__ = [str(_ROOT / "agent")]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- Load internal dependencies at module level ----
_rewards = _load_module("ppo_reward", _HERE / "reward.py")
_safety = _load_module("ppo_safety", _HERE / "safety_filter.py")
_encoder = _load_module("ppo_encoder", _HERE / "state_encoder.py")
_policy = _load_module("ppo_policy", _HERE / "ppo_policy.py")

encode_state = _encoder.encode_state
compute_safe_action_mask = _safety.compute_safe_action_mask
NUM_CHANNELS = _policy.NUM_CHANNELS
NUM_ACTIONS = _policy.NUM_ACTIONS

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

CONFIDENCE_THRESHOLD = 0.22
CHECKPOINT_PATH = str(_ROOT / "ml/checkpoints/hybrid_ppo/imitation_cnn.pt")

# ---- Lazy-loaded fallback Agent class ----
_fallback_cls = None


def _get_fallback_cls():
    global _fallback_cls
    if _fallback_cls is None:
        mod = _load_module(
            "online_robust_fb",
            _ROOT / "agent/hybrid_agent_online_robust/agent.py",
        )
        _fallback_cls = mod.Agent
    return _fallback_cls


class Agent:
    team_id = "HybridPPOImitation"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        FallbackAgent = _get_fallback_cls()
        self._fallback = FallbackAgent(agent_id=agent_id)

        # ---- PPO model ----
        self._model = None
        self._model_ok = False

        if not TORCH_AVAILABLE:
            return

        try:
            PPOPolicy = _policy.PPOPolicy
            self._model = PPOPolicy(
                input_channels=NUM_CHANNELS, num_actions=NUM_ACTIONS
            )
            ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu",
                              weights_only=False)
            self._model.load_state_dict(ckpt["model_state_dict"])
            self._model.eval()
            self._model_ok = True
        except Exception:
            self._model = None
            self._model_ok = False

    def act(self, obs: dict) -> int:
        p = obs["players"][self.agent_id]
        if not int(p[2]):
            return 0  # dead

        mask = compute_safe_action_mask(obs, self.agent_id)

        if not mask.any():
            return 0  # no safe action → STOP

        if not self._model_ok or self._model is None:
            return self._fallback.act(obs)

        try:
            state = encode_state(obs, self.agent_id)
            state_t = torch.from_numpy(state).float().unsqueeze(0)
            mask_t = torch.from_numpy(mask).bool()

            with torch.no_grad():
                logits, _ = self._model(state_t)
                logits = logits.squeeze(0)
                logits[~mask_t] = -1e9
                probs = torch.softmax(logits, dim=-1)
                max_prob, action = torch.max(probs, dim=-1)

        except Exception:
            return self._fallback.act(obs)

        num_safe = int(mask.sum())
        if num_safe > 1 and float(max_prob.item()) < CONFIDENCE_THRESHOLD:
            return self._fallback.act(obs)

        return int(action.item())
