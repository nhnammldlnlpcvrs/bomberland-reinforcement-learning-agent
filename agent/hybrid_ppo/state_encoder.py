"""State encoding for hybrid PPO agent.

Wraps ml.features.encode_observation and adds a normalized step-count
channel for late-game strategy awareness. Produces (13, 13, 13) tensors.
"""

import sys
from pathlib import Path

import numpy as np

# Ensure project root is on path so ml.features is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.features import encode_observation, BOARD_SIZE, CHANNEL_NAMES

EXTRA_CHANNELS = ["steps_remaining"]
FULL_CHANNEL_NAMES = list(CHANNEL_NAMES) + EXTRA_CHANNELS
NUM_CHANNELS = len(FULL_CHANNEL_NAMES)


def encode_state(obs: dict, agent_id: int, max_steps: int = 500) -> np.ndarray:
    """Encode observation into (NUM_CHANNELS, 13, 13) float32 tensor.

    Args:
        obs: Raw Bomberland observation dict with 'map', 'players', 'bombs'.
        agent_id: Index of this agent in the players array.
        max_steps: Maximum match steps for normalization.

    Returns:
        np.ndarray of shape (13, 13, 13), dtype float32.
    """
    frame = dict(obs)
    frame["_agent_index"] = agent_id
    encoded = encode_observation(frame)
    base_tensor = encoded["tensor"]  # (12, 13, 13)

    step = int(obs.get("step", encoded.get("step", 0)) or 0)
    steps_channel = np.full(
        (1, BOARD_SIZE, BOARD_SIZE),
        step / max(max_steps, 1),
        dtype=np.float32,
    )

    result = np.concatenate([base_tensor, steps_channel], axis=0).astype(np.float32)
    return result
