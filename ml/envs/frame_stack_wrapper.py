from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FrameStackObservationWrapper(gym.Wrapper):
    """Stack the last K encoded Bomberland observations along channel axis."""

    def __init__(self, env: gym.Env, frame_stack: int = 1):
        super().__init__(env)
        self.frame_stack = max(1, int(frame_stack))
        self.frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        low = np.repeat(env.observation_space.low, self.frame_stack, axis=0)
        high = np.repeat(env.observation_space.high, self.frame_stack, axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = env.action_space

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.asarray(obs, dtype=np.float32)
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(obs)
        info = dict(info)
        info["frame_stack"] = self.frame_stack
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(np.asarray(obs, dtype=np.float32))
        info = dict(info)
        info["frame_stack"] = self.frame_stack
        return self._stacked(), reward, terminated, truncated, info

    def action_masks(self):
        if hasattr(self.env, "action_masks"):
            return self.env.action_masks()
        return None

    @property
    def last_obs(self):
        return getattr(self.env, "last_obs", None)

    def _stacked(self) -> np.ndarray:
        if not self.frames:
            raise RuntimeError("Frame stack is empty; reset must be called before reading observations.")
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)
