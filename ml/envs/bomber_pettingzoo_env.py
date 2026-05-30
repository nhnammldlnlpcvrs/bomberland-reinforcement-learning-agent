from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from pettingzoo import ParallelEnv
except Exception:  # pragma: no cover
    ParallelEnv = object

from gymnasium import spaces

from agent.rl_agent_pure.action_mask import legal_action_mask
from agent.rl_agent_pure.constants import BOARD_SIZE, N_CHANNELS, NUM_ACTIONS
from agent.rl_agent_pure.encoder import encode_observation
from engine.game import BomberEnv


class BomberParallelEnv(ParallelEnv):
    """PettingZoo ParallelEnv scaffold for full self-play.

    The Gymnasium single-learner wrapper is the maintained training path. This
    class exposes simultaneous actions for future multi-policy self-play.
    """

    metadata = {"name": "bomberland_parallel_v0"}

    def __init__(self, max_steps=500, seed=None):
        self.env = BomberEnv(max_steps=max_steps, seed=seed)
        self.possible_agents = [f"player_{idx}" for idx in range(4)]
        self.agents = list(self.possible_agents)
        self._obs = None
        self.observation_spaces = {
            agent: spaces.Box(0.0, 1.0, shape=(N_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {agent: spaces.Discrete(NUM_ACTIONS) for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self._obs = {**self.env.reset(seed=seed), "step": 0}
        observations = {agent: encode_observation(self._obs, idx) for idx, agent in enumerate(self.possible_agents)}
        infos = {agent: {"action_mask": legal_action_mask(self._obs, idx)} for idx, agent in enumerate(self.possible_agents)}
        return observations, infos

    def step(self, actions):
        ordered = [int(actions.get(agent, 0)) for agent in self.possible_agents]
        self._obs, terminated, truncated = self.env.step(ordered)
        self._obs = {**self._obs, "step": int(self.env.current_step)}
        observations = {agent: encode_observation(self._obs, idx) for idx, agent in enumerate(self.possible_agents)}
        rewards = {agent: 0.0 for agent in self.possible_agents}
        terminations = {agent: bool(terminated) for agent in self.possible_agents}
        truncations = {agent: bool(truncated) for agent in self.possible_agents}
        infos = {agent: {"action_mask": legal_action_mask(self._obs, idx)} for idx, agent in enumerate(self.possible_agents)}
        if terminated or truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
