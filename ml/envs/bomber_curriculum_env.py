from __future__ import annotations

import random
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.action_mask import legal_action_mask
from agent.rl_agent_pure.constants import (
    BOMB_TIMER,
    BOARD_SIZE,
    DOWN,
    LEFT,
    MAX_STEPS,
    MOVE_ACTIONS,
    MOVE_DELTAS,
    PLACE_BOMB,
    RIGHT,
    STOP,
    UP,
)
from agent.rl_agent_pure.encoder import encode_observation
from agent.rl_agent_pure.utils import (
    bomb_positions,
    boxes_in_blast,
    compute_danger_map,
    normalize_obs,
    reachable_area,
)
from engine.bomb import Bomb
from ml.envs.bomber_gym_env import BomberGymEnv
from ml.envs.curriculum_scenarios import (
    find_bomb_box_value_state,
    find_bomb_then_escape_state,
    find_escape_only_state,
    has_escape_after_bomb,
    is_in_blast_corridor,
    scenario_summary,
)


CURRICULUM_MODES = ("escape_only", "bomb_then_escape", "bomb_box_value", "full_game_mix", "mixed")


class BomberCurriculumEnv(gym.Env):
    """Training-only curriculum wrapper for bomb/escape micro-skills.

    It uses the normal BomberGymEnv/engine step path, but resets into small
    controlled scenarios by editing the training env state after reset. This
    is deliberately isolated from submission/inference code.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        mode: str = "mixed",
        agent_id: int = 0,
        opponent_pool=None,
        max_steps: int = 80,
        seed: int | None = None,
        max_reset_attempts: int = 16,
        mix_schedule: str | None = None,
        retain_full_game_ratio: float = 0.0,
        training_bomb_gate: bool = True,
    ):
        super().__init__()
        if mode not in CURRICULUM_MODES:
            raise ValueError(f"Unknown curriculum mode {mode!r}; expected one of {CURRICULUM_MODES}")
        self.mode = mode
        self.agent_id = int(agent_id)
        self.max_steps = int(max_steps)
        self.max_reset_attempts = int(max_reset_attempts)
        self.rng = random.Random(seed)
        self.retain_full_game_ratio = max(0.0, min(1.0, float(retain_full_game_ratio)))
        self.mix_schedule = self._parse_mix_schedule(mix_schedule)
        self.base = BomberGymEnv(
            agent_id=agent_id,
            opponent_pool=opponent_pool or ["random", "simple"],
            max_steps=max(MAX_STEPS, max_steps),
            seed=seed,
            training_bomb_gate=training_bomb_gate,
        )
        self.observation_space = self.base.observation_space
        self.action_space = self.base.action_space
        self.current_mode = mode
        self.episode_step = 0
        self.fallback_count = 0
        self.scenario_info: dict = {}
        self.initial_boxes = 0
        self.initial_area = 0
        self.bomb_placed = False
        self.bomb_step: int | None = None
        self.bomb_pos: tuple[int, int] | None = None
        self.bomb_expected_boxes = 0
        self.escape_rewarded = False
        self.box_rewarded = False
        self.completed = False
        self.mode_counts = {name: 0 for name in CURRICULUM_MODES if name != "mixed"}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        requested_mode = (options or {}).get("mode") if options else None
        self.current_mode = requested_mode or self._sample_mode()
        if self.current_mode in self.mode_counts:
            self.mode_counts[self.current_mode] += 1
        self.episode_step = 0
        self.bomb_placed = False
        self.bomb_step = None
        self.bomb_pos = None
        self.bomb_expected_boxes = 0
        self.escape_rewarded = False
        self.box_rewarded = False
        self.completed = False

        obs, _info = self.base.reset(seed=seed)
        scenario_ok = False
        fallback_reason = ""
        for attempt in range(max(1, self.max_reset_attempts)):
            if attempt > 0:
                obs, _info = self.base.reset(seed=None if seed is None else seed + attempt)
            if self.current_mode == "full_game_mix":
                scenario_ok = True
                break
            obs_dict = self._inject_scenario(self.current_mode)
            self.base.last_obs = obs_dict
            scenario_ok, fallback_reason = self._scenario_matches(self.current_mode, obs_dict)
            if scenario_ok:
                obs = encode_observation(obs_dict, self.agent_id)
                break

        if not scenario_ok:
            self.fallback_count += 1
            self.current_mode = "full_game_mix"
            obs, _info = self.base.reset(seed=seed)
            fallback_reason = fallback_reason or "max_attempts_exhausted"

        obs_dict = self.base.last_obs
        board, players, bombs, _ = normalize_obs(obs_dict)
        row, col = int(players[self.agent_id, 0]), int(players[self.agent_id, 1])
        danger = compute_danger_map(board, players, bombs)
        self.initial_boxes = int((board == 2).sum())
        self.initial_area = int(reachable_area(board, bomb_positions(bombs), danger, (row, col), max_depth=10).sum())
        self.scenario_info = scenario_summary(obs_dict, self.agent_id)
        self.scenario_info.update({
            "mode": self.current_mode,
            "fallback": not scenario_ok,
            "fallback_reason": fallback_reason,
            "fallback_count": self.fallback_count,
            "mode_counts": dict(self.mode_counts),
        })
        return encode_observation(obs_dict, self.agent_id), self._info({}, invalid_action=False)

    def step(self, action):
        prev_obs = self.base.last_obs
        action = int(action)
        mask = legal_action_mask(prev_obs, self.agent_id)
        invalid_action = not (0 <= action < len(mask) and bool(mask[action]))
        obs, _base_reward, terminated, truncated, base_info = self.base.step(action)
        next_obs = self.base.last_obs
        self.episode_step += 1
        reward, components = self._curriculum_reward(prev_obs, next_obs, action, invalid_action, terminated or truncated)
        mode_truncated = self.episode_step >= self.max_steps
        done = bool(terminated or self.completed)
        info = self._info(components, invalid_action=invalid_action)
        info["base_info"] = base_info
        return obs, reward, done, bool(truncated or mode_truncated), info

    def action_masks(self):
        return self.base.action_masks()

    def _parse_mix_schedule(self, value: str | None) -> list[tuple[str, float]]:
        if not value:
            return [("escape_only", 0.25), ("bomb_then_escape", 0.30), ("bomb_box_value", 0.35), ("full_game_mix", 0.10)]
        out = []
        for item in value.split(","):
            if not item.strip():
                continue
            name, weight = item.split(":")
            name = name.strip()
            if name not in CURRICULUM_MODES or name == "mixed":
                raise ValueError(f"Invalid mix_schedule mode {name!r}")
            out.append((name, float(weight)))
        total = sum(weight for _name, weight in out)
        return [(name, weight / total) for name, weight in out] if total > 0 else [("full_game_mix", 1.0)]

    def _sample_mode(self) -> str:
        if self.mode != "mixed":
            if self.retain_full_game_ratio > 0 and self.mode != "full_game_mix":
                if self.rng.random() < self.retain_full_game_ratio:
                    return "full_game_mix"
            return self.mode
        value = self.rng.random()
        cumulative = 0.0
        for name, weight in self.mix_schedule:
            cumulative += weight
            if value <= cumulative:
                return name
        return self.mix_schedule[-1][0]

    def _scenario_matches(self, mode: str, obs: dict) -> tuple[bool, str]:
        if mode == "escape_only":
            result = find_escape_only_state(obs, self.agent_id)
        elif mode == "bomb_then_escape":
            result = find_bomb_then_escape_state(obs, self.agent_id)
        elif mode == "bomb_box_value":
            result = find_bomb_box_value_state(obs, self.agent_id)
        else:
            return True, "full_game"
        return bool(result.get("ok")), str(result.get("reason", "unknown"))

    def _inject_scenario(self, mode: str) -> dict:
        env = self.base.env
        grid = env.map.grid
        pos = self._choose_training_pos()
        self._clear_patch(grid, pos)
        self._place_players(pos)
        env.bombs = []
        if mode == "escape_only":
            self._build_escape_only(pos)
        elif mode == "bomb_then_escape":
            self._build_bomb_then_escape(pos)
        elif mode == "bomb_box_value":
            self._build_bomb_box_value(pos)
        env.current_step = 0
        obs = {**env._get_obs(), "step": 0}
        return obs

    def _choose_training_pos(self) -> tuple[int, int]:
        candidates = [(5, 5), (5, 7), (7, 5), (7, 7), (3, 5), (5, 3), (7, 9), (9, 7)]
        return self.rng.choice(candidates)

    def _clear_patch(self, grid: np.ndarray, pos: tuple[int, int], radius: int = 3) -> None:
        row, col = pos
        for r in range(max(1, row - radius), min(BOARD_SIZE - 1, row + radius + 1)):
            for c in range(max(1, col - radius), min(BOARD_SIZE - 1, col + radius + 1)):
                grid[r, c] = 0
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1

    def _place_players(self, pos: tuple[int, int]) -> None:
        env = self.base.env
        safe_spots = [(11, 11), (1, 11), (11, 1)]
        for idx, player in enumerate(env.players):
            if idx == self.agent_id:
                player.x, player.y = pos
                player.alive = True
                player.bombs_left = 1
                player.bomb_radius_bonus = 0
            else:
                player.x, player.y = safe_spots.pop(0)
                player.alive = True
                player.bombs_left = 0
                player.bomb_radius_bonus = 0

    def _build_escape_only(self, pos: tuple[int, int]) -> None:
        env = self.base.env
        env.bombs = [Bomb(pos[0], pos[1], self.agent_id, radius=2, timer=4)]
        env.players[self.agent_id].bombs_left = 0
        # Keep one side open and add a tempting corridor so the skill is moving out, not waiting.
        for action in (LEFT, RIGHT, UP, DOWN):
            dr, dc = MOVE_DELTAS[action]
            env.map.grid[pos[0] + dr, pos[1] + dc] = 0

    def _build_bomb_then_escape(self, pos: tuple[int, int]) -> None:
        env = self.base.env
        row, col = pos
        env.map.grid[row, col + 1] = 2
        env.map.grid[row - 1, col] = 0
        env.map.grid[row + 1, col] = 0
        env.map.grid[row, col - 1] = 0

    def _build_bomb_box_value(self, pos: tuple[int, int]) -> None:
        env = self.base.env
        row, col = pos
        for br, bc in ((row, col + 1), (row + 1, col), (row, col - 2)):
            if 0 < br < BOARD_SIZE - 1 and 0 < bc < BOARD_SIZE - 1:
                env.map.grid[br, bc] = 2
        env.map.grid[row - 1, col] = 0
        env.map.grid[row - 2, col] = 0

    def _curriculum_reward(self, prev_obs: dict, obs: dict, action: int, invalid_action: bool, done: bool) -> tuple[float, dict]:
        if self.current_mode == "full_game_mix":
            # The normal env reward is intentionally not reused here because
            # this wrapper focuses on short tactical micro-skills. Full-game
            # performance is measured with ml.evaluate_rl_pure.
            return self._full_game_mix_reward(prev_obs, obs, action, invalid_action, done)
        if self.current_mode == "escape_only":
            return self._escape_only_reward(prev_obs, obs, action, invalid_action, done)
        if self.current_mode == "bomb_then_escape":
            return self._bomb_escape_reward(prev_obs, obs, action, invalid_action, done, box_value=False)
        return self._bomb_escape_reward(prev_obs, obs, action, invalid_action, done, box_value=True)

    def _base_components(self) -> dict[str, float]:
        return {
            "time_penalty": -0.2,
            "invalid_action": 0.0,
            "escape_blast_zone": 0.0,
            "increase_bomb_distance": 0.0,
            "stay_in_blast_corridor": 0.0,
            "place_valid_bomb": 0.0,
            "second_bomb_spam": 0.0,
            "survive_after_bomb": 0.0,
            "successful_bomb_escape": 0.0,
            "bomb_destroy_box_survived": 0.0,
            "reachable_area_increase": 0.0,
            "death": 0.0,
            "own_bomb_death": 0.0,
            "excessive_stop": 0.0,
            "missed_bomb_objective": 0.0,
            "scenario_complete": 0.0,
        }

    def _positions(self, prev_obs: dict, obs: dict) -> tuple[tuple[int, int], tuple[int, int]]:
        _pb, prev_players, _pbo, _ps = normalize_obs(prev_obs)
        _b, players, _bo, _s = normalize_obs(obs)
        return (
            (int(prev_players[self.agent_id, 0]), int(prev_players[self.agent_id, 1])),
            (int(players[self.agent_id, 0]), int(players[self.agent_id, 1])),
        )

    def _escape_only_reward(self, prev_obs: dict, obs: dict, action: int, invalid_action: bool, done: bool) -> tuple[float, dict]:
        components = self._base_components()
        prev_board, prev_players, prev_bombs, _ = normalize_obs(prev_obs)
        board, players, bombs, _ = normalize_obs(obs)
        prev_pos, pos = self._positions(prev_obs, obs)
        alive = bool(players[self.agent_id, 2])
        components["invalid_action"] = -10.0 if invalid_action else 0.0
        components["excessive_stop"] = -3.0 if action == STOP else 0.0
        if prev_bombs.size:
            bomb = prev_bombs[0]
            prev_dist = abs(prev_pos[0] - int(bomb[0])) + abs(prev_pos[1] - int(bomb[1]))
            curr_dist = abs(pos[0] - int(bomb[0])) + abs(pos[1] - int(bomb[1]))
            if curr_dist > prev_dist:
                components["increase_bomb_distance"] = 4.0
        if is_in_blast_corridor(obs, pos, self.agent_id):
            components["stay_in_blast_corridor"] = -8.0
        else:
            components["escape_blast_zone"] = 20.0
        danger = compute_danger_map(board, players, bombs)
        if alive and not is_in_blast_corridor(obs, pos, self.agent_id) and danger[pos[0], pos[1]] > 3:
            components["scenario_complete"] = 80.0
            self.completed = True
        if not alive:
            components["death"] = -300.0
        return float(sum(components.values())), components

    def _bomb_escape_reward(self, prev_obs: dict, obs: dict, action: int, invalid_action: bool, done: bool, box_value: bool) -> tuple[float, dict]:
        components = self._base_components()
        prev_board, prev_players, prev_bombs, _ = normalize_obs(prev_obs)
        board, players, bombs, _ = normalize_obs(obs)
        prev_pos, pos = self._positions(prev_obs, obs)
        alive = bool(players[self.agent_id, 2])
        components["invalid_action"] = -10.0 if invalid_action else 0.0
        components["excessive_stop"] = -4.0 if action == STOP else 0.0

        if action == PLACE_BOMB and not self.bomb_placed:
            if has_escape_after_bomb(prev_obs, self.agent_id):
                expected_boxes = boxes_in_blast(prev_board, prev_players, prev_pos[0], prev_pos[1], self.agent_id)
                if expected_boxes > 0 or not box_value:
                    components["place_valid_bomb"] = 10.0 if box_value else 15.0
                self.bomb_expected_boxes = int(expected_boxes)
            else:
                components["place_valid_bomb"] = -30.0
            self.bomb_placed = True
            self.bomb_step = self.episode_step
            self.bomb_pos = prev_pos
        elif action == PLACE_BOMB and self.bomb_placed:
            components["second_bomb_spam"] = -35.0

        if not self.bomb_placed:
            if self.episode_step >= max(8, self.max_steps // 3):
                components["missed_bomb_objective"] = -2.0
            return float(sum(components.values())), components

        age = self.episode_step - int(self.bomb_step or 0)
        if self.bomb_pos is not None:
            prev_dist = abs(prev_pos[0] - self.bomb_pos[0]) + abs(prev_pos[1] - self.bomb_pos[1])
            curr_dist = abs(pos[0] - self.bomb_pos[0]) + abs(pos[1] - self.bomb_pos[1])
            if curr_dist > prev_dist:
                components["increase_bomb_distance"] = 6.0
        if alive:
            components["survive_after_bomb"] = 2.0 if age <= BOMB_TIMER + 1 else 0.0
        if is_in_blast_corridor(obs, pos, self.agent_id) and age <= BOMB_TIMER:
            components["stay_in_blast_corridor"] = -12.0

        current_boxes = int((board == 2).sum())
        destroyed = max(0, self.initial_boxes - current_boxes)
        danger = compute_danger_map(board, players, bombs)
        escaped = alive and danger[pos[0], pos[1]] > 3 and not is_in_blast_corridor(obs, pos, self.agent_id)
        if escaped and age >= 3 and not self.escape_rewarded:
            components["successful_bomb_escape"] = 100.0
            self.escape_rewarded = True
        area = int(reachable_area(board, bomb_positions(bombs), danger, pos, max_depth=10).sum())
        if area > self.initial_area:
            components["reachable_area_increase"] = 8.0
        if box_value and destroyed > 0 and escaped and not self.box_rewarded:
            components["bomb_destroy_box_survived"] = 90.0 * destroyed
            components["scenario_complete"] = 80.0
            self.box_rewarded = True
            self.completed = True
        elif not box_value and escaped and age >= 5:
            components["scenario_complete"] = 70.0
            self.completed = True
        if not alive:
            components["death"] = -350.0
            if 0 <= age <= 8:
                components["own_bomb_death"] = -350.0
        return float(sum(components.values())), components

    def _full_game_mix_reward(self, prev_obs: dict, obs: dict, action: int, invalid_action: bool, done: bool) -> tuple[float, dict]:
        components = self._base_components()
        _prev_board, prev_players, _prev_bombs, _ = normalize_obs(prev_obs)
        board, players, bombs, _ = normalize_obs(obs)
        pos = (int(players[self.agent_id, 0]), int(players[self.agent_id, 1]))
        alive = bool(players[self.agent_id, 2])
        components["invalid_action"] = -10.0 if invalid_action else 0.0
        components["excessive_stop"] = -5.0 if action == STOP else 0.0
        if alive:
            components["survive_after_bomb"] = 0.2
        if bool(prev_players[self.agent_id, 2]) and not alive:
            components["death"] = -300.0
        if alive and bombs.size and is_in_blast_corridor(obs, pos, self.agent_id):
            components["stay_in_blast_corridor"] = -5.0
        return float(sum(components.values())), components

    def _info(self, components: dict, invalid_action: bool) -> dict:
        obs = self.base.last_obs
        board, players, bombs, _ = normalize_obs(obs)
        pos = (int(players[self.agent_id, 0]), int(players[self.agent_id, 1]))
        danger = compute_danger_map(board, players, bombs)
        return {
            "curriculum_mode": self.current_mode,
            "scenario": dict(self.scenario_info),
            "fallback_count": int(self.fallback_count),
            "sampled_mode_counts": dict(self.mode_counts),
            "invalid_action": bool(invalid_action),
            "alive": bool(players[self.agent_id, 2]),
            "survival_step": int(self.episode_step),
            "bomb_placed": bool(self.bomb_placed),
            "bomb_age": -1 if self.bomb_step is None else int(self.episode_step - self.bomb_step),
            "boxes_destroyed": int(max(0, self.initial_boxes - int((board == 2).sum()))),
            "current_danger": int(danger[pos[0], pos[1]]),
            "in_blast_corridor": bool(is_in_blast_corridor(obs, pos, self.agent_id)),
            "reward_components": dict(components),
        }
