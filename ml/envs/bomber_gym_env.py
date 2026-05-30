from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agent import BoxFarmerAgent, RandomAgent, SimpleRuleAgent, SmarterRuleAgent, TacticalRuleAgent
from agent.rl_agent_pure.action_mask import legal_action_mask
from agent.rl_agent_pure.constants import BOARD_SIZE, MAX_STEPS, N_CHANNELS, NUM_ACTIONS, PLACE_BOMB, REWARD_WEIGHTS, STOP
from agent.rl_agent_pure.encoder import encode_observation
from agent.rl_agent_pure.policy import predict_with_mask
from agent.rl_agent_pure.utils import boxes_in_blast, has_escape_after_bomb, normalize_obs, reachable_area, bomb_positions, compute_danger_map
from competition.evaluation.runtime_guard import load_agent_instance
from engine.game import BomberEnv


BASELINE_AGENTS = {
    "random": RandomAgent,
    "simple": SimpleRuleAgent,
    "box_farmer": BoxFarmerAgent,
    "smarter": SmarterRuleAgent,
    "tactical": TacticalRuleAgent,
}
OPPONENT_ALIASES = {
    "online_robust": "agent/hybrid_agent_online_robust",
    "hybrid_agent_rl": "agent/hybrid_agent_rl",
    "rl_agent_pure": "agent/rl_agent_pure",
}


class PPOCheckpointAgent:
    def __init__(self, checkpoint_path, agent_id):
        self.agent_id = int(agent_id)
        self.model = None
        try:
            from stable_baselines3 import PPO
            from agent.rl_agent_pure.model import BomberFeaturesExtractor

            self.model = PPO.load(
                str(checkpoint_path),
                device="cpu",
                custom_objects={
                    "policy_kwargs": {
                        "features_extractor_class": BomberFeaturesExtractor,
                        "features_extractor_kwargs": {"features_dim": 256},
                        "normalize_images": False,
                    },
                },
            )
        except Exception:
            self.model = None

    def act(self, obs):
        if self.model is None:
            valid = np.flatnonzero(legal_action_mask(obs, self.agent_id))
            return int(valid[0]) if valid.size else STOP
        return predict_with_mask(self.model, obs, self.agent_id, deterministic=True)


class BomberGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        agent_id=0,
        opponent_pool=None,
        max_steps=MAX_STEPS,
        seed=None,
        reward_weights=None,
        training_bomb_gate=False,
    ):
        super().__init__()
        self.agent_id = int(agent_id)
        self.max_steps = int(max_steps)
        self.reward_weights = dict(REWARD_WEIGHTS)
        if reward_weights:
            self.reward_weights.update(reward_weights)
        self.training_bomb_gate = bool(training_bomb_gate)
        self.env = BomberEnv(max_steps=max_steps, seed=seed)
        self.rng = random.Random(seed)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(N_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.opponent_pool = opponent_pool or ["random", "simple", "tactical", "agent/hybrid_agent_online_robust"]
        self.opponents = {}
        self.last_obs = None
        self.visited = set()
        self.position_history = deque(maxlen=8)
        self.episode_reward = 0.0
        self.self_bomb_credit_steps = 0
        self.bomb_events = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        self.last_obs = self.env.reset(seed=seed)
        self.last_obs = {**self.last_obs, "step": 0}
        self.opponents = self._make_opponents()
        self.episode_reward = 0.0
        self.self_bomb_credit_steps = 0
        self.bomb_events = []
        row, col = self._self_pos(self.last_obs)
        self.visited = {(row, col)}
        self.position_history.clear()
        self.position_history.append((row, col))
        return encode_observation(self.last_obs, self.agent_id), self._info(self.last_obs, {})

    def step(self, action):
        prev_obs = self.last_obs
        action = int(action)
        mask = legal_action_mask(prev_obs, self.agent_id)
        invalid_action = not (0 <= action < NUM_ACTIONS and bool(mask[action]))
        if invalid_action:
            action = STOP
        gated_bomb = False
        if self.training_bomb_gate and action == PLACE_BOMB and not self._bomb_context_allowed(prev_obs):
            action = self._escape_first_action(prev_obs, mask)
            gated_bomb = True

        actions = []
        obs_for_agents = {**prev_obs, "step": int(self.env.current_step)}
        for idx in range(4):
            if idx == self.agent_id:
                actions.append(action)
            else:
                try:
                    actions.append(int(self.opponents[idx].act(obs_for_agents)))
                except Exception:
                    actions.append(STOP)

        next_obs, terminated, truncated = self.env.step(actions)
        next_obs = {**next_obs, "step": int(self.env.current_step)}
        self.last_obs = next_obs
        reward, components = self._reward(prev_obs, next_obs, action, invalid_action, terminated or truncated, gated_bomb)
        self.episode_reward += reward
        return encode_observation(next_obs, self.agent_id), reward, terminated, truncated, self._info(next_obs, components, invalid_action)

    def action_masks(self):
        return legal_action_mask(self.last_obs, self.agent_id)

    def _make_opponents(self):
        opponents = {}
        for idx in range(4):
            if idx == self.agent_id:
                continue
            name = self.rng.choice(self.opponent_pool)
            opponents[idx] = self._load_opponent(name, idx)
        return opponents

    def _load_opponent(self, name, agent_id):
        name = OPPONENT_ALIASES.get(name, name)
        if name in BASELINE_AGENTS:
            return BASELINE_AGENTS[name](agent_id)
        path = ROOT / name if not Path(name).is_absolute() else Path(name)
        if path.suffix == ".zip" and path.exists():
            return PPOCheckpointAgent(path, agent_id)
        if path.is_dir():
            path = path / "agent.py"
        if path.exists():
            return load_agent_instance(str(path), agent_id)
        return RandomAgent(agent_id)

    def _self_pos(self, obs):
        players = np.asarray(obs["players"])
        return int(players[self.agent_id, 0]), int(players[self.agent_id, 1])

    def _enemy_in_blast(self, board, players, row, col):
        radius = 1 + max(0, int(players[self.agent_id, 4]))
        for enemy_id, enemy in enumerate(players):
            if enemy_id == self.agent_id or not int(enemy[2]):
                continue
            enemy_pos = (int(enemy[0]), int(enemy[1]))
            if enemy_pos == (row, col):
                return True
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                for dist in range(1, radius + 1):
                    nr, nc = row + dr * dist, col + dc * dist
                    if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE) or int(board[nr, nc]) == 1:
                        break
                    if (nr, nc) == enemy_pos:
                        return True
                    if int(board[nr, nc]) == 2:
                        break
        return False

    def _bomb_context_allowed(self, obs):
        board, players, bombs, _ = normalize_obs(obs)
        row, col = self._self_pos(obs)
        useful = boxes_in_blast(board, players, row, col, self.agent_id) > 0 or self._enemy_in_blast(board, players, row, col)
        return useful and has_escape_after_bomb(board, players, bombs, self.agent_id)

    def _escape_first_action(self, obs, mask):
        board, players, bombs, _ = normalize_obs(obs)
        row, col = self._self_pos(obs)
        sim_bomb = np.array([[row, col, 7, self.agent_id]], dtype=np.int16)
        sim_bombs = sim_bomb if bombs.size == 0 else np.vstack([bombs, sim_bomb])
        danger = compute_danger_map(board, players, sim_bombs)
        best_action = STOP
        best_danger = danger[row, col] if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE else 0
        for action in (1, 2, 3, 4):
            if not bool(mask[action]):
                continue
            dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
            nr, nc = row + dr, col + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and danger[nr, nc] > best_danger:
                best_action = action
                best_danger = danger[nr, nc]
        return best_action

    def _reward(self, prev_obs, obs, action, invalid_action, done, gated_bomb=False):
        w = self.reward_weights
        prev_board, prev_players, prev_bombs, _ = normalize_obs(prev_obs)
        board, players, bombs, _ = normalize_obs(obs)
        prev_alive = bool(prev_players[self.agent_id, 2])
        alive = bool(players[self.agent_id, 2])
        components = {key: 0.0 for key in w}
        metrics = {
            "boxes_destroyed": 0,
            "items_collected": 0,
            "enemies_eliminated": 0,
            "place_bomb": 0,
            "useful_bomb": 0,
            "bomb_escape_success": 0,
            "bomb_suicide": 0,
        }

        if alive:
            components["survival_step"] = w["survival_step"]
        if prev_alive and not alive:
            components["death"] = w["death"]
        prev_enemy_alive = sum(int(p[2]) for i, p in enumerate(prev_players) if i != self.agent_id)
        enemy_alive = sum(int(p[2]) for i, p in enumerate(players) if i != self.agent_id)
        metrics["enemies_eliminated"] = max(0, prev_enemy_alive - enemy_alive)
        components["enemy_eliminated"] = metrics["enemies_eliminated"] * w["enemy_eliminated"]
        metrics["boxes_destroyed"] = max(0, int((prev_board == 2).sum()) - int((board == 2).sum()))

        prev_power = int(prev_players[self.agent_id, 3]) + int(prev_players[self.agent_id, 4])
        power = int(players[self.agent_id, 3]) + int(players[self.agent_id, 4])
        metrics["items_collected"] = max(0, power - prev_power)
        components["collect_item"] = metrics["items_collected"] * w["collect_item"]

        pos = self._self_pos(obs)
        if pos not in self.visited:
            components["enter_new_cell"] = w["enter_new_cell"]
        prev_area = reachable_area(prev_board, bomb_positions(prev_bombs), compute_danger_map(prev_board, prev_players, prev_bombs), self._self_pos(prev_obs)).sum()
        area = reachable_area(board, bomb_positions(bombs), compute_danger_map(board, players, bombs), pos).sum()
        if area > prev_area:
            components["increase_reachable_area"] = w["increase_reachable_area"]
        self.visited.add(pos)

        danger = compute_danger_map(board, players, bombs)
        if alive and danger[pos[0], pos[1]] <= 2:
            components["standing_in_danger"] = w["standing_in_danger"]
        if invalid_action:
            components["invalid_action"] = w["invalid_action"]
        if gated_bomb:
            components["useless_bomb"] += w["useless_bomb"]
        if action == STOP:
            components["excessive_stop"] = w["excessive_stop"]
        if pos in self.position_history:
            components["repeated_position"] = w["repeated_position"]
        self.position_history.append(pos)

        if action == PLACE_BOMB:
            prev_pos = self._self_pos(prev_obs)
            expected_boxes = boxes_in_blast(prev_board, prev_players, prev_pos[0], prev_pos[1], self.agent_id)
            metrics["place_bomb"] = 1
            if expected_boxes > 0:
                metrics["useful_bomb"] = 1
                self.self_bomb_credit_steps = 8
            else:
                components["useless_bomb"] = w["useless_bomb"]
            if not has_escape_after_bomb(prev_board, prev_players, prev_bombs, self.agent_id):
                components["bomb_without_escape"] = w["bomb_without_escape"]
            self.bomb_events.append({
                "placed_step": int(self.env.current_step),
                "position": prev_pos,
                "initial_boxes": int((prev_board == 2).sum()),
                "expected_boxes": int(expected_boxes),
                "early_death_penalized": False,
                "resolved": False,
            })
        current_boxes = int((board == 2).sum())
        pos = self._self_pos(obs)
        for event in self.bomb_events:
            age = int(self.env.current_step) - event["placed_step"]
            if prev_alive and not alive and 0 <= age <= 7 and not event.get("early_death_penalized", False):
                components["post_bomb_early_death"] += w["post_bomb_early_death"]
                event["early_death_penalized"] = True
                if event["expected_boxes"] > 0:
                    components["bomb_suicide"] += w["bomb_suicide"]
                    metrics["bomb_suicide"] += 1
            if alive and 0 <= age <= 5:
                bomb_row, bomb_col = event["position"]
                prev_pos = self._self_pos(prev_obs)
                prev_dist = abs(prev_pos[0] - bomb_row) + abs(prev_pos[1] - bomb_col)
                curr_dist = abs(pos[0] - bomb_row) + abs(pos[1] - bomb_col)
                if curr_dist > prev_dist:
                    components["post_bomb_move_away"] += w["post_bomb_move_away"]
                if (pos[0] == bomb_row or pos[1] == bomb_col) and curr_dist <= 3:
                    components["post_bomb_corridor_stay"] += w["post_bomb_corridor_stay"]
            if event["resolved"] or int(self.env.current_step) - event["placed_step"] < 8:
                continue
            event["resolved"] = True
            destroyed = max(0, event["initial_boxes"] - current_boxes)
            escaped_danger = alive and danger[pos[0], pos[1]] > 2
            if escaped_danger:
                components["successful_bomb_escape"] += w["successful_bomb_escape"]
                metrics["bomb_escape_success"] += 1
            elif alive:
                components["trapped_after_bomb"] += w["trapped_after_bomb"]
            elif event["expected_boxes"] > 0:
                components["bomb_suicide"] += w["bomb_suicide"]
                metrics["bomb_suicide"] += 1
            if destroyed > 0 and event["expected_boxes"] > 0 and escaped_danger:
                components["good_bomb_value"] += w["good_bomb_value"]
                components["bomb_destroy_box"] += destroyed * w["bomb_destroy_box"]
        self.bomb_events = [event for event in self.bomb_events if not event["resolved"]]
        if self.self_bomb_credit_steps > 0:
            self.self_bomb_credit_steps -= 1

        survivors = [i for i, p in enumerate(players) if int(p[2])]
        if done and survivors == [self.agent_id]:
            components["win"] = w["win"]
            components["last_survivor_bonus"] = w["last_survivor_bonus"]

        components["_metrics"] = metrics
        return float(sum(value for key, value in components.items() if key != "_metrics")), components

    def _info(self, obs, components, invalid_action=False):
        players = np.asarray(obs["players"])
        alive = bool(players[self.agent_id, 2])
        survivors = [i for i, p in enumerate(players) if int(p[2])]
        metrics = components.get("_metrics", {}) if isinstance(components, dict) else {}
        clean_components = {key: value for key, value in dict(components).items() if key != "_metrics"}
        return {
            "alive": alive,
            "win": survivors == [self.agent_id],
            "loss": not alive,
            "draw": len(survivors) > 1 and self.agent_id in survivors,
            "invalid_action": bool(invalid_action),
            "survival_step": int(self.env.current_step),
            "reward_components": clean_components,
            "episode_reward": float(self.episode_reward),
            "boxes_destroyed": int(metrics.get("boxes_destroyed", 0)),
            "items_collected": int(metrics.get("items_collected", 0)),
            "enemies_eliminated": int(metrics.get("enemies_eliminated", 0)),
            "place_bomb": int(metrics.get("place_bomb", 0)),
            "useful_bomb": int(metrics.get("useful_bomb", 0)),
            "bomb_escape_success": int(metrics.get("bomb_escape_success", 0)),
            "bomb_suicide": int(metrics.get("bomb_suicide", 0)),
        }
