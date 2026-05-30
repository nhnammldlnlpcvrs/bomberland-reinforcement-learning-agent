import numpy as np

from constants import A_STOP, ALL_ACTIONS, INVALID_SCORE, RL_WEIGHT
from memory import AgentMemory
from rl_policy import RLPolicy
from rule_policy import score_actions
from safety import current_position_is_danger, escape_action, get_safe_actions
from utils import bomb_set, my_state


class Agent:
    team_id = "HybridAgentRL"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.memory = AgentMemory(agent_id)
        self.rl = RLPolicy()
        self.rl_weight = RL_WEIGHT

    def act(self, obs: dict) -> int:
        try:
            r, c, alive, _bombs_left, _bonus = my_state(obs, self.agent_id)
            if not alive:
                return A_STOP
            pos = (r, c)
            bset = bomb_set(obs["bombs"])
            self.memory.maybe_reset(pos, bset)
            self.memory.observe_position(pos)

            if current_position_is_danger(obs, self.agent_id):
                action = escape_action(obs, self.agent_id, self.memory)
                self.memory.observe_action(action)
                return int(action)

            safe_actions = get_safe_actions(obs, self.agent_id)
            rule_scores, mask, _danger = score_actions(
                obs, self.agent_id, safe_actions=safe_actions, memory=self.memory
            )
            final = dict(rule_scores)
            logits = self.rl.predict(obs, self.agent_id)
            if logits is not None:
                logits = np.asarray(logits, dtype=np.float32)
                logits = logits - float(np.max(logits))
                for action in ALL_ACTIONS:
                    if bool(mask[action]) and final[action] > INVALID_SCORE / 2:
                        final[action] += self.rl_weight * float(logits[action])

            best_action = A_STOP
            best_score = INVALID_SCORE
            for action in ALL_ACTIONS:
                if bool(mask[action]) and final[action] > best_score:
                    best_action = action
                    best_score = final[action]
            if best_action == 5:
                self.memory.observe_bomb(pos)
            self.memory.observe_action(best_action)
            return int(best_action)
        except Exception:
            return A_STOP
