from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.action_mask import legal_action_mask
from agent.rl_agent_pure.constants import PLACE_BOMB, REWARD_WEIGHTS, STOP
from agent.rl_agent_pure.utils import boxes_in_blast, bomb_positions, compute_danger_map, has_escape_after_bomb, normalize_obs, reachable_area
from agent import BoxFarmerAgent, GeniusRuleAgent, RandomAgent, SimpleRuleAgent, SmarterRuleAgent, TacticalRuleAgent
from competition.evaluation.runtime_guard import load_agent_instance
from engine.game import BomberEnv


OPPONENTS = {
    "random": "RandomAgent",
    "simple": "SimpleRuleAgent",
    "tactical": "TacticalRuleAgent",
    "online_robust": "agent/hybrid_agent_online_robust",
    "hybrid_agent_rl": "agent/hybrid_agent_rl",
}

BASELINES = {
    "RandomAgent": RandomAgent,
    "random": RandomAgent,
    "SimpleRuleAgent": SimpleRuleAgent,
    "simple": SimpleRuleAgent,
    "SmarterRuleAgent": SmarterRuleAgent,
    "smarter": SmarterRuleAgent,
    "GeniusRuleAgent": GeniusRuleAgent,
    "genius": GeniusRuleAgent,
    "BoxFarmerAgent": BoxFarmerAgent,
    "box_farmer": BoxFarmerAgent,
    "TacticalRuleAgent": TacticalRuleAgent,
    "tactical": TacticalRuleAgent,
}

LOCAL_MODULE_NAMES = {
    "agent",
    "constants",
    "utils",
    "model",
    "policy",
    "encoder",
    "action_mask",
    "features",
    "safety",
    "rl_policy",
    "rule_policy",
    "memory",
}

RL_AGENT_FILES = (
    "agent.py",
    "constants.py",
    "utils.py",
    "model.py",
    "policy.py",
    "encoder.py",
    "action_mask.py",
)
TEMPORAL_AGENT_FILES = (
    "agent.py",
    "constants.py",
    "utils.py",
    "model.py",
    "policy.py",
    "encoder.py",
    "action_mask.py",
    "frame_buffer.py",
    "metadata.json",
)


def prepare_agent_path(agent_path: str, frame_stack: int = 1) -> str:
    path = Path(agent_path)
    if path.suffix.lower() != ".zip":
        return agent_path

    if not path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {agent_path}")

    temporal = int(frame_stack) > 1
    source_dir = ROOT / "agent" / ("rl_agent_temporal" if temporal else "rl_agent_pure")
    eval_dir = ROOT / "ml" / "checkpoints" / ("rl_agent_temporal" if temporal else "rl_agent_pure") / "eval_agents" / path.stem
    eval_dir.mkdir(parents=True, exist_ok=True)
    for filename in (TEMPORAL_AGENT_FILES if temporal else RL_AGENT_FILES):
        shutil.copy2(source_dir / filename, eval_dir / filename)
    shutil.copy2(path, eval_dir / "policy.zip")
    if temporal:
        (eval_dir / "metadata.json").write_text(
            json.dumps({
                "frame_stack": int(frame_stack),
                "base_channels": 19,
                "observation_shape": [int(frame_stack) * 19, 13, 13],
                "note": "Temporary eval packaging for frame-stacked research checkpoint."
            }, indent=2),
            encoding="utf-8",
        )
    return str(eval_dir)


def _clear_submission_import_state():
    for name in LOCAL_MODULE_NAMES:
        sys.modules.pop(name, None)
    agent_root = ROOT / "agent"
    cleaned = []
    for entry in sys.path:
        try:
            path = Path(entry).resolve()
        except Exception:
            cleaned.append(entry)
            continue
        if path.parent == agent_root:
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned


def make_eval_agents(agent_paths, seed=None):
    if seed is not None:
        random.seed(seed)
    agents = []
    for idx, agent_path in enumerate(agent_paths):
        if agent_path in BASELINES:
            agents.append(BASELINES[agent_path](idx))
            continue
        path = Path(agent_path)
        if path.is_dir():
            path = path / "agent.py"
        _clear_submission_import_state()
        agents.append(load_agent_instance(str(path), idx))
    _clear_submission_import_state()
    return agents


def final_ranks(death_groups, survivors):
    ranks = [0] * 4
    groups = list(death_groups)
    if survivors:
        groups.append(list(survivors))
    for rank, group in enumerate(reversed(groups)):
        for idx in group:
            ranks[idx] = rank
    return ranks


def reward_components(prev_obs, obs, agent_id, action, invalid_action, done, visited, position_history, bomb_credit):
    w = REWARD_WEIGHTS
    prev_board, prev_players, prev_bombs, _ = normalize_obs(prev_obs)
    board, players, bombs, _ = normalize_obs(obs)
    components = {key: 0.0 for key in w}
    prev_alive = bool(prev_players[agent_id, 2])
    alive = bool(players[agent_id, 2])
    if alive:
        components["survival_step"] = w["survival_step"]
    if prev_alive and not alive:
        components["death"] = w["death"]

    prev_enemy_alive = sum(int(p[2]) for i, p in enumerate(prev_players) if i != agent_id)
    enemy_alive = sum(int(p[2]) for i, p in enumerate(players) if i != agent_id)
    components["enemy_eliminated"] = max(0, prev_enemy_alive - enemy_alive) * w["enemy_eliminated"]
    boxes_destroyed = max(0, int((prev_board == 2).sum()) - int((board == 2).sum()))
    if bomb_credit[0] > 0:
        components["destroy_box"] = boxes_destroyed * w["destroy_box"]

    prev_power = int(prev_players[agent_id, 3]) + int(prev_players[agent_id, 4])
    power = int(players[agent_id, 3]) + int(players[agent_id, 4])
    components["collect_item"] = max(0, power - prev_power) * w["collect_item"]

    pos = (int(players[agent_id, 0]), int(players[agent_id, 1]))
    prev_pos = (int(prev_players[agent_id, 0]), int(prev_players[agent_id, 1]))
    if pos not in visited:
        components["enter_new_cell"] = w["enter_new_cell"]
    prev_area = reachable_area(prev_board, bomb_positions(prev_bombs), compute_danger_map(prev_board, prev_players, prev_bombs), prev_pos).sum()
    area = reachable_area(board, bomb_positions(bombs), compute_danger_map(board, players, bombs), pos).sum()
    if area > prev_area:
        components["increase_reachable_area"] = w["increase_reachable_area"]
    visited.add(pos)

    danger = compute_danger_map(board, players, bombs)
    if alive and danger[pos[0], pos[1]] <= 2:
        components["standing_in_danger"] = w["standing_in_danger"]
    if invalid_action:
        components["invalid_action"] = w["invalid_action"]
    if action == STOP:
        components["excessive_stop"] = w["excessive_stop"]
    if pos in position_history:
        components["repeated_position"] = w["repeated_position"]
    position_history.append(pos)
    if len(position_history) > 8:
        del position_history[:-8]

    if action == PLACE_BOMB:
        if boxes_in_blast(prev_board, prev_players, prev_pos[0], prev_pos[1], agent_id) > 0:
            components["good_bomb_value"] = w["good_bomb_value"]
            bomb_credit[0] = 8
        else:
            components["useless_bomb"] = w["useless_bomb"]
        if not has_escape_after_bomb(prev_board, prev_players, prev_bombs, agent_id):
            components["bomb_without_escape"] = w["bomb_without_escape"]
    if bomb_credit[0] > 0:
        bomb_credit[0] -= 1

    survivors = [i for i, p in enumerate(players) if int(p[2])]
    if done and survivors == [agent_id]:
        components["win"] = w["win"]
        components["last_survivor_bonus"] = w["last_survivor_bonus"]
    return components


def evaluate(agent_path, opponent, episodes, max_steps, seed, frame_stack=1):
    agent_path = prepare_agent_path(agent_path, frame_stack=frame_stack)
    opp_path = OPPONENTS.get(opponent, opponent)
    totals = {k: 0 for k in (
        "wins",
        "draws",
        "deaths",
        "invalid_actions",
        "crashes",
        "timeouts",
        "items_collected",
        "boxes_destroyed",
        "enemy_eliminations",
        "place_bomb_count",
        "useful_bomb_count",
        "useless_bomb_count",
        "bomb_escape_success",
        "bomb_escape_total",
        "bomb_suicide_count",
        "boxes_destroyed_after_bomb",
        "death_within_7_steps_after_bomb",
        "post_bomb_survival_steps",
    )}
    rank_sum = survival_sum = score_sum = reward_sum = 0.0
    component_sums = {key: 0.0 for key in REWARD_WEIGHTS}
    action_counts = {str(action): 0 for action in range(6)}
    env = BomberEnv(max_steps=max_steps, seed=seed)
    for ep in range(episodes):
        roster = [agent_path, opp_path, opp_path, opp_path]
        rng = random.Random(seed + ep)
        rng.shuffle(roster)
        slot = roster.index(agent_path)
        agents = make_eval_agents(roster, seed=seed + ep)
        obs = {**env.reset(seed=seed + ep), "step": 0}
        initial_boxes = int((np.asarray(obs["map"]) == 2).sum())
        initial_power = int(obs["players"][slot, 3]) + int(obs["players"][slot, 4])
        prev_alive = [bool(p[2]) for p in obs["players"]]
        prev_enemy_alive = sum(prev_alive) - int(prev_alive[slot])
        visited = {(int(obs["players"][slot, 0]), int(obs["players"][slot, 1]))}
        position_history = list(visited)
        bomb_credit = [0]
        bomb_events = []
        death_groups = []
        survival = [0] * 4
        done = False
        step = 0
        while not done and step < max_steps:
            prev_obs = obs
            actions = []
            candidate_action = 0
            invalid_action = False
            for idx, agent in enumerate(agents):
                started = time.perf_counter()
                try:
                    action = int(agent.act(obs))
                except Exception:
                    action = 0
                    if idx == slot:
                        totals["crashes"] += 1
                if idx == slot and (time.perf_counter() - started) > 0.1:
                    totals["timeouts"] += 1
                if idx == slot:
                    candidate_action = action
                    invalid_action = not 0 <= action <= 5 or not legal_action_mask(obs, slot)[action]
                    if invalid_action:
                        totals["invalid_actions"] += 1
                actions.append(action if 0 <= action <= 5 else 0)
            obs, terminated, truncated = env.step(actions)
            obs = {**obs, "step": step + 1}
            done = terminated or truncated
            step += 1
            if candidate_action == PLACE_BOMB and not invalid_action:
                prev_board, prev_players, _prev_bombs, _ = normalize_obs(prev_obs)
                prev_pos = (int(prev_players[slot, 0]), int(prev_players[slot, 1]))
                useful = boxes_in_blast(prev_board, prev_players, prev_pos[0], prev_pos[1], slot) > 0
                totals["place_bomb_count"] += 1
                totals["useful_bomb_count" if useful else "useless_bomb_count"] += 1
                bomb_events.append({
                    "start_step": step,
                    "initial_boxes": int((prev_board == 2).sum()),
                    "resolved": False,
                    "death_recorded": False,
                })
            board_now = np.asarray(obs["map"])
            alive_now_for_bombs = bool(obs["players"][slot, 2])
            for event in bomb_events:
                age = step - event["start_step"]
                if not event["death_recorded"] and not alive_now_for_bombs and 0 <= age <= 7:
                    event["death_recorded"] = True
                    totals["death_within_7_steps_after_bomb"] += 1
                    totals["post_bomb_survival_steps"] += max(0, age)
                if event["resolved"] or step - event["start_step"] < 8:
                    continue
                event["resolved"] = True
                totals["bomb_escape_total"] += 1
                if alive_now_for_bombs:
                    totals["bomb_escape_success"] += 1
                    totals["post_bomb_survival_steps"] += 8
                else:
                    totals["bomb_suicide_count"] += 1
                totals["boxes_destroyed_after_bomb"] += max(0, event["initial_boxes"] - int((board_now == 2).sum()))
            action_counts[str(candidate_action if 0 <= candidate_action <= 5 else 0)] += 1
            components = reward_components(prev_obs, obs, slot, candidate_action, invalid_action, done, visited, position_history, bomb_credit)
            reward_sum += sum(components.values())
            for key, value in components.items():
                component_sums[key] += value
            alive = [bool(p[2]) for p in obs["players"]]
            deaths = []
            for idx in range(4):
                if prev_alive[idx] and not alive[idx]:
                    deaths.append(idx)
                    survival[idx] = step
            if deaths:
                death_groups.append(deaths)
            enemy_alive = sum(alive) - int(alive[slot])
            totals["enemy_eliminations"] += max(0, prev_enemy_alive - enemy_alive)
            prev_enemy_alive = enemy_alive
            prev_alive = alive
        survivors = [idx for idx, alive in enumerate(prev_alive) if alive]
        for idx in survivors:
            survival[idx] = step
        ranks = final_ranks(death_groups, survivors)
        if survivors == [slot]:
            totals["wins"] += 1
        elif slot in survivors:
            totals["draws"] += 1
        if slot not in survivors:
            totals["deaths"] += 1
        rank_sum += ranks[slot]
        survival_sum += survival[slot]
        score_sum += 3 - ranks[slot]
        totals["boxes_destroyed"] += max(0, initial_boxes - int((np.asarray(obs["map"]) == 2).sum()))
        final_power = int(obs["players"][slot, 3]) + int(obs["players"][slot, 4])
        totals["items_collected"] += max(0, final_power - initial_power)
    n = max(1, episodes)
    return {
        "opponent": opponent,
        "episodes": episodes,
        "win_rate": totals["wins"] / n,
        "draw_rate": totals["draws"] / n,
        "death_rate": totals["deaths"] / n,
        "average_score": score_sum / n,
        "average_reward": reward_sum / n,
        "average_rank": rank_sum / n,
        "average_survival_step": survival_sum / n,
        "invalid_action_count": totals["invalid_actions"],
        "item_collected": totals["items_collected"],
        "boxes_destroyed": totals["boxes_destroyed"],
        "enemy_eliminations": totals["enemy_eliminations"],
        "crash_count": totals["crashes"],
        "timeout_count": totals["timeouts"],
        "place_bomb_count": totals["place_bomb_count"],
        "place_bomb_frequency": totals["place_bomb_count"] / max(1, sum(action_counts.values())),
        "useful_bomb_count": totals["useful_bomb_count"],
        "useless_bomb_count": totals["useless_bomb_count"],
        "bomb_escape_success_rate": totals["bomb_escape_success"] / max(1, totals["bomb_escape_total"]),
        "bomb_suicide_rate": totals["bomb_suicide_count"] / max(1, totals["bomb_escape_total"]),
        "death_within_7_steps_after_bomb": totals["death_within_7_steps_after_bomb"],
        "post_bomb_survival_steps_avg": totals["post_bomb_survival_steps"] / max(1, totals["place_bomb_count"]),
        "average_boxes_destroyed_per_bomb": totals["boxes_destroyed_after_bomb"] / max(1, totals["place_bomb_count"]),
        "action_counts": action_counts,
        "reward_components_avg": {key: value / n for key, value in component_sums.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_path", default="agent/rl_agent_pure")
    parser.add_argument("--opponents", nargs="+", default=["random", "simple", "online_robust", "hybrid_agent_rl"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="logs/rl_agent_pure_eval.json")
    parser.add_argument("--frame_stack", type=int, default=1)
    args = parser.parse_args()
    results = [evaluate(args.agent_path, opponent, args.episodes, args.max_steps, args.seed, frame_stack=args.frame_stack) for opponent in args.opponents]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
