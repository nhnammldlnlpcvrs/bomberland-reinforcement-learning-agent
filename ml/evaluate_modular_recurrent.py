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

from agent import RandomAgent, SimpleRuleAgent
from agent.rl_agent_pure.action_mask import legal_action_mask
from agent.rl_agent_pure.constants import PLACE_BOMB
from agent.rl_agent_pure.utils import boxes_in_blast, has_escape_after_bomb, normalize_obs
from competition.evaluation.runtime_guard import load_agent_instance
from engine.game import BomberEnv


OPPONENTS = {
    "random": "RandomAgent",
    "simple": "SimpleRuleAgent",
}
BASELINES = {
    "RandomAgent": RandomAgent,
    "random": RandomAgent,
    "SimpleRuleAgent": SimpleRuleAgent,
    "simple": SimpleRuleAgent,
}
LOCAL_MODULE_NAMES = {
    "agent",
    "constants",
    "utils",
    "encoder",
    "action_mask",
    "modular_model",
}
MODULAR_FILES = (
    "agent.py",
    "constants.py",
    "utils.py",
    "encoder.py",
    "action_mask.py",
    "modular_model.py",
    "metadata.json",
)


def clear_import_state() -> None:
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


def prepare_modular_agent(checkpoint: str, threshold: float, value_threshold: float | None = None) -> str:
    source_dir = ROOT / "agent" / "rl_agent_recurrent_modular"
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Modular checkpoint not found: {checkpoint}")
    suffix = f"{ckpt.stem}_thr{threshold:.2f}"
    if value_threshold is not None:
        suffix += f"_v{value_threshold:.2f}"
    eval_dir = ROOT / "ml" / "checkpoints" / "rl_agent_recurrent_modular" / "eval_agents" / suffix.replace(".", "p")
    eval_dir.mkdir(parents=True, exist_ok=True)
    for filename in MODULAR_FILES:
        shutil.copy2(source_dir / filename, eval_dir / filename)
    shutil.copy2(ckpt, eval_dir / "modular_policy.pt")
    (eval_dir / "metadata.json").write_text(
        json.dumps(
            {
                "team_id": "rl_agent_recurrent_modular",
                "checkpoint": "modular_policy.pt",
                "bomb_threshold": float(threshold),
                "bomb_value_threshold": float(value_threshold) if value_threshold is not None else 0.0,
                "escape_context_steps": 7,
                "research_only": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(eval_dir)


def make_agents(roster, seed=None):
    if seed is not None:
        random.seed(seed)
    agents = []
    for idx, agent_path in enumerate(roster):
        if agent_path in BASELINES:
            agents.append(BASELINES[agent_path](idx))
            continue
        path = Path(agent_path)
        if path.is_dir():
            path = path / "agent.py"
        clear_import_state()
        agents.append(load_agent_instance(str(path), idx))
    clear_import_state()
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


def _empty_activation():
    return {
        "total_states": 0,
        "movement_head_usage": 0,
        "bomb_head_usage": 0,
        "escape_head_usage": 0,
        "bomb_head_activation_frequency": 0,
        "bomb_action_accepted": 0,
        "bomb_action_rejected_illegal": 0,
        "inference_errors": 0,
        "fallback_random": 0,
        "bombs_with_escape_available": 0,
        "bombs_without_escape_available": 0,
        "bombs_destroying_boxes": 0,
        "bombs_with_zero_value": 0,
        "escape_survived_after_bomb": 0,
        "escape_died_after_bomb": 0,
    }


def _merge_counters(dst, counters):
    mapping = {
        "total_states": "total_states",
        "movement_head_used": "movement_head_usage",
        "bomb_head_used": "bomb_head_usage",
        "escape_head_used": "escape_head_usage",
        "bomb_head_activated": "bomb_head_activation_frequency",
        "bomb_action_accepted": "bomb_action_accepted",
        "bomb_action_rejected_illegal": "bomb_action_rejected_illegal",
        "inference_errors": "inference_errors",
        "fallback_random": "fallback_random",
    }
    for src, dest in mapping.items():
        dst[dest] += int(counters.get(src, 0))


def evaluate_modular(agent_dir: str, opponent: str, episodes: int, max_steps: int, seed: int) -> dict:
    opp_path = OPPONENTS.get(opponent, opponent)
    totals = {
        "wins": 0,
        "draws": 0,
        "deaths": 0,
        "invalid_actions": 0,
        "crashes": 0,
        "timeouts": 0,
        "place_bomb_count": 0,
        "useful_bomb_count": 0,
        "useless_bomb_count": 0,
        "bomb_suicide_count": 0,
        "bomb_escape_total": 0,
        "boxes_destroyed_after_bomb": 0,
        "death_within_7_steps_after_bomb": 0,
        "post_bomb_survival_steps": 0,
    }
    activation = _empty_activation()
    action_counts = {str(i): 0 for i in range(6)}
    survival_sum = score_sum = rank_sum = 0.0
    env = BomberEnv(max_steps=max_steps, seed=seed)
    for ep in range(episodes):
        roster = [agent_dir, opp_path, opp_path, opp_path]
        rng = random.Random(seed + ep)
        rng.shuffle(roster)
        slot = roster.index(agent_dir)
        agents = make_agents(roster, seed=seed + ep)
        obs = {**env.reset(seed=seed + ep), "step": 0}
        prev_alive = [bool(p[2]) for p in obs["players"]]
        death_groups = []
        survival = [0] * 4
        bomb_events = []
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
            action_counts[str(candidate_action if 0 <= candidate_action <= 5 else 0)] += 1

            if candidate_action == PLACE_BOMB and not invalid_action:
                board, players, bombs, _ = normalize_obs(prev_obs)
                row, col = int(players[slot, 0]), int(players[slot, 1])
                useful = boxes_in_blast(board, players, row, col, slot) > 0
                escape_available = has_escape_after_bomb(board, players, bombs, slot)
                totals["place_bomb_count"] += 1
                totals["useful_bomb_count" if useful else "useless_bomb_count"] += 1
                activation["bombs_destroying_boxes" if useful else "bombs_with_zero_value"] += 1
                activation["bombs_with_escape_available" if escape_available else "bombs_without_escape_available"] += 1
                bomb_events.append(
                    {
                        "start_step": step,
                        "initial_boxes": int((board == 2).sum()),
                        "resolved": False,
                        "death_recorded": False,
                    }
                )

            board_now = np.asarray(obs["map"])
            alive_now = bool(obs["players"][slot, 2])
            for event in bomb_events:
                age = step - event["start_step"]
                if not event["death_recorded"] and not alive_now and 0 <= age <= 7:
                    event["death_recorded"] = True
                    totals["death_within_7_steps_after_bomb"] += 1
                    totals["post_bomb_survival_steps"] += max(0, age)
                    activation["escape_died_after_bomb"] += 1
                if event["resolved"] or age < 8:
                    continue
                event["resolved"] = True
                totals["bomb_escape_total"] += 1
                totals["boxes_destroyed_after_bomb"] += max(0, event["initial_boxes"] - int((board_now == 2).sum()))
                if alive_now:
                    activation["escape_survived_after_bomb"] += 1
                    totals["post_bomb_survival_steps"] += 8
                else:
                    totals["bomb_suicide_count"] += 1

            alive = [bool(p[2]) for p in obs["players"]]
            deaths = []
            for idx in range(4):
                if prev_alive[idx] and not alive[idx]:
                    deaths.append(idx)
                    survival[idx] = step
            if deaths:
                death_groups.append(deaths)
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
        _merge_counters(activation, getattr(agents[slot], "debug_counters", {}))

    n = max(1, episodes)
    total_actions = max(1, sum(action_counts.values()))
    activation["bomb_head_activation_rate"] = activation["bomb_head_activation_frequency"] / max(1, activation["total_states"])
    activation["bomb_head_usage_rate"] = activation["bomb_head_usage"] / max(1, activation["total_states"])
    activation["escape_head_usage_rate"] = activation["escape_head_usage"] / max(1, activation["total_states"])
    return {
        "opponent": opponent,
        "episodes": episodes,
        "win_rate": totals["wins"] / n,
        "draw_rate": totals["draws"] / n,
        "death_rate": totals["deaths"] / n,
        "average_score": score_sum / n,
        "average_rank": rank_sum / n,
        "average_survival_step": survival_sum / n,
        "invalid_action_count": totals["invalid_actions"],
        "crash_count": totals["crashes"],
        "timeout_count": totals["timeouts"],
        "place_bomb_count": totals["place_bomb_count"],
        "place_bomb_frequency": totals["place_bomb_count"] / total_actions,
        "useful_bomb_count": totals["useful_bomb_count"],
        "useless_bomb_count": totals["useless_bomb_count"],
        "bomb_suicide_rate": totals["bomb_suicide_count"] / max(1, totals["bomb_escape_total"]),
        "death_within_7_steps_after_bomb": totals["death_within_7_steps_after_bomb"],
        "post_bomb_survival_steps_avg": totals["post_bomb_survival_steps"] / max(1, totals["place_bomb_count"]),
        "average_boxes_destroyed_per_bomb": totals["boxes_destroyed_after_bomb"] / max(1, totals["place_bomb_count"]),
        "action_counts": action_counts,
        "activation": activation,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate research-only modular recurrent BC agent.")
    parser.add_argument("--checkpoint", default="ml/checkpoints/rl_agent_recurrent/modular_bomb_calibrated_escape_refined.pt")
    parser.add_argument("--thresholds", default="0.45,0.5")
    parser.add_argument("--value_thresholds", default="")
    parser.add_argument("--opponents", nargs="+", default=["random", "simple"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=9800)
    parser.add_argument("--output", default="logs/modular_recurrent_eval.json")
    parser.add_argument("--activation_output", default="logs/modular_activation_analysis.json")
    args = parser.parse_args()

    all_results = []
    activation_report = {}
    threshold_values = [float(v) for v in args.thresholds.split(",")]
    value_values = [float(v) for v in args.value_thresholds.split(",") if v.strip()]
    pairs = [(s, None) for s in threshold_values] if not value_values else [(s, v) for s in threshold_values for v in value_values]
    for threshold, value_threshold in pairs:
        agent_dir = prepare_modular_agent(args.checkpoint, threshold, value_threshold)
        threshold_rows = []
        for idx, opponent in enumerate(args.opponents):
            result = evaluate_modular(agent_dir, opponent, args.episodes, args.max_steps, args.seed + idx * 1000)
            result["threshold"] = threshold
            result["value_threshold"] = value_threshold
            threshold_rows.append(result)
            all_results.append(result)
        key = str(threshold) if value_threshold is None else f"{threshold}|{value_threshold}"
        activation_report[key] = {row["opponent"]: row["activation"] for row in threshold_rows}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    Path(args.activation_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.activation_output).write_text(json.dumps(activation_report, indent=2), encoding="utf-8")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
