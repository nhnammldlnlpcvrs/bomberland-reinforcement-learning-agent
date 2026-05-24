"""Shadow evaluation for neural action ranker.

Runs the neural prior model in parallel with the production heuristic agent.
The heuristic agent makes ALL real actions; the neural model only predicts.
Measures agreement, disagreement patterns, and strategic usefulness.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine.game import BomberEnv
from agent import RandomAgent, SimpleRuleAgent, SmarterRuleAgent, GeniusRuleAgent, BoxFarmerAgent, TacticalRuleAgent
from competition.evaluation.runtime_guard import load_agent_instance
from ml.features import encode_observation
from ml.models.simple_cnn_policy import TORCH_AVAILABLE, load_checkpoint

if TORCH_AVAILABLE:
    import torch
else:
    torch = None

ACTION_NAMES = ["STOP", "LEFT", "RIGHT", "UP", "DOWN", "PLACE_BOMB"]
BOARD_SIZE = 13
TILE_WALL = 1
TILE_BOX = 2

DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


# ==============================================================================
# Safety mask
# ==============================================================================


def compute_validity_mask(obs, agent_id):
    """Lightweight validity mask — which actions are physically possible.

    STOP (0):        always valid
    MOVE (1-4):      valid if destination in-bounds, not wall/box/bomb
    BOMB (5):        valid if bombs_left > 0 and not standing on a bomb
    """
    mask = np.zeros(6, dtype=bool)
    mask[0] = True

    p = obs["players"][agent_id]
    my_r, my_c = int(p[0]), int(p[1])
    alive = int(p[2])
    bombs_left = int(p[3])

    if not alive:
        return mask

    game_map = obs["map"]

    bombs_arr = np.asarray(obs["bombs"], dtype=np.int32)
    if bombs_arr.size == 0:
        bomb_set = set()
    else:
        if bombs_arr.ndim == 1:
            bombs_arr = bombs_arr.reshape(1, -1)
        bomb_set = {(int(bombs_arr[i, 0]), int(bombs_arr[i, 1])) for i in range(bombs_arr.shape[0])}

    for action, (dr, dc) in DIRS.items():
        nr, nc = my_r + dr, my_c + dc
        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
            tile = game_map[nr, nc]
            if tile not in (TILE_WALL, TILE_BOX) and (nr, nc) not in bomb_set:
                mask[action] = True

    if bombs_left > 0 and (my_r, my_c) not in bomb_set:
        mask[5] = True

    return mask


# ==============================================================================
# Neural predictor
# ==============================================================================


class NeuralShadow:
    """Loads a neural ranker checkpoint and runs inference on demand."""

    def __init__(self, checkpoint_path):
        self.model = None
        self.checkpoint_metrics = None
        self.available = False

        if not TORCH_AVAILABLE:
            self.error = "PyTorch not installed"
            return

        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            self.error = f"checkpoint not found: {checkpoint_path}"
            return

        try:
            self.model, checkpoint = load_checkpoint(str(ckpt), map_location="cpu")
            self.model.eval()
            self.checkpoint_metrics = checkpoint.get("best_metrics", {})
            self.available = True
        except Exception as exc:
            self.error = f"failed to load checkpoint: {exc}"

    def predict(self, obs, agent_id):
        """Run inference. Returns dict with logits, raw_preds, masked_preds, etc."""
        if not self.available:
            return None

        try:
            obs_copy = dict(obs)
            obs_copy["_agent_index"] = agent_id
            encoded = encode_observation(obs_copy)
            tensor = torch.from_numpy(encoded["tensor"]).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(tensor)[0].numpy().astype(np.float64)

            mask = compute_validity_mask(obs, agent_id)

            # Raw predictions
            raw_probs = _softmax(logits)
            raw_entropy = _entropy(raw_probs)
            max_entropy = np.log(max(1, np.count_nonzero(np.isfinite(logits))))
            raw_entropy_norm = raw_entropy / max_entropy if max_entropy > 0 else 0.0
            raw_order = np.argsort(-logits)
            raw_top1 = int(raw_order[0])
            raw_top2 = [int(raw_order[0]), int(raw_order[1])]

            # Masked predictions
            masked_logits = logits.copy()
            masked_logits[~mask] = -1e9
            masked_probs = _softmax(masked_logits)
            masked_entropy = _entropy(masked_probs)
            masked_max_entropy = np.log(max(1, np.count_nonzero(mask)))
            masked_entropy_norm = masked_entropy / masked_max_entropy if masked_max_entropy > 0 else 0.0
            masked_order = np.argsort(-masked_logits)
            masked_top1 = int(masked_order[0])
            masked_top2 = [int(masked_order[0]), int(masked_order[1])]

            return {
                "raw": {
                    "top1": raw_top1,
                    "top1_name": ACTION_NAMES[raw_top1],
                    "top2": raw_top2,
                    "top2_names": [ACTION_NAMES[t] for t in raw_top2],
                    "logits": [float(x) for x in logits],
                    "probs": [float(x) for x in raw_probs],
                    "entropy": float(raw_entropy),
                    "entropy_normalized": float(raw_entropy_norm),
                },
                "masked": {
                    "top1": masked_top1,
                    "top1_name": ACTION_NAMES[masked_top1],
                    "top2": masked_top2,
                    "top2_names": [ACTION_NAMES[t] for t in masked_top2],
                    "logits": [float(x) for x in masked_logits],
                    "probs": [float(x) for x in masked_probs],
                    "entropy": float(masked_entropy),
                    "entropy_normalized": float(masked_entropy_norm),
                },
                "validity_mask": [bool(x) for x in mask],
            }
        except Exception as exc:
            return {"error": str(exc)}


def _softmax(x):
    x_shifted = x - np.max(x)
    exp = np.exp(x_shifted)
    return exp / exp.sum()


def _entropy(probs):
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


# ==============================================================================
# Match runner
# ==============================================================================


def _make_baseline_agent(path, agent_id):
    """Create a baseline agent by name."""
    mapping = {
        "RandomAgent": RandomAgent,
        "SimpleRuleAgent": SimpleRuleAgent,
        "SmarterRuleAgent": SmarterRuleAgent,
        "GeniusRuleAgent": GeniusRuleAgent,
        "BoxFarmerAgent": BoxFarmerAgent,
        "TacticalRuleAgent": TacticalRuleAgent,
    }
    if path in mapping:
        return mapping[path](agent_id), path
    raise ValueError(f"Unknown baseline agent: {path}")


def _load_heuristic_agent(agent_path, agent_id):
    """Load the production heuristic agent."""
    p = Path(agent_path)
    if p.is_dir():
        p = p / "agent.py"
    if not p.exists():
        raise FileNotFoundError(f"Agent file not found: {p}")
    agent = load_agent_instance(str(p), agent_id)
    name = getattr(agent, "team_id", p.parent.name)
    return agent, name


def _determine_outcome(agent_id, alive_final, survivors):
    if not alive_final:
        return "loss"
    if len(survivors) == 1 and agent_id in survivors:
        return "win"
    return "draw"


def _final_ranks(n_players, death_groups, survivors):
    ranks = [0] * n_players
    ordered_groups = list(death_groups)
    if survivors:
        ordered_groups.append(list(survivors))
    for rank, group in enumerate(reversed(ordered_groups)):
        for player_id in group:
            ranks[player_id] = rank
    return ranks


def run_shadow_eval(args):
    """Run matches with neural shadow evaluation."""
    # ---- Load agents ----
    heuristic_agent, heuristic_name = _load_heuristic_agent(args.agent_path, 0)
    opponents = []
    opponent_names = []
    for i, opp_path in enumerate(args.opponents):
        if opp_path in ("None", "RandomAgent", "SimpleRuleAgent", "SmarterRuleAgent",
                         "GeniusRuleAgent", "BoxFarmerAgent", "TacticalRuleAgent"):
            if opp_path == "None":
                import random
                choices = ["RandomAgent", "SimpleRuleAgent", "SmarterRuleAgent",
                           "GeniusRuleAgent", "BoxFarmerAgent", "TacticalRuleAgent"]
                opp_path = random.choice(choices)
            agent, name = _make_baseline_agent(opp_path, i + 1)
        else:
            agent, name = _load_heuristic_agent(opp_path, i + 1)
        opponents.append(agent)
        opponent_names.append(name)

    agents = [heuristic_agent] + opponents
    names = [heuristic_name] + opponent_names
    n_players = 4

    # ---- Load neural model ----
    neural = NeuralShadow(args.checkpoint)

    # ---- Run matches ----
    env = BomberEnv(max_steps=args.max_steps, seed=args.seed)
    all_episodes = []
    all_step_logs = []

    for episode in range(args.num_episodes):
        episode_seed = None if args.seed is None else args.seed + episode
        obs = env.reset(seed=episode_seed)
        done = False
        step = 0
        death_groups = []
        prev_alive = [bool(p[2]) for p in obs["players"]]
        survival_steps = [0] * n_players
        episode_steps = []

        while not done and step < args.max_steps:
            # 1. Get actions from all agents
            actions = []
            for i in range(n_players):
                try:
                    action = agents[i].act(obs)
                except Exception:
                    action = 0
                actions.append(action)

            # 2. Shadow neural evaluation (before env step, on current obs)
            neural_result = neural.predict(obs, agent_id=0)

            # 3. Build step log
            heuristic_action = int(actions[0])
            entry = {
                "step": step,
                "heuristic_action": heuristic_action,
                "heuristic_action_name": ACTION_NAMES[heuristic_action],
            }

            if neural_result is not None:
                if "error" in neural_result:
                    entry["neural_error"] = neural_result["error"]
                else:
                    entry["raw"] = neural_result["raw"]
                    entry["masked"] = neural_result["masked"]
                    entry["validity_mask"] = neural_result["validity_mask"]
                    entry["agreement"] = {
                        "raw_top1_matches_heuristic": neural_result["raw"]["top1"] == heuristic_action,
                        "raw_top2_contains_heuristic": heuristic_action in neural_result["raw"]["top2"],
                        "masked_top1_matches_heuristic": neural_result["masked"]["top1"] == heuristic_action,
                        "masked_top2_contains_heuristic": heuristic_action in neural_result["masked"]["top2"],
                    }
            episode_steps.append(entry)

            # 4. Step environment
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1

            # 5. Track deaths
            alive_now = [bool(p[2]) for p in obs["players"]]
            deaths_this_step = []
            for i in range(n_players):
                if prev_alive[i] and not alive_now[i]:
                    deaths_this_step.append(i)
                    survival_steps[i] = step
            if deaths_this_step:
                death_groups.append(deaths_this_step)
            prev_alive = alive_now

        # Episode-end
        alive_final = [bool(p[2]) for p in obs["players"]]
        survivors = [i for i in range(n_players) if alive_final[i]]
        for pid in survivors:
            survival_steps[pid] = step
        ranks = _final_ranks(n_players, death_groups, survivors)
        outcome = _determine_outcome(0, alive_final[0], survivors)
        winner_name = names[survivors[0]] if len(survivors) == 1 else "draw"

        # Compute episode summary
        ep_summary = _compute_episode_summary(episode_steps, outcome)

        episode_record = {
            "episode": episode,
            "seed": episode_seed,
            "outcome": outcome,
            "winner": winner_name,
            "total_steps": step,
            "agent_rank": int(ranks[0]),
            "survival_step": int(survival_steps[0]),
            "death_order": [names[pid] for group in death_groups for pid in group],
            "summary": ep_summary,
            "steps": episode_steps if not args.summary_only else [],
        }
        all_episodes.append(episode_record)
        all_step_logs.extend(
            {"episode": episode, **s} for s in episode_steps
        )

        if not args.summary_only:
            status = f"Ep {episode + 1}: {outcome}"
            if ep_summary:
                status += (
                    f" | raw_top1={ep_summary.get('raw_top1_agreement', 0):.2f}"
                    f" masked_top1={ep_summary.get('masked_top1_agreement', 0):.2f}"
                )
            print(status)

    # ---- Aggregate ----
    aggregate = _compute_aggregate(all_episodes, all_step_logs)

    # ---- Output ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "agent_path": args.agent_path,
            "checkpoint": args.checkpoint,
            "opponents": args.opponents,
            "num_episodes": args.num_episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
        "checkpoint_metrics": neural.checkpoint_metrics,
        "neural_available": neural.available,
        "neural_error": getattr(neural, "error", None),
        "episodes": all_episodes,
        "aggregate": aggregate,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    # ---- Console report ----
    _print_report(aggregate, neural, output_path)

    return aggregate


# ==============================================================================
# Summary metrics
# ==============================================================================


def _compute_episode_summary(step_logs, outcome):
    """Compute per-episode metrics from step logs."""
    valid_steps = [s for s in step_logs if "raw" in s]
    if not valid_steps:
        return {"outcome": outcome, "steps_with_neural": 0}

    n = len(valid_steps)
    raw_top1_agree = sum(1 for s in valid_steps if s["agreement"]["raw_top1_matches_heuristic"]) / n
    raw_top2_agree = sum(1 for s in valid_steps if s["agreement"]["raw_top2_contains_heuristic"]) / n
    masked_top1_agree = sum(1 for s in valid_steps if s["agreement"]["masked_top1_matches_heuristic"]) / n
    masked_top2_agree = sum(1 for s in valid_steps if s["agreement"]["masked_top2_contains_heuristic"]) / n

    raw_entropy = np.mean([s["raw"]["entropy_normalized"] for s in valid_steps])
    masked_entropy = np.mean([s["masked"]["entropy_normalized"] for s in valid_steps])

    # Masked-top1 differs from heuristic but is valid
    masked_diff_valid = 0
    for s in valid_steps:
        if not s["agreement"]["masked_top1_matches_heuristic"]:
            heuristic = s["heuristic_action"]
            if heuristic < len(s["validity_mask"]) and s["validity_mask"][heuristic]:
                masked_diff_valid += 1
    masked_diff_valid_rate = masked_diff_valid / n

    # Raw-top1 is invalid under mask
    raw_top1_invalid = sum(
        1 for s in valid_steps
        if s["validity_mask"] and not s["validity_mask"][s["raw"]["top1"]]
    ) / n

    # Action distributions
    heuristic_dist = np.bincount(
        [s["heuristic_action"] for s in valid_steps], minlength=6
    ).astype(np.float64) / n
    raw_top1_dist = np.bincount(
        [s["raw"]["top1"] for s in valid_steps], minlength=6
    ).astype(np.float64) / n
    masked_top1_dist = np.bincount(
        [s["masked"]["top1"] for s in valid_steps], minlength=6
    ).astype(np.float64) / n

    return {
        "outcome": outcome,
        "steps_with_neural": n,
        "total_steps": len(step_logs),
        "raw_top1_agreement": round(float(raw_top1_agree), 4),
        "raw_top2_agreement": round(float(raw_top2_agree), 4),
        "masked_top1_agreement": round(float(masked_top1_agree), 4),
        "masked_top2_agreement": round(float(masked_top2_agree), 4),
        "raw_entropy_normalized_mean": round(float(raw_entropy), 4),
        "masked_entropy_normalized_mean": round(float(masked_entropy), 4),
        "masked_top1_differs_from_heuristic_but_valid_rate": round(float(masked_diff_valid_rate), 4),
        "raw_top1_invalid_under_mask_rate": round(float(raw_top1_invalid), 4),
        "heuristic_action_dist": {ACTION_NAMES[i]: round(float(heuristic_dist[i]), 4) for i in range(6)},
        "raw_top1_dist": {ACTION_NAMES[i]: round(float(raw_top1_dist[i]), 4) for i in range(6)},
        "masked_top1_dist": {ACTION_NAMES[i]: round(float(masked_top1_dist[i]), 4) for i in range(6)},
    }


def _compute_aggregate(all_episodes, all_step_logs):
    """Compute aggregate metrics across all episodes."""
    valid_steps = [s for s in all_step_logs if "raw" in s]
    if not valid_steps:
        return {"error": "no valid neural predictions"}

    n = len(valid_steps)
    num_episodes = len(all_episodes)

    raw_top1_agree = sum(1 for s in valid_steps if s["agreement"]["raw_top1_matches_heuristic"]) / n
    raw_top2_agree = sum(1 for s in valid_steps if s["agreement"]["raw_top2_contains_heuristic"]) / n
    masked_top1_agree = sum(1 for s in valid_steps if s["agreement"]["masked_top1_matches_heuristic"]) / n
    masked_top2_agree = sum(1 for s in valid_steps if s["agreement"]["masked_top2_contains_heuristic"]) / n

    raw_entropy = np.mean([s["raw"]["entropy_normalized"] for s in valid_steps])
    masked_entropy = np.mean([s["masked"]["entropy_normalized"] for s in valid_steps])

    # masked_top1_is_different_from_heuristic_but_valid_rate
    masked_diff_valid = 0
    for s in valid_steps:
        if not s["agreement"]["masked_top1_matches_heuristic"]:
            heuristic = s["heuristic_action"]
            if heuristic < len(s["validity_mask"]) and s["validity_mask"][heuristic]:
                masked_diff_valid += 1
    masked_diff_valid_rate = masked_diff_valid / n

    # raw_top1_invalid_under_mask_rate
    raw_top1_invalid = sum(
        1 for s in valid_steps
        if s["validity_mask"] and not s["validity_mask"][s["raw"]["top1"]]
    ) / n

    # Action distributions
    heuristic_dist = np.bincount(
        [s["heuristic_action"] for s in valid_steps], minlength=6
    ).astype(np.float64) / n
    raw_top1_dist = np.bincount(
        [s["raw"]["top1"] for s in valid_steps], minlength=6
    ).astype(np.float64) / n
    masked_top1_dist = np.bincount(
        [s["masked"]["top1"] for s in valid_steps], minlength=6
    ).astype(np.float64) / n

    # Agreement by outcome
    outcome_agreement = {}
    for outcome in ("win", "draw", "loss"):
        outcome_steps = []
        for ep in all_episodes:
            if ep["outcome"] == outcome:
                for s in ep["steps"]:
                    if "raw" in s:
                        outcome_steps.append(s)
        if outcome_steps:
            outcome_agreement[outcome] = {
                "episodes": sum(1 for ep in all_episodes if ep["outcome"] == outcome),
                "steps": len(outcome_steps),
                "raw_top1_agreement": round(float(
                    sum(1 for s in outcome_steps if s["agreement"]["raw_top1_matches_heuristic"]) / len(outcome_steps)
                ), 4),
                "masked_top1_agreement": round(float(
                    sum(1 for s in outcome_steps if s["agreement"]["masked_top1_matches_heuristic"]) / len(outcome_steps)
                ), 4),
                "masked_top2_agreement": round(float(
                    sum(1 for s in outcome_steps if s["agreement"]["masked_top2_contains_heuristic"]) / len(outcome_steps)
                ), 4),
            }

    # Disagreement hotspots: steps where masked_top2 does NOT contain heuristic action
    hotspots = []
    for s in valid_steps:
        if not s["agreement"]["masked_top2_contains_heuristic"]:
            hotspots.append({
                "step": s["step"],
                "episode": s.get("episode", 0),
                "heuristic_action": s["heuristic_action"],
                "heuristic_action_name": ACTION_NAMES[s["heuristic_action"]],
                "masked_top1": s["masked"]["top1"],
                "masked_top1_name": s["masked"]["top1_name"],
                "masked_top2": s["masked"]["top2"],
                "raw_top1": s["raw"]["top1"],
                "raw_top1_name": s["raw"]["top1_name"],
            })
    # Sort by episode then step, limit to 50 most recent
    hotspots.sort(key=lambda h: (h["episode"], h["step"]))

    return {
        "num_episodes": num_episodes,
        "total_steps": n,
        "raw_top1_agreement": round(float(raw_top1_agree), 4),
        "raw_top2_agreement": round(float(raw_top2_agree), 4),
        "masked_top1_agreement": round(float(masked_top1_agree), 4),
        "masked_top2_agreement": round(float(masked_top2_agree), 4),
        "raw_entropy_normalized_mean": round(float(raw_entropy), 4),
        "masked_entropy_normalized_mean": round(float(masked_entropy), 4),
        "masked_top1_differs_from_heuristic_but_valid_rate": round(float(masked_diff_valid_rate), 4),
        "raw_top1_invalid_under_mask_rate": round(float(raw_top1_invalid), 4),
        "heuristic_action_dist": {ACTION_NAMES[i]: round(float(heuristic_dist[i]), 4) for i in range(6)},
        "raw_top1_dist": {ACTION_NAMES[i]: round(float(raw_top1_dist[i]), 4) for i in range(6)},
        "masked_top1_dist": {ACTION_NAMES[i]: round(float(masked_top1_dist[i]), 4) for i in range(6)},
        "bomb_pred_rate_raw": round(float(raw_top1_dist[5]), 4),
        "bomb_pred_rate_masked": round(float(masked_top1_dist[5]), 4),
        "stop_pred_rate_raw": round(float(raw_top1_dist[0]), 4),
        "stop_pred_rate_masked": round(float(masked_top1_dist[0]), 4),
        "agreement_by_outcome": outcome_agreement,
        "disagreement_hotspots": hotspots[:50],
    }


def _print_report(aggregate, neural, output_path):
    """Print a human-readable summary report."""
    print()
    print("=" * 60)
    print("SHADOW RANKER EVALUATION REPORT")
    print("=" * 60)

    print(f"\nNeural available: {neural.available}")
    if not neural.available:
        print(f"  Reason: {getattr(neural, 'error', 'unknown')}")
        return

    print(f"Episodes: {aggregate['num_episodes']}")
    print(f"Total steps with neural: {aggregate['total_steps']}")

    print(f"\n--- Agreement ---")
    print(f"Raw top-1 agreement:     {aggregate['raw_top1_agreement']:.3f}")
    print(f"Raw top-2 agreement:     {aggregate['raw_top2_agreement']:.3f}")
    print(f"Masked top-1 agreement:  {aggregate['masked_top1_agreement']:.3f}")
    print(f"Masked top-2 agreement:  {aggregate['masked_top2_agreement']:.3f}")

    print(f"\n--- Entropy ---")
    print(f"Raw entropy (norm):      {aggregate['raw_entropy_normalized_mean']:.3f}")
    print(f"Masked entropy (norm):   {aggregate['masked_entropy_normalized_mean']:.3f}")

    print(f"\n--- Diagnostic Rates ---")
    print(f"Masked-top1 != heuristic but valid: {aggregate['masked_top1_differs_from_heuristic_but_valid_rate']:.3f}")
    print(f"Raw-top1 invalid under mask:         {aggregate['raw_top1_invalid_under_mask_rate']:.3f}")

    print(f"\n--- Action Distribution ---")
    print(f"{'Action':<12} {'Heuristic':>10} {'Raw Top1':>10} {'Masked Top1':>10}")
    for i, name in enumerate(ACTION_NAMES):
        h = aggregate["heuristic_action_dist"][name]
        r = aggregate["raw_top1_dist"][name]
        m = aggregate["masked_top1_dist"][name]
        print(f"{name:<12} {h:>10.3f} {r:>10.3f} {m:>10.3f}")

    print(f"\n--- Bomb & Stop Rates ---")
    print(f"Bomb pred (raw):    {aggregate['bomb_pred_rate_raw']:.3f}")
    print(f"Bomb pred (masked): {aggregate['bomb_pred_rate_masked']:.3f}")
    print(f"Stop pred (raw):    {aggregate['stop_pred_rate_raw']:.3f}")
    print(f"Stop pred (masked): {aggregate['stop_pred_rate_masked']:.3f}")

    if aggregate["agreement_by_outcome"]:
        print(f"\n--- Agreement by Outcome ---")
        for outcome, metrics in aggregate["agreement_by_outcome"].items():
            print(f"  {outcome}: raw_top1={metrics['raw_top1_agreement']:.3f} "
                  f"masked_top1={metrics['masked_top1_agreement']:.3f} "
                  f"masked_top2={metrics['masked_top2_agreement']:.3f} "
                  f"(n={metrics['episodes']} ep, {metrics['steps']} steps)")

    n_hotspots = len(aggregate.get("disagreement_hotspots", []))
    print(f"\n--- Disagreement Hotspots ---")
    print(f"Steps where masked-top2 excludes heuristic: {n_hotspots}")
    if n_hotspots > 0:
        print(f"  (logged in output JSON)")

    print(f"\nOutput: {output_path}")
    print("=" * 60)


# ==============================================================================
# CLI
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Shadow evaluation for neural action ranker."
    )
    parser.add_argument(
        "--agent_path", default="agent/hybrid_agent",
        help="Path to production heuristic agent"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to neural ranker checkpoint (.pt)"
    )
    parser.add_argument(
        "--opponents", nargs=3,
        default=["TacticalRuleAgent", "TacticalRuleAgent", "TacticalRuleAgent"],
        help="3 opponent specs (baseline name or agent path)"
    )
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: logs/shadow_eval_<timestamp>.json)"
    )
    parser.add_argument(
        "--summary_only", action="store_true",
        help="Omit per-step logs from output (smaller file)"
    )
    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"logs/shadow_eval_{timestamp}.json"

    run_shadow_eval(args)


if __name__ == "__main__":
    main()
