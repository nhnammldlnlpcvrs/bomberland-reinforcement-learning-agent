from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.aux_model import BomberAuxModel
from agent.rl_agent_pure.constants import PLACE_BOMB
from agent.rl_agent_pure.model import BomberFeaturesExtractor
from agent.rl_agent_pure.utils import boxes_in_blast, compute_danger_map, normalize_obs
from ml.collect_targeted_aux_rollouts import _counterfactual_labels
from ml.envs.bomber_gym_env import BomberGymEnv


SCENARIO_TO_ID = {
    "safe_useful_bomb_avoided": 0,
    "unsafe_bomb_context": 1,
    "escape_risk_state": 2,
    "normal_safe_state": 3,
}
ID_TO_SCENARIO = {value: key for key, value in SCENARIO_TO_ID.items()}


def _load_aux(path: str, device: torch.device) -> BomberAuxModel:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = BomberAuxModel(features_dim=int(checkpoint.get("features_dim", 256))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _load_thresholds(path: str | None, args) -> dict[str, float]:
    thresholds = {
        "death": args.death_threshold,
        "escape_available": args.escape_available_threshold,
        "bomb_escape_available": args.bomb_escape_available_threshold,
        "trapped_if_bomb": args.trapped_threshold,
        "future_blast": args.future_blast_threshold,
    }
    if path and Path(path).exists():
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in thresholds:
            thresholds[key] = float(loaded.get(key, thresholds[key]))
    return thresholds


def _policy_action(policy: PPO, obs: np.ndarray) -> int:
    action, _state = policy.predict(obs, deterministic=True)
    return int(np.asarray(action).reshape(-1)[0])


def _aux_probs(model: BomberAuxModel, obs: np.ndarray, device: torch.device) -> dict[str, float]:
    with torch.no_grad():
        tensor = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
        out = model(tensor)
    return {
        "death_prob": float(torch.sigmoid(out["death_logit"])[0].cpu()),
        "escape_available_prob": float(torch.sigmoid(out["escape_available_logit"])[0].cpu()),
        "bomb_escape_available_prob": float(torch.sigmoid(out["bomb_escape_available_logit"])[0].cpu()),
        "trapped_if_bomb_prob": float(torch.sigmoid(out["trapped_if_bomb_logit"])[0].cpu()),
        "future_blast_prob": float(torch.sigmoid(out["future_blast_logit"])[0].cpu()),
        "safe_tiles_after_bomb_pred": float(out["safe_tiles_after_bomb"][0].cpu()),
        "box_value_pred": float(out["box_value"][0].cpu()),
    }


def _classify(action: int, labels: dict, probs: dict, thresholds: dict, args) -> str | None:
    would_destroy = float(labels["would_destroy_boxes_if_bomb"])
    death_safe = probs["death_prob"] < args.safe_death_threshold
    escape_available = probs["escape_available_prob"] >= thresholds["escape_available"]
    bomb_escape_available = probs["bomb_escape_available_prob"] >= thresholds["bomb_escape_available"]
    trapped = probs["trapped_if_bomb_prob"] >= thresholds["trapped_if_bomb"]
    future_blast = probs["future_blast_prob"] >= thresholds["future_blast"]
    high_death = probs["death_prob"] >= thresholds["death"]

    if (
        action != PLACE_BOMB
        and would_destroy > 0
        and death_safe
        and bomb_escape_available
        and not trapped
    ):
        return "safe_useful_bomb_avoided"
    if would_destroy > 0 and (trapped or not bomb_escape_available):
        return "unsafe_bomb_context"
    if future_blast or high_death:
        return "escape_risk_state"
    if escape_available and death_safe and would_destroy <= 0:
        return "normal_safe_state"
    return None


def build(args):
    device = torch.device(args.device)
    thresholds = _load_thresholds(args.thresholds, args)
    aux = _load_aux(args.aux_model, device)
    policy_kwargs = {
        "features_extractor_class": BomberFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    policy = PPO.load(args.policy, device=args.device, custom_objects={"policy_kwargs": policy_kwargs})
    env = BomberGymEnv(agent_id=args.agent_id, opponent_pool=args.opponents, max_steps=args.max_steps, seed=args.seed)

    rows: list[dict] = []
    scanned = 0
    for episode in range(args.episodes):
        obs, _info = env.reset(seed=args.seed + episode)
        done = False
        truncated = False
        while not (done or truncated):
            action = _policy_action(policy, obs)
            labels = _counterfactual_labels(env.last_obs, args.agent_id)
            probs = _aux_probs(aux, obs, device)
            scenario = _classify(action, labels, probs, thresholds, args)
            scanned += 1
            if scenario is not None:
                board, players, bombs, _step = normalize_obs(env.last_obs)
                row, col = int(players[args.agent_id, 0]), int(players[args.agent_id, 1])
                danger = compute_danger_map(board, players, bombs)
                rows.append(
                    {
                        "obs": obs.astype(np.float32),
                        "action": int(action),
                        "scenario_type": SCENARIO_TO_ID[scenario],
                        "step": int(env.env.current_step),
                        "row": row,
                        "col": col,
                        "current_danger": int(danger[row, col]),
                        "boxes_in_blast": int(boxes_in_blast(board, players, row, col, args.agent_id)),
                        **labels,
                        **probs,
                    }
                )
            obs, _reward, done, truncated, _info = env.step(action)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        arrays = {
            "observations": np.asarray([row["obs"] for row in rows], dtype=np.float32),
            "actions": np.asarray([row["action"] for row in rows], dtype=np.int64),
            "scenario_types": np.asarray([row["scenario_type"] for row in rows], dtype=np.int64),
            "steps": np.asarray([row["step"] for row in rows], dtype=np.int64),
            "positions": np.asarray([[row["row"], row["col"]] for row in rows], dtype=np.int16),
            "current_danger": np.asarray([row["current_danger"] for row in rows], dtype=np.float32),
            "boxes_in_blast": np.asarray([row["boxes_in_blast"] for row in rows], dtype=np.float32),
        }
        for key in (
            "has_escape_path_now",
            "has_escape_after_bomb",
            "would_destroy_boxes_if_bomb",
            "in_future_blast",
            "trapped_if_bomb",
            "safe_tiles_after_bomb_count",
            "blast_corridor_distance",
            "death_prob",
            "escape_available_prob",
            "bomb_escape_available_prob",
            "trapped_if_bomb_prob",
            "future_blast_prob",
            "safe_tiles_after_bomb_pred",
            "box_value_pred",
        ):
            arrays[key] = np.asarray([row[key] for row in rows], dtype=np.float32)
    else:
        arrays = {
            "observations": np.zeros((0, 19, 13, 13), dtype=np.float32),
            "actions": np.zeros((0,), dtype=np.int64),
            "scenario_types": np.zeros((0,), dtype=np.int64),
        }
    np.savez_compressed(output, **arrays)

    stats = {
        "scanned_states": int(scanned),
        "saved_states": int(len(rows)),
        "thresholds": thresholds,
        "scenario_counts": {
            ID_TO_SCENARIO[int(key)]: int((arrays["scenario_types"] == key).sum())
            for key in np.unique(arrays["scenario_types"])
        } if len(rows) else {},
    }
    if len(rows):
        stats.update(
            {
                "would_destroy_boxes_mean": float(arrays["would_destroy_boxes_if_bomb"].mean()),
                "bomb_escape_available_rate": float(arrays["has_escape_after_bomb"].mean()),
                "trapped_if_bomb_rate": float(arrays["trapped_if_bomb"].mean()),
                "death_prob_mean": float(arrays["death_prob"].mean()),
            }
        )
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Extract aux-guided normal-env scenarios for offline curriculum/pretraining.")
    parser.add_argument("--policy", default="ml/checkpoints/rl_agent_pure/ppo_bomb_selector_best.zip")
    parser.add_argument("--aux_model", default="ml/checkpoints/rl_agent_pure/aux_curriculum_model_v3.pt")
    parser.add_argument("--thresholds", default="ml/checkpoints/rl_agent_pure/aux_thresholds_v3.json")
    parser.add_argument("--output", default="ml/datasets/aux_guided_scenarios.npz")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7600)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponents", nargs="*", default=["random", "simple"])
    parser.add_argument("--death_threshold", type=float, default=0.7)
    parser.add_argument("--safe_death_threshold", type=float, default=0.3)
    parser.add_argument("--escape_available_threshold", type=float, default=0.3)
    parser.add_argument("--bomb_escape_available_threshold", type=float, default=0.3)
    parser.add_argument("--trapped_threshold", type=float, default=0.5)
    parser.add_argument("--future_blast_threshold", type=float, default=0.7)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
