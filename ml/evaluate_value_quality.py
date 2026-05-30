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
from agent.rl_agent_pure.utils import boxes_in_blast, normalize_obs
from ml.collect_targeted_aux_rollouts import _counterfactual_labels
from ml.envs.bomber_gym_env import BomberGymEnv


def _load_ppo(path: str, device: str) -> PPO:
    policy_kwargs = {
        "features_extractor_class": BomberFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    return PPO.load(path, device=device, custom_objects={"policy_kwargs": policy_kwargs})


def _load_aux(path: str, device: torch.device) -> BomberAuxModel | None:
    if not path or not Path(path).exists():
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = BomberAuxModel(features_dim=int(checkpoint.get("features_dim", 256))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _load_thresholds(path: str | None) -> dict[str, float]:
    defaults = {
        "death": 0.7,
        "escape_available": 0.3,
        "bomb_escape_available": 0.3,
        "trapped_if_bomb": 0.5,
        "future_blast": 0.7,
    }
    if path and Path(path).exists():
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        defaults.update({key: float(loaded.get(key, value)) for key, value in defaults.items()})
    return defaults


def _discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        out[idx] = running
    return out


def _aux_probs(aux: BomberAuxModel | None, obs: np.ndarray, device: torch.device) -> dict[str, float]:
    if aux is None:
        return {
            "death_prob": 0.0,
            "bomb_escape_prob": 1.0,
            "trapped_prob": 0.0,
            "future_blast_prob": 0.0,
        }
    with torch.no_grad():
        out = aux(torch.as_tensor(obs[None], dtype=torch.float32, device=device))
    return {
        "death_prob": float(torch.sigmoid(out["death_logit"])[0].cpu()),
        "bomb_escape_prob": float(torch.sigmoid(out["bomb_escape_available_logit"])[0].cpu()),
        "trapped_prob": float(torch.sigmoid(out["trapped_if_bomb_logit"])[0].cpu()),
        "future_blast_prob": float(torch.sigmoid(out["future_blast_logit"])[0].cpu()),
    }


def _collect(args, actor: PPO, aux: BomberAuxModel | None, thresholds: dict[str, float], device: torch.device) -> dict:
    observations = []
    rewards = []
    returns = []
    done_flags = []
    opponent_names = []
    actions = []
    labels_by_key: dict[str, list] = {
        "bomb_related": [],
        "future_death": [],
        "high_death_risk": [],
        "safe_useful_bomb_avoided": [],
        "terminal_near": [],
        "random_opponent": [],
        "simple_opponent": [],
    }
    episode_stats = []
    for opponent in args.opponents:
        for episode in range(args.episodes):
            env = BomberGymEnv(
                agent_id=args.agent_id,
                opponent_pool=[opponent],
                max_steps=args.max_steps,
                seed=args.seed + episode,
                training_bomb_gate=False,
            )
            obs, _info = env.reset(seed=args.seed + episode)
            ep_obs = []
            ep_rewards = []
            ep_actions = []
            ep_labels = {key: [] for key in labels_by_key}
            alive_seq = []
            done = False
            truncated = False
            while not (done or truncated):
                action, _state = actor.predict(obs, deterministic=True)
                action = int(np.asarray(action).reshape(-1)[0])
                obs_dict = env.last_obs
                board, players, _bombs, _step = normalize_obs(obs_dict)
                row, col = int(players[args.agent_id, 0]), int(players[args.agent_id, 1])
                counter = _counterfactual_labels(obs_dict, args.agent_id)
                probs = _aux_probs(aux, obs, device)
                boxes = int(boxes_in_blast(board, players, row, col, args.agent_id))
                ep_obs.append(obs.astype(np.float32))
                ep_actions.append(action)
                ep_labels["bomb_related"].append(bool(action == PLACE_BOMB or counter["would_destroy_boxes_if_bomb"] > 0))
                ep_labels["high_death_risk"].append(bool(probs["death_prob"] >= thresholds["death"]))
                ep_labels["safe_useful_bomb_avoided"].append(
                    bool(
                        action != PLACE_BOMB
                        and boxes > 0
                        and probs["death_prob"] < args.safe_death_threshold
                        and probs["bomb_escape_prob"] >= thresholds["bomb_escape_available"]
                        and probs["trapped_prob"] < thresholds["trapped_if_bomb"]
                    )
                )
                ep_labels["random_opponent"].append(opponent == "random")
                ep_labels["simple_opponent"].append(opponent == "simple")
                next_obs, reward, done, truncated, info = env.step(action)
                ep_rewards.append(float(reward))
                alive_seq.append(bool(info.get("alive", False)))
                obs = next_obs
            ep_returns = _discounted_returns(ep_rewards, args.gamma)
            death_within = np.zeros(len(ep_rewards), dtype=bool)
            terminal_near = np.zeros(len(ep_rewards), dtype=bool)
            for idx in range(len(ep_rewards)):
                horizon = alive_seq[idx:min(len(alive_seq), idx + args.death_horizon + 1)]
                death_within[idx] = bool(horizon and any(not item for item in horizon))
                terminal_near[idx] = idx >= max(0, len(ep_rewards) - args.death_horizon)
            ep_labels["future_death"] = death_within.tolist()
            ep_labels["terminal_near"] = terminal_near.tolist()
            observations.extend(ep_obs)
            rewards.extend(ep_rewards)
            returns.extend(ep_returns.tolist())
            done_flags.extend([False] * max(0, len(ep_rewards) - 1) + [True] if ep_rewards else [])
            opponent_names.extend([opponent] * len(ep_rewards))
            actions.extend(ep_actions)
            for key in labels_by_key:
                labels_by_key[key].extend(ep_labels[key])
            episode_stats.append({"opponent": opponent, "steps": len(ep_rewards), "return": float(ep_returns[0]) if len(ep_returns) else 0.0})
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "returns": np.asarray(returns, dtype=np.float32),
        "dones": np.asarray(done_flags, dtype=np.int8),
        "actions": np.asarray(actions, dtype=np.int64),
        "opponents": np.asarray(opponent_names),
        "labels": {key: np.asarray(value, dtype=bool) for key, value in labels_by_key.items()},
        "episode_stats": episode_stats,
    }


def _predict_values(model: PPO, obs: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    values = []
    model.policy.eval()
    with torch.no_grad():
        for start in range(0, len(obs), batch_size):
            batch = torch.as_tensor(obs[start:start + batch_size], dtype=torch.float32, device=device)
            values.append(model.policy.predict_values(batch).flatten().detach().cpu().numpy())
    return np.concatenate(values).astype(np.float32) if values else np.zeros((0,), dtype=np.float32)


def _explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    if len(target) == 0:
        return float("nan")
    variance = float(np.var(target))
    if variance < 1e-8:
        return float("nan")
    return float(1.0 - np.var(target - pred) / variance)


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    if len(target) == 0:
        return {"count": 0, "mae": None, "mse": None, "explained_variance": None}
    err = pred - target
    return {
        "count": int(len(target)),
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err * err)),
        "bias": float(np.mean(err)),
        "explained_variance": _explained_variance(pred, target),
    }


def _quantile_metrics(pred: np.ndarray, target: np.ndarray, bins: int = 5) -> list[dict]:
    if len(target) == 0:
        return []
    quantiles = np.quantile(target, np.linspace(0.0, 1.0, bins + 1))
    out = []
    for idx in range(bins):
        lo, hi = quantiles[idx], quantiles[idx + 1]
        if idx == bins - 1:
            mask = (target >= lo) & (target <= hi)
        else:
            mask = (target >= lo) & (target < hi)
        row = _metrics(pred[mask], target[mask])
        row.update({"bin": idx, "return_min": float(lo), "return_max": float(hi)})
        out.append(row)
    return out


def _logit_drift(a: PPO, b: PPO, obs: np.ndarray, device: torch.device, batch_size: int) -> dict:
    max_delta = 0.0
    sum_delta = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, min(len(obs), 4096), batch_size):
            batch = torch.as_tensor(obs[start:start + batch_size], dtype=torch.float32, device=device)
            logits_a = a.policy.get_distribution(batch).distribution.logits.detach().cpu()
            logits_b = b.policy.get_distribution(batch).distribution.logits.detach().cpu()
            delta = torch.abs(logits_a - logits_b)
            max_delta = max(max_delta, float(delta.max()))
            sum_delta += float(delta.sum())
            count += int(delta.numel())
    return {"max_logit_delta": max_delta, "mean_logit_delta": sum_delta / max(1, count)}


def evaluate_value_quality(args):
    device = torch.device(args.device)
    thresholds = _load_thresholds(args.thresholds)
    actor = _load_ppo(args.actor_policy or args.baseline_policy, args.device)
    baseline = _load_ppo(args.baseline_policy, args.device)
    aux_ckpt = _load_ppo(args.aux_policy, args.device)
    aux = _load_aux(args.aux_model, device)
    data = _collect(args, actor, aux, thresholds, device)

    output_dataset = Path(args.dataset_output)
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dataset,
        observations=data["observations"],
        rewards=data["rewards"],
        returns=data["returns"],
        dones=data["dones"],
        actions=data["actions"],
        opponents=data["opponents"],
        **{f"label_{key}": value.astype(np.int8) for key, value in data["labels"].items()},
    )

    predictions = {
        "baseline": _predict_values(baseline, data["observations"], device, args.batch_size),
        "aux_critic": _predict_values(aux_ckpt, data["observations"], device, args.batch_size),
    }
    subset_masks = {"all": np.ones(len(data["returns"]), dtype=bool), **data["labels"]}
    table = []
    for checkpoint, pred in predictions.items():
        for subset, mask in subset_masks.items():
            row = {"checkpoint": checkpoint, "subset": subset}
            row.update(_metrics(pred[mask], data["returns"][mask]))
            table.append(row)
    report = {
        "dataset_stats": {
            "states": int(len(data["returns"])),
            "episodes": int(len(data["episode_stats"])),
            "opponents": args.opponents,
            "return_mean": float(data["returns"].mean()) if len(data["returns"]) else 0.0,
            "return_std": float(data["returns"].std()) if len(data["returns"]) else 0.0,
            "subset_counts": {key: int(mask.sum()) for key, mask in subset_masks.items()},
        },
        "metrics": table,
        "calibration_by_return_quantile": {
            checkpoint: _quantile_metrics(pred, data["returns"], bins=args.quantile_bins)
            for checkpoint, pred in predictions.items()
        },
        "actor_drift": {
            "baseline_vs_aux_critic": _logit_drift(baseline, aux_ckpt, data["observations"], device, args.batch_size)
        },
        "episode_stats": data["episode_stats"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO critic/value quality on rollout returns and important Bomberland subsets.")
    parser.add_argument("--baseline_policy", default="ml/checkpoints/rl_agent_pure/ppo_bomb_selector_best.zip")
    parser.add_argument("--aux_policy", default="ml/checkpoints/rl_agent_pure/ppo_aux_integrated_best.zip")
    parser.add_argument("--actor_policy", default=None, help="Policy used to collect rollouts; defaults to baseline.")
    parser.add_argument("--aux_model", default="ml/checkpoints/rl_agent_pure/aux_curriculum_model_v3.pt")
    parser.add_argument("--thresholds", default="ml/checkpoints/rl_agent_pure/aux_thresholds_v3.json")
    parser.add_argument("--output", default="logs/value_quality_report.json")
    parser.add_argument("--dataset_output", default="ml/datasets/value_quality_rollouts.npz")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--opponents", nargs="*", default=["random", "simple"])
    parser.add_argument("--seed", type=int, default=8100)
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--death_horizon", type=int, default=7)
    parser.add_argument("--safe_death_threshold", type=float, default=0.3)
    parser.add_argument("--quantile_bins", type=int, default=5)
    args = parser.parse_args()
    evaluate_value_quality(args)


if __name__ == "__main__":
    main()
