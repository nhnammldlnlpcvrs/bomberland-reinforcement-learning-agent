from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from agent.rl_agent_pure.constants import NUM_ACTIONS, PLACE_BOMB, PPO_CONFIG
from agent.rl_agent_pure.model import BomberFeaturesExtractor
from ml.envs.bomber_gym_env import BomberGymEnv


def _policy_logits(policy, observations):
    features = policy.extract_features(observations)
    if isinstance(features, tuple):
        features = features[0]
    latent_pi, _latent_vf = policy.mlp_extractor(features)
    return policy.action_net(latent_pi)


def _make_model(args):
    env = Monitor(BomberGymEnv(agent_id=0, opponent_pool=["random", "simple"], max_steps=200, seed=args.seed))
    policy_kwargs = {
        "features_extractor_class": BomberFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    if args.base_policy and Path(args.base_policy).exists():
        return PPO.load(args.base_policy, env=env, device=args.device, custom_objects={"policy_kwargs": policy_kwargs})
    return PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=args.device,
        verbose=0,
        **PPO_CONFIG,
    )


def _metrics(logits, actions, post_bomb_escape=None):
    pred = logits.argmax(dim=1).detach().cpu().numpy()
    true = actions.detach().cpu().numpy()
    confusion = np.zeros((NUM_ACTIONS, NUM_ACTIONS), dtype=np.int64)
    for t, p in zip(true, pred):
        confusion[int(t), int(p)] += 1
    per_action = {}
    for action in range(NUM_ACTIONS):
        denom = max(1, int((true == action).sum()))
        per_action[str(action)] = float((pred[true == action] == action).sum() / denom)
    pred_bomb = pred == PLACE_BOMB
    true_bomb = true == PLACE_BOMB
    metrics = {
        "accuracy": float((pred == true).mean()) if len(true) else 0.0,
        "per_action_accuracy": per_action,
        "place_bomb_prediction_frequency": float((pred == PLACE_BOMB).mean()) if len(pred) else 0.0,
        "place_bomb_precision": float((pred_bomb & true_bomb).sum() / max(1, pred_bomb.sum())),
        "place_bomb_recall": float((pred_bomb & true_bomb).sum() / max(1, true_bomb.sum())),
        "predicted_action_distribution": {str(i): int(v) for i, v in enumerate(np.bincount(pred, minlength=NUM_ACTIONS))},
        "confusion_matrix": confusion.tolist(),
    }
    if post_bomb_escape is not None:
        mask = post_bomb_escape.detach().cpu().numpy().astype(bool)
        metrics["post_bomb_escape_accuracy"] = float((pred[mask] == true[mask]).mean()) if mask.any() else 0.0
    return metrics


def _dataset_flags(data, actions):
    is_bomb = data["is_bomb_action"].astype(bool) if "is_bomb_action" in data else actions == PLACE_BOMB
    if "is_post_bomb_escape" in data:
        post_escape = data["is_post_bomb_escape"].astype(bool)
    elif "post_bomb_escape" in data:
        post_escape = data["post_bomb_escape"].astype(bool)
    else:
        post_escape = np.zeros(len(actions), dtype=bool)
    return is_bomb, post_escape


def _mode_mask(mode, is_bomb, post_escape):
    if mode == "all":
        return np.ones(len(is_bomb), dtype=bool)
    if mode == "bomb_only":
        return is_bomb
    if mode == "post_bomb_escape":
        return post_escape
    if mode == "mixed_bomb_escape":
        return is_bomb | post_escape | ~(is_bomb | post_escape)
    raise ValueError(f"Unsupported BC mode: {mode}")


def train(args):
    data = np.load(args.dataset)
    obs = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    weights = data["sample_weight"].astype(np.float32) if "sample_weight" in data else np.ones(len(actions), dtype=np.float32)
    is_bomb, post_escape = _dataset_flags(data, actions)
    selected = _mode_mask(args.mode, is_bomb, post_escape)
    if not np.any(selected):
        raise ValueError(f"BC mode {args.mode!r} selected zero samples from {args.dataset}")
    obs = obs[selected]
    actions = actions[selected]
    weights = weights[selected]
    is_bomb = is_bomb[selected]
    post_escape = post_escape[selected]
    weights = weights.copy()
    if args.mode == "mixed_bomb_escape":
        weights[is_bomb] *= args.mixed_bomb_weight
        weights[post_escape] *= args.mixed_escape_weight
    elif args.mode == "post_bomb_escape":
        weights[is_bomb] *= args.escape_bomb_weight
        move_mask = np.isin(actions, [1, 2, 3, 4])
        weights[move_mask] *= args.escape_move_weight
    else:
        weights[is_bomb] *= args.extra_bomb_weight
        weights[post_escape] *= args.extra_escape_weight

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(actions))
    split = int(len(indices) * (1.0 - args.val_fraction))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_ds = TensorDataset(
        torch.from_numpy(obs[train_idx]),
        torch.from_numpy(actions[train_idx]),
        torch.from_numpy(weights[train_idx]),
    )
    sampler = WeightedRandomSampler(weights[train_idx], num_samples=len(train_idx), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)

    val_obs = torch.from_numpy(obs[val_idx])
    val_actions = torch.from_numpy(actions[val_idx])
    val_post_escape = torch.from_numpy(post_escape[val_idx].astype(np.bool_))

    model = _make_model(args)
    policy = model.policy
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    device = policy.device

    class_counts = np.bincount(actions[train_idx], minlength=NUM_ACTIONS).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights[PLACE_BOMB] *= args.class_bomb_weight
    if args.mode == "post_bomb_escape" and args.class_bomb_weight <= 0:
        class_weights[PLACE_BOMB] = 0.0
    class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)

    history = []
    for epoch in range(args.epochs):
        losses = []
        correct = 0
        total = 0
        for batch_obs, batch_actions, batch_weights in train_loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            batch_weights = batch_weights.to(device)
            logits = _policy_logits(policy, batch_obs)
            ce = F.cross_entropy(logits, batch_actions, weight=class_weights, reduction="none")
            loss = (ce * batch_weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
            correct += int((logits.argmax(dim=1) == batch_actions).sum().item())
            total += int(batch_actions.numel())

        policy.eval()
        with torch.no_grad():
            val_logits = _policy_logits(policy, val_obs.to(device))
            val_loss = F.cross_entropy(val_logits, val_actions.to(device), weight=class_weights).item()
            metric = _metrics(val_logits, val_actions.to(device), val_post_escape.to(device))
        policy.train()
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "train_accuracy": correct / max(1, total),
            "val_loss": float(val_loss),
            **metric,
        }
        history.append(row)
        print(json.dumps(row))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    summary = {
        "dataset": args.dataset,
        "mode": args.mode,
        "samples": int(len(actions)),
        "action_distribution": {str(i): int(v) for i, v in enumerate(np.bincount(actions, minlength=NUM_ACTIONS))},
        "bomb_samples": int(is_bomb.sum()),
        "post_bomb_escape_samples": int(post_escape.sum()),
        "bomb_class_weight": float(args.class_bomb_weight),
        "escape_bomb_weight": float(args.escape_bomb_weight),
        "escape_move_weight": float(args.escape_move_weight),
        "output": args.output,
        "final": history[-1] if history else {},
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ml/datasets/rl_bc_stage1.npz")
    parser.add_argument("--output", default="agent/rl_agent_pure/bc_policy.zip")
    parser.add_argument("--base_policy", default="agent/rl_agent_pure/policy.zip")
    parser.add_argument("--mode", choices=["all", "bomb_only", "post_bomb_escape", "mixed_bomb_escape"], default="all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--extra_bomb_weight", type=float, default=4.0)
    parser.add_argument("--extra_escape_weight", type=float, default=2.0)
    parser.add_argument("--mixed_bomb_weight", type=float, default=1.0)
    parser.add_argument("--mixed_escape_weight", type=float, default=6.0)
    parser.add_argument("--escape_bomb_weight", type=float, default=0.0)
    parser.add_argument("--escape_move_weight", type=float, default=4.0)
    parser.add_argument("--class_bomb_weight", type=float, default=3.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
