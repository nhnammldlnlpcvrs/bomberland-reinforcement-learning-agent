from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from agent.rl_agent_pure.model import BomberFeaturesExtractor
from ml.envs.bomber_gym_env import BomberGymEnv

PLACE_BOMB = 5
NUM_ACTIONS = 6


def _policy_logits(policy, observations):
    features = policy.extract_features(observations)
    if isinstance(features, tuple):
        features = features[0]
    latent_pi, _latent_vf = policy.mlp_extractor(features)
    return policy.action_net(latent_pi)


def _load_model(path, device, seed):
    env = Monitor(BomberGymEnv(agent_id=0, opponent_pool=["random", "simple"], max_steps=200, seed=seed))
    policy_kwargs = {
        "features_extractor_class": BomberFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    return PPO.load(path, env=env, device=device, custom_objects={"policy_kwargs": policy_kwargs})


def _eval_metrics(policy, bomb_obs, escape_obs, escape_action, device):
    policy.eval()
    out = {}
    with torch.no_grad():
        bomb_probs = []
        for start in range(0, len(bomb_obs), 1024):
            obs = torch.as_tensor(bomb_obs[start:start + 1024], dtype=torch.float32, device=device)
            logits = _policy_logits(policy, obs)
            bomb_probs.append(torch.softmax(logits, dim=1)[:, PLACE_BOMB].cpu().numpy())
        bomb_probs = np.concatenate(bomb_probs) if bomb_probs else np.zeros(0, dtype=np.float32)
        preds = []
        for start in range(0, len(escape_obs), 1024):
            obs = torch.as_tensor(escape_obs[start:start + 1024], dtype=torch.float32, device=device)
            logits = _policy_logits(policy, obs)
            preds.append(logits.argmax(dim=1).cpu().numpy())
        preds = np.concatenate(preds) if preds else np.zeros(0, dtype=np.int64)
    out["bomb_prob_mean"] = float(np.mean(bomb_probs)) if len(bomb_probs) else 0.0
    out["bomb_prob_ge_0_05"] = float((bomb_probs >= 0.05).mean()) if len(bomb_probs) else 0.0
    out["escape_accuracy"] = float((preds == escape_action).mean()) if len(preds) else 0.0
    out["predicted_action_distribution"] = {str(i): int(v) for i, v in enumerate(np.bincount(preds, minlength=NUM_ACTIONS))}
    per_action = {}
    for action in range(NUM_ACTIONS):
        mask = escape_action == action
        per_action[str(action)] = float((preds[mask] == action).mean()) if mask.any() else 0.0
    out["escape_per_action_accuracy"] = per_action
    return out


def train(args):
    data = np.load(args.dataset)
    bomb_obs = data["bomb_obs"].astype(np.float32)
    escape_obs = data["escape_obs"].astype(np.float32)
    escape_action = data["escape_action"].astype(np.int64)
    model = _load_model(args.base_policy, args.device, args.seed)
    policy = model.policy
    device = policy.device
    before = _eval_metrics(policy, bomb_obs, escape_obs, escape_action, device)

    if args.freeze_features:
        for param in policy.features_extractor.parameters():
            param.requires_grad = False
        for param in policy.mlp_extractor.parameters():
            param.requires_grad = False
        for param in policy.value_net.parameters():
            param.requires_grad = False
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    bomb_loader = DataLoader(TensorDataset(torch.from_numpy(bomb_obs)), batch_size=args.batch_size, shuffle=True, drop_last=False)
    escape_loader = DataLoader(
        TensorDataset(torch.from_numpy(escape_obs), torch.from_numpy(escape_action)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    history = []
    for epoch in range(args.epochs):
        bomb_iter = iter(bomb_loader)
        escape_iter = iter(escape_loader)
        losses = []
        bomb_losses = []
        escape_losses = []
        steps = max(len(bomb_loader), len(escape_loader))
        for _ in range(steps):
            try:
                (batch_bomb,) = next(bomb_iter)
            except StopIteration:
                bomb_iter = iter(bomb_loader)
                (batch_bomb,) = next(bomb_iter)
            try:
                batch_escape, batch_action = next(escape_iter)
            except StopIteration:
                escape_iter = iter(escape_loader)
                batch_escape, batch_action = next(escape_iter)
            batch_bomb = batch_bomb.to(device)
            batch_escape = batch_escape.to(device)
            batch_action = batch_action.to(device)
            bomb_logits = _policy_logits(policy, batch_bomb)
            escape_logits = _policy_logits(policy, batch_escape)
            bomb_target = torch.full((len(batch_bomb),), PLACE_BOMB, dtype=torch.long, device=device)
            bomb_loss = F.cross_entropy(bomb_logits, bomb_target)
            escape_loss = F.cross_entropy(escape_logits, batch_action)
            loss = args.bomb_loss_weight * bomb_loss + args.escape_loss_weight * escape_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
            bomb_losses.append(float(bomb_loss.item()))
            escape_losses.append(float(escape_loss.item()))
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "bomb_loss": float(np.mean(bomb_losses)),
            "escape_loss": float(np.mean(escape_losses)),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    after = _eval_metrics(policy, bomb_obs, escape_obs, escape_action, device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    summary = {
        "dataset": args.dataset,
        "base_policy": args.base_policy,
        "output": args.output,
        "bomb_samples": int(len(bomb_obs)),
        "escape_samples": int(len(escape_obs)),
        "bomb_loss_weight": args.bomb_loss_weight,
        "escape_loss_weight": args.escape_loss_weight,
        "freeze_features": bool(args.freeze_features),
        "before": before,
        "after": after,
        "history": history,
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Distill useful bomb context plus escape movement sequence.")
    parser.add_argument("--dataset", default="ml/datasets/rl_bc_useful_bomb_sequences.npz")
    parser.add_argument("--base_policy", default="ml/checkpoints/rl_agent_pure/ppo_useful_safe_bomb_best.zip")
    parser.add_argument("--output", default="ml/checkpoints/rl_agent_pure/ppo_useful_bomb_sequence_distilled.zip")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--bomb_loss_weight", type=float, default=0.3)
    parser.add_argument("--escape_loss_weight", type=float, default=6.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--freeze_features", action="store_true", default=True)
    parser.add_argument("--unfreeze_all", action="store_false", dest="freeze_features")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
