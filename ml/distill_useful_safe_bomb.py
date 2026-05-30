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


def _metrics(policy, obs, device):
    policy.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(obs), 1024):
            batch = torch.as_tensor(obs[start:start + 1024], dtype=torch.float32, device=device)
            logits = _policy_logits(policy, batch)
            probs.append(torch.softmax(logits, dim=1)[:, PLACE_BOMB].detach().cpu().numpy())
    probs = np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)
    return {
        "bomb_prob_mean": float(np.mean(probs)) if len(probs) else 0.0,
        "bomb_prob_median": float(np.median(probs)) if len(probs) else 0.0,
        "bomb_prob_ge_0_01": float((probs >= 0.01).mean()) if len(probs) else 0.0,
        "bomb_prob_ge_0_05": float((probs >= 0.05).mean()) if len(probs) else 0.0,
        "bomb_prob_ge_0_10": float((probs >= 0.10).mean()) if len(probs) else 0.0,
    }


def train(args):
    data = np.load(args.dataset)
    obs = data["observations"].astype(np.float32)
    if args.max_samples and len(obs) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(obs), size=args.max_samples, replace=False)
        obs = obs[idx]

    model = _load_model(args.base_policy, args.device, args.seed)
    policy = model.policy
    device = policy.device
    before = _metrics(policy, obs, device)

    if args.freeze_features:
        for param in policy.features_extractor.parameters():
            param.requires_grad = False
        for param in policy.mlp_extractor.parameters():
            param.requires_grad = False

    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    loader = DataLoader(TensorDataset(torch.from_numpy(obs)), batch_size=args.batch_size, shuffle=True)
    history = []
    for epoch in range(args.epochs):
        losses = []
        means = []
        for (batch_obs,) in loader:
            batch_obs = batch_obs.to(device)
            logits = _policy_logits(policy, batch_obs)
            bomb_logit = logits[:, PLACE_BOMB]
            if args.loss == "softmax_ce":
                loss = F.cross_entropy(logits, torch.full((len(batch_obs),), PLACE_BOMB, dtype=torch.long, device=batch_obs.device))
            elif args.loss == "bce":
                target = torch.full_like(bomb_logit, args.target_prob)
                loss = F.binary_cross_entropy_with_logits(bomb_logit, target)
            else:
                target = torch.full_like(bomb_logit, args.target_logit)
                loss = F.mse_loss(bomb_logit, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
            means.append(float(torch.sigmoid(bomb_logit.detach()).mean().item()))
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "train_sigmoid_bomb_logit_mean": float(np.mean(means)),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    after = _metrics(policy, obs, device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    summary = {
        "dataset": args.dataset,
        "base_policy": args.base_policy,
        "output": args.output,
        "samples": int(len(obs)),
        "freeze_features": bool(args.freeze_features),
        "loss": args.loss,
        "target_prob": args.target_prob,
        "target_logit": args.target_logit,
        "before": before,
        "after": after,
        "history": history,
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Nudge PLACE_BOMB logit only on useful-safe bomb contexts.")
    parser.add_argument("--dataset", default="ml/datasets/rl_bc_useful_safe_bomb_contexts.npz")
    parser.add_argument("--base_policy", default="ml/checkpoints/rl_agent_pure/ppo_escape_conservative_best.zip")
    parser.add_argument("--output", default="ml/checkpoints/rl_agent_pure/ppo_conservative_useful_bomb_distilled.zip")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--loss", choices=["softmax_ce", "bce", "mse"], default="softmax_ce")
    parser.add_argument("--target_prob", type=float, default=0.08)
    parser.add_argument("--target_logit", type=float, default=-2.4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--freeze_features", action="store_true", default=True)
    parser.add_argument("--unfreeze_all", action="store_false", dest="freeze_features")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
