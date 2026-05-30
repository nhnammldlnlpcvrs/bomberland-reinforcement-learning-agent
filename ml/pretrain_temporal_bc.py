from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.constants import NUM_ACTIONS, PLACE_BOMB
from agent.rl_agent_pure.model import BomberFeaturesExtractor
from ml.envs.bomber_gym_env import BomberGymEnv
from ml.envs.frame_stack_wrapper import FrameStackObservationWrapper


def _make_env(frame_stack: int, seed: int):
    env = BomberGymEnv(agent_id=0, opponent_pool=["random", "simple"], max_steps=200, seed=seed)
    if frame_stack > 1:
        env = FrameStackObservationWrapper(env, frame_stack)
    return Monitor(env)


def _load_model(path: str, frame_stack: int, device: str, seed: int) -> PPO:
    policy_kwargs = {
        "features_extractor_class": BomberFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    return PPO.load(path, env=_make_env(frame_stack, seed), device=device, custom_objects={"policy_kwargs": policy_kwargs})


def _policy_logits(policy, observations: torch.Tensor) -> torch.Tensor:
    features = policy.extract_features(observations)
    if isinstance(features, tuple):
        features = features[0]
    latent_pi, _latent_vf = policy.mlp_extractor(features)
    return policy.action_net(latent_pi)


def _set_trainable(policy, mode: str):
    for param in policy.parameters():
        param.requires_grad = False
    if mode == "action_head":
        for param in policy.action_net.parameters():
            param.requires_grad = True
    elif mode == "policy_head":
        for param in policy.mlp_extractor.policy_net.parameters():
            param.requires_grad = True
        for param in policy.action_net.parameters():
            param.requires_grad = True
    elif mode == "all_policy":
        for param in policy.features_extractor.parameters():
            param.requires_grad = True
        for param in policy.mlp_extractor.policy_net.parameters():
            param.requires_grad = True
        for param in policy.action_net.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown trainable mode: {mode}")
    return [param for param in policy.parameters() if param.requires_grad]


def _metrics(policy, obs: np.ndarray, actions: np.ndarray, is_bomb: np.ndarray, is_escape: np.ndarray, device, batch_size: int) -> dict:
    preds = []
    bomb_probs = []
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(obs), batch_size):
            batch = torch.as_tensor(obs[start:start + batch_size], dtype=torch.float32, device=device)
            logits = _policy_logits(policy, batch)
            probs = torch.softmax(logits, dim=1)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            bomb_probs.append(probs[:, PLACE_BOMB].cpu().numpy())
    pred = np.concatenate(preds) if preds else np.zeros(0, dtype=np.int64)
    bomb_prob = np.concatenate(bomb_probs) if bomb_probs else np.zeros(0, dtype=np.float32)
    out = {
        "accuracy": float((pred == actions).mean()) if len(actions) else 0.0,
        "global_predicted_bomb_frequency": float((pred == PLACE_BOMB).mean()) if len(pred) else 0.0,
        "global_bomb_prob_mean": float(bomb_prob.mean()) if len(bomb_prob) else 0.0,
        "bomb_context_prob_mean": float(bomb_prob[is_bomb].mean()) if np.any(is_bomb) else 0.0,
        "bomb_context_pred_frequency": float((pred[is_bomb] == PLACE_BOMB).mean()) if np.any(is_bomb) else 0.0,
        "escape_accuracy": float((pred[is_escape] == actions[is_escape]).mean()) if np.any(is_escape) else 0.0,
        "predicted_action_distribution": {str(i): int(v) for i, v in enumerate(np.bincount(pred, minlength=NUM_ACTIONS))},
    }
    per_action = {}
    for action in range(NUM_ACTIONS):
        mask = actions == action
        per_action[str(action)] = float((pred[mask] == action).mean()) if np.any(mask) else 0.0
    out["per_action_accuracy"] = per_action
    return out


def _train_one(args, bomb_weight: float, output: str) -> dict:
    data = np.load(args.dataset)
    obs = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    is_bomb = data["is_bomb_action"].astype(bool) if "is_bomb_action" in data else actions == PLACE_BOMB
    is_escape = data["is_post_bomb_escape"].astype(bool) if "is_post_bomb_escape" in data else np.zeros(len(actions), dtype=bool)
    if not np.any(is_bomb) or not np.any(is_escape):
        raise ValueError("Frame-stack BC dataset must contain both bomb and escape samples.")

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(actions))
    split = int(len(indices) * (1.0 - args.val_fraction))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_bomb_idx = train_idx[is_bomb[train_idx]]
    train_escape_idx = train_idx[is_escape[train_idx]]
    bomb_loader = DataLoader(TensorDataset(torch.from_numpy(obs[train_bomb_idx])), batch_size=args.batch_size, shuffle=True)
    escape_loader = DataLoader(
        TensorDataset(torch.from_numpy(obs[train_escape_idx]), torch.from_numpy(actions[train_escape_idx])),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = _load_model(args.base_policy, args.frame_stack, args.device, args.seed)
    policy = model.policy
    device = policy.device
    before = _metrics(policy, obs[val_idx], actions[val_idx], is_bomb[val_idx], is_escape[val_idx], device, args.batch_size)
    params = _set_trainable(policy, args.trainable)
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    best_state = None
    best_metric = -1e9
    for epoch in range(args.epochs):
        policy.train()
        losses = []
        bomb_iter = iter(bomb_loader)
        escape_iter = iter(escape_loader)
        steps = max(len(bomb_loader), len(escape_loader))
        for _ in range(steps):
            try:
                (batch_bomb_obs,) = next(bomb_iter)
            except StopIteration:
                bomb_iter = iter(bomb_loader)
                (batch_bomb_obs,) = next(bomb_iter)
            try:
                batch_escape_obs, batch_escape_actions = next(escape_iter)
            except StopIteration:
                escape_iter = iter(escape_loader)
                batch_escape_obs, batch_escape_actions = next(escape_iter)
            batch_bomb_obs = batch_bomb_obs.to(device)
            batch_escape_obs = batch_escape_obs.to(device)
            batch_escape_actions = batch_escape_actions.to(device)
            bomb_logits = _policy_logits(policy, batch_bomb_obs)
            escape_logits = _policy_logits(policy, batch_escape_obs)
            bomb_targets = torch.full((len(batch_bomb_obs),), PLACE_BOMB, dtype=torch.long, device=device)
            bomb_loss = F.cross_entropy(bomb_logits, bomb_targets)
            escape_loss = F.cross_entropy(escape_logits, batch_escape_actions)
            loss = bomb_weight * bomb_loss + args.escape_loss_weight * escape_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
        metric = _metrics(policy, obs[val_idx], actions[val_idx], is_bomb[val_idx], is_escape[val_idx], device, args.batch_size)
        metric["epoch"] = epoch + 1
        metric["train_loss"] = float(np.mean(losses)) if losses else 0.0
        history.append(metric)
        score = metric["escape_accuracy"] + args.bomb_metric_weight * metric["bomb_context_prob_mean"]
        if metric["global_predicted_bomb_frequency"] <= args.max_global_bomb_pred and score > best_metric:
            best_metric = score
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        print(json.dumps({"output": output, "bomb_weight": bomb_weight, **metric}), flush=True)
    if best_state is not None:
        policy.load_state_dict(best_state)
    after = _metrics(policy, obs[val_idx], actions[val_idx], is_bomb[val_idx], is_escape[val_idx], device, args.batch_size)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    summary = {
        "dataset": args.dataset,
        "base_policy": args.base_policy,
        "output": output,
        "frame_stack": args.frame_stack,
        "trainable": args.trainable,
        "bomb_loss_weight": bomb_weight,
        "escape_loss_weight": args.escape_loss_weight,
        "samples": int(len(actions)),
        "bomb_samples": int(is_bomb.sum()),
        "escape_samples": int(is_escape.sum()),
        "action_distribution": {str(i): int(v) for i, v in enumerate(np.bincount(actions, minlength=NUM_ACTIONS))},
        "before": before,
        "after": after,
        "history": history,
    }
    Path(output).with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def train(args):
    summaries = []
    weights = [float(item) for item in args.bomb_loss_weights]
    outputs = args.outputs
    if outputs and len(outputs) != len(weights):
        raise ValueError("--outputs length must match --bomb_loss_weights")
    for weight in weights:
        suffix = f"{int(round(weight * 1000)):03d}"
        output = outputs[weights.index(weight)] if outputs else f"{args.output_prefix}{suffix}.zip"
        summaries.append(_train_one(args, weight, output))
    report = {"summaries": summaries}
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Supervised BC warm-start for frame-stacked temporal PPO on safe bomb + escape sequences.")
    parser.add_argument("--dataset", default="ml/datasets/fs4_selected_bomb_bc.npz")
    parser.add_argument("--base_policy", default="ml/checkpoints/rl_agent_temporal/ppo_framestack4_lr1e4_ent002_100k.zip")
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--bomb_loss_weights", nargs="+", default=["0.01", "0.02", "0.05"])
    parser.add_argument("--outputs", nargs="*", default=None)
    parser.add_argument("--output_prefix", default="ml/checkpoints/rl_agent_temporal/ppo_fs4_bc_bomb")
    parser.add_argument("--report_output", default="logs/pretrain_temporal_bc_report.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--escape_loss_weight", type=float, default=8.0)
    parser.add_argument("--bomb_metric_weight", type=float, default=0.2)
    parser.add_argument("--max_global_bomb_pred", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--trainable", choices=["action_head", "policy_head", "all_policy"], default="action_head")
    parser.add_argument("--seed", type=int, default=9700)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
