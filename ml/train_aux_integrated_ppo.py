from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.aux_model import BomberAuxModel
from agent.rl_agent_pure.model import BomberFeaturesExtractor
from ml.envs.bomber_gym_env import BomberGymEnv
from ml.evaluate_rl_pure import evaluate
from ml.pretrain_ppo_features_aux import _aux_loss, _evaluate_aux, _load_dataset, _make_loader


def _load_ppo(path: str, device: str, env=None) -> PPO:
    policy_kwargs = {
        "features_extractor_class": BomberFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    return PPO.load(path, env=env, device=device, custom_objects={"policy_kwargs": policy_kwargs})


def _policy_logits(model: PPO, obs: torch.Tensor) -> torch.Tensor:
    model.policy.eval()
    with torch.no_grad():
        dist = model.policy.get_distribution(obs)
        return dist.distribution.logits.detach().cpu()


def _logit_drift(model: PPO, reference: torch.Tensor, obs: torch.Tensor) -> dict:
    after = _policy_logits(model, obs)
    delta = torch.abs(after - reference)
    return {"max_logit_delta": float(delta.max()), "mean_logit_delta": float(delta.mean())}


def _value_parameters(model: PPO) -> list[torch.nn.Parameter]:
    params = []
    for name, param in model.policy.named_parameters():
        allow = ("value_net" in name) or ("mlp_extractor.value_net" in name)
        param.requires_grad = allow
        if allow:
            params.append(param)
    return params


def _collect_rollout_returns(model: PPO, args) -> tuple[torch.Tensor, torch.Tensor]:
    env = BomberGymEnv(
        agent_id=args.agent_id,
        opponent_pool=args.opponents,
        max_steps=args.max_steps,
        seed=args.seed,
        training_bomb_gate=args.training_bomb_gate,
    )
    observations: list[np.ndarray] = []
    rewards: list[float] = []
    episode_breaks: list[int] = []
    obs, _info = env.reset(seed=args.seed)
    while len(observations) < args.stage1_timesteps:
        observations.append(obs.astype(np.float32))
        action, _state = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, _info = env.step(int(np.asarray(action).reshape(-1)[0]))
        rewards.append(float(reward))
        if terminated or truncated:
            episode_breaks.append(len(rewards))
            obs, _info = env.reset(seed=args.seed + len(episode_breaks))
    if not episode_breaks or episode_breaks[-1] != len(rewards):
        episode_breaks.append(len(rewards))

    returns = np.zeros(len(rewards), dtype=np.float32)
    start = 0
    for end in episode_breaks:
        running = 0.0
        for idx in range(end - 1, start - 1, -1):
            running = rewards[idx] + args.gamma * running
            returns[idx] = running
        start = end
    return torch.as_tensor(np.asarray(observations), dtype=torch.float32), torch.as_tensor(returns, dtype=torch.float32)


def _train_value_on_rollouts(model: PPO, obs: torch.Tensor, returns: torch.Tensor, args, device: torch.device) -> list[dict]:
    trainable = _value_parameters(model)
    if not trainable:
        return [{"error": "no_value_parameters_found"}]
    dataset = TensorDataset(obs, returns)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(trainable, lr=args.stage1_value_lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(args.stage1_epochs):
        model.policy.train()
        running = 0.0
        count = 0
        for batch_obs, batch_returns in loader:
            batch_obs = batch_obs.to(device)
            batch_returns = batch_returns.to(device)
            values = model.policy.predict_values(batch_obs).flatten()
            loss = F.smooth_l1_loss(values, batch_returns)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 0.5)
            optimizer.step()
            running += float(loss.item()) * len(batch_obs)
            count += len(batch_obs)
        with torch.no_grad():
            values = model.policy.predict_values(obs[: min(len(obs), 2048)].to(device)).flatten().cpu()
            target = returns[: min(len(returns), 2048)]
            val_loss = float(F.smooth_l1_loss(values, target))
        history.append({"epoch": epoch + 1, "value_train_loss": running / max(1, count), "value_probe_loss": val_loss})
        print(json.dumps({"stage1_value": history[-1]}))
    return history


def _train_aux_replay(args, device: torch.device) -> dict:
    tensors, train_idx, val_idx, _normalization = _load_dataset(args.aux_dataset, args.seed, 0.2)
    train_loader = _make_loader(tensors, train_idx, args.batch_size, shuffle=True)
    val_loader = _make_loader(tensors, val_idx, args.batch_size, shuffle=False)
    aux = BomberAuxModel(features_dim=256).to(device)
    if args.aux_features and Path(args.aux_features).exists():
        checkpoint = torch.load(args.aux_features, map_location=device, weights_only=False)
        aux.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimizer = torch.optim.Adam(aux.parameters(), lr=args.aux_lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(args.aux_epochs):
        aux.train()
        running = 0.0
        count = 0
        for batch in train_loader:
            batch = [item.to(device) for item in batch]
            out = aux(batch[0])
            loss = _aux_loss(out, batch, args) * args.aux_loss_coef
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(aux.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item()) * len(batch[0])
            count += len(batch[0])
        metrics = _evaluate_aux(aux, val_loader, device)
        metrics["epoch"] = epoch + 1
        metrics["scaled_aux_train_loss"] = running / max(1, count)
        history.append(metrics)
        print(json.dumps({"stage1_aux_replay": metrics}))
    output = Path(args.stage1_aux_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": aux.state_dict(), "features_dim": 256, "metrics": history[-1] if history else {}, "history": history}, output)
    return {"path": str(output), "metrics": history[-1] if history else {}}


class BaselineKLCallback(BaseCallback):
    def __init__(self, baseline_path: str, dataset: str, coef: float, batch_size: int = 256, train_freq: int = 512, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.baseline_path = baseline_path
        self.dataset = dataset
        self.coef = float(coef)
        self.batch_size = int(batch_size)
        self.train_freq = int(train_freq)
        self.obs = None
        self.teacher_probs = None

    def _logits(self, policy, observations):
        dist = policy.get_distribution(observations)
        return dist.distribution.logits

    def _on_training_start(self):
        if self.coef <= 0:
            return
        data = np.load(self.dataset)
        obs = data["observations"].astype(np.float32)
        if len(obs) > 4096:
            idx = np.random.choice(len(obs), size=4096, replace=False)
            obs = obs[idx]
        teacher = _load_ppo(self.baseline_path, str(self.model.policy.device))
        device = self.model.policy.device
        probs = []
        teacher.policy.eval()
        with torch.no_grad():
            for start in range(0, len(obs), 512):
                batch = torch.as_tensor(obs[start:start + 512], dtype=torch.float32, device=device)
                probs.append(torch.softmax(self._logits(teacher.policy, batch), dim=1).cpu().numpy())
        self.obs = obs
        self.teacher_probs = np.concatenate(probs, axis=0).astype(np.float32)

    def _on_step(self):
        if self.obs is None or self.n_calls % self.train_freq != 0:
            return True
        idx = np.random.choice(len(self.obs), size=min(self.batch_size, len(self.obs)), replace=True)
        device = self.model.policy.device
        obs = torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device)
        teacher_probs = torch.as_tensor(self.teacher_probs[idx], dtype=torch.float32, device=device)
        log_probs = torch.log_softmax(self._logits(self.model.policy, obs), dim=1)
        teacher_log_probs = torch.log(torch.clamp(teacher_probs, min=1e-8))
        loss = (teacher_probs * (teacher_log_probs - log_probs)).sum(dim=1).mean() * self.coef
        self.model.policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 0.5)
        self.model.policy.optimizer.step()
        return True


def _make_stage2_env(args, rank: int):
    return Monitor(BomberGymEnv(
        agent_id=args.agent_id,
        opponent_pool=args.opponents,
        max_steps=args.max_steps,
        seed=args.seed + 1000 + rank,
        training_bomb_gate=args.training_bomb_gate,
    ))


def _run_stage2(args, device: torch.device, drift_obs: torch.Tensor, before_logits: torch.Tensor) -> dict:
    env = DummyVecEnv([lambda rank=rank: _make_stage2_env(args, rank) for rank in range(args.n_envs)])
    model = _load_ppo(args.stage1_output, args.device, env=env)
    model.learning_rate = args.stage2_lr
    model.lr_schedule = get_schedule_fn(args.stage2_lr)
    model.clip_range = get_schedule_fn(args.stage2_clip_range)
    model.ent_coef = args.stage2_ent_coef
    model.target_kl = args.stage2_target_kl
    callbacks = []
    if args.kl_coef > 0:
        callbacks.append(BaselineKLCallback(args.baseline_policy, args.aux_dataset, args.kl_coef, args.batch_size))
    model.learn(total_timesteps=args.stage2_timesteps, callback=callbacks)
    drift = _logit_drift(model, before_logits, drift_obs)
    output = Path(args.stage2_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    return {"path": str(output), **drift}


def _eval_checkpoint(path: str, args, seed: int) -> list[dict]:
    return [evaluate(path, opponent, args.eval_episodes, args.eval_max_steps, seed + idx * 1000) for idx, opponent in enumerate(args.eval_opponents)]


def _summarize_eval(rows: list[dict]) -> dict:
    simple = next((row for row in rows if row["opponent"] == "simple"), rows[-1])
    random_row = next((row for row in rows if row["opponent"] == "random"), rows[0])
    return {
        "random_win": float(random_row["win_rate"]),
        "random_death": float(random_row["death_rate"]),
        "simple_win": float(simple["win_rate"]),
        "simple_death": float(simple["death_rate"]),
        "simple_bomb_rate": float(simple["place_bomb_frequency"]),
        "simple_bomb_suicide": float(simple["bomb_suicide_rate"]),
        "simple_death_after_bomb": int(simple["death_within_7_steps_after_bomb"]),
        "simple_boxes_per_bomb": float(simple["average_boxes_destroyed_per_bomb"]),
        "invalid": int(sum(row["invalid_action_count"] for row in rows)),
        "crash": int(sum(row["crash_count"] for row in rows)),
        "timeout": int(sum(row["timeout_count"] for row in rows)),
    }


def _passes_stage2(summary: dict, baseline: dict) -> bool:
    return (
        summary["invalid"] == 0
        and summary["crash"] == 0
        and summary["timeout"] == 0
        and summary["random_win"] >= 0.95
        and summary["simple_death"] <= baseline["simple_death"] + 0.02
        and summary["simple_death_after_bomb"] <= baseline["simple_death_after_bomb"]
        and summary["simple_bomb_suicide"] <= baseline["simple_bomb_suicide"]
        and (summary["simple_bomb_rate"] == 0.0 or summary["simple_boxes_per_bomb"] > 1.0)
    )


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    aux_tensors, _train_idx, val_idx, _norm = _load_dataset(args.aux_dataset, args.seed, 0.2)
    drift_obs = aux_tensors[0][val_idx[: min(args.drift_samples, len(val_idx))]].to(device)
    baseline = _load_ppo(args.baseline_policy, args.device)
    warm = _load_ppo(args.start_policy, args.device)
    baseline_logits = _policy_logits(baseline, drift_obs)
    warm_logits = _policy_logits(warm, drift_obs)

    report: dict = {
        "baseline_policy": args.baseline_policy,
        "start_policy": args.start_policy,
        "initial_warm_vs_baseline_drift": {
            "max_logit_delta": float(torch.abs(warm_logits - baseline_logits).max()),
            "mean_logit_delta": float(torch.abs(warm_logits - baseline_logits).mean()),
        },
    }

    obs, returns = _collect_rollout_returns(warm, args)
    value_history = _train_value_on_rollouts(warm, obs, returns, args, device)
    stage1_drift = _logit_drift(warm, warm_logits, drift_obs)
    stage1_path = Path(args.stage1_output)
    stage1_path.parent.mkdir(parents=True, exist_ok=True)
    if stage1_drift["max_logit_delta"] <= args.max_stage1_logit_delta:
        warm.save(stage1_path)
        stage1_saved = True
    else:
        stage1_saved = False
    report["stage1"] = {
        "path": str(stage1_path),
        "saved": stage1_saved,
        "rollout_samples": int(len(obs)),
        "value_history": value_history,
        **stage1_drift,
    }
    if args.aux_loss_coef > 0 and args.aux_epochs > 0:
        report["stage1"]["aux_replay"] = _train_aux_replay(args, device)

    eval_table = {}
    for name, path in (
        ("baseline", args.baseline_policy),
        ("warm_start", args.start_policy),
        ("stage1", str(stage1_path) if stage1_saved else args.start_policy),
    ):
        rows = _eval_checkpoint(path, args, args.eval_seed)
        eval_table[name] = {"rows": rows, "summary": _summarize_eval(rows)}
    report["evaluation"] = eval_table

    selected = args.baseline_policy
    selected_reason = "baseline_default"
    baseline_summary = eval_table["baseline"]["summary"]
    stage1_summary = eval_table["stage1"]["summary"]
    stage1_ok = (
        stage1_saved
        and stage1_drift["max_logit_delta"] <= args.max_stage1_logit_delta
        and stage1_summary["invalid"] == 0
        and stage1_summary["crash"] == 0
        and stage1_summary["timeout"] == 0
        and stage1_summary["simple_death"] <= baseline_summary["simple_death"] + args.stage1_max_death_regression
    )
    if stage1_ok:
        selected = str(stage1_path)
        selected_reason = "stage1_actor_frozen_passed"

    if args.run_stage2 and stage1_ok:
        stage2 = _run_stage2(args, device, drift_obs, warm_logits)
        rows = _eval_checkpoint(stage2["path"], args, args.eval_seed)
        eval_table["stage2"] = {"rows": rows, "summary": _summarize_eval(rows)}
        report["stage2"] = stage2
        if _passes_stage2(eval_table["stage2"]["summary"], baseline_summary):
            selected = stage2["path"]
            selected_reason = "stage2_tiny_actor_passed"
        else:
            selected_reason = f"{selected_reason}; stage2_rejected"
    elif args.run_stage2:
        report["stage2"] = {"skipped": True, "reason": "stage1_gate_failed"}

    selected_path = Path(args.selected_output)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected, selected_path)
    report["selected"] = {"path": str(selected_path), "source": selected, "reason": selected_reason}

    output = Path(args.report_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Conservative aux/critic warm-start PPO integration with strict actor-drift gates.")
    parser.add_argument("--baseline_policy", default="ml/checkpoints/rl_agent_pure/ppo_bomb_selector_best.zip")
    parser.add_argument("--start_policy", default="ml/checkpoints/rl_agent_pure/ppo_aux_pretrained_critic.zip")
    parser.add_argument("--aux_features", default="ml/checkpoints/rl_agent_pure/aux_pretrained_features.pt")
    parser.add_argument("--aux_dataset", default="ml/datasets/aux_pretrain_dataset_v3.npz")
    parser.add_argument("--stage1_output", default="ml/checkpoints/rl_agent_pure/ppo_aux_stage1_frozen.zip")
    parser.add_argument("--stage1_aux_output", default="ml/checkpoints/rl_agent_pure/aux_integrated_stage1.pt")
    parser.add_argument("--stage2_output", default="ml/checkpoints/rl_agent_pure/ppo_aux_stage2_tiny_actor.zip")
    parser.add_argument("--selected_output", default="ml/checkpoints/rl_agent_pure/ppo_aux_integrated_best.zip")
    parser.add_argument("--report_output", default="logs/aux_integrated_ppo_report.json")
    parser.add_argument("--stage1_timesteps", type=int, default=5000)
    parser.add_argument("--stage1_epochs", type=int, default=2)
    parser.add_argument("--stage1_value_lr", type=float, default=5e-5)
    parser.add_argument("--run_stage2", action="store_true")
    parser.add_argument("--stage2_timesteps", type=int, default=3000)
    parser.add_argument("--stage2_lr", type=float, default=1e-5)
    parser.add_argument("--stage2_target_kl", type=float, default=0.003)
    parser.add_argument("--stage2_clip_range", type=float, default=0.03)
    parser.add_argument("--stage2_ent_coef", type=float, default=0.005)
    parser.add_argument("--kl_coef", type=float, default=0.03)
    parser.add_argument("--aux_loss_coef", type=float, default=0.001)
    parser.add_argument("--aux_lr", type=float, default=1e-4)
    parser.add_argument("--aux_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--opponents", nargs="*", default=["random", "simple"])
    parser.add_argument("--training_bomb_gate", action="store_true", default=True)
    parser.add_argument("--no_training_bomb_gate", action="store_false", dest="training_bomb_gate")
    parser.add_argument("--eval_opponents", nargs="*", default=["random", "simple"])
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--eval_max_steps", type=int, default=500)
    parser.add_argument("--eval_seed", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7900)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--drift_samples", type=int, default=512)
    parser.add_argument("--max_stage1_logit_delta", type=float, default=1e-6)
    parser.add_argument("--stage1_max_death_regression", type=float, default=0.02)
    parser.add_argument("--death_weight", type=float, default=1.0)
    parser.add_argument("--escape_weight", type=float, default=0.5)
    parser.add_argument("--escape_available_weight", type=float, default=1.5)
    parser.add_argument("--bomb_escape_available_weight", type=float, default=1.5)
    parser.add_argument("--trapped_weight", type=float, default=1.5)
    parser.add_argument("--future_blast_weight", type=float, default=1.0)
    parser.add_argument("--box_weight", type=float, default=0.5)
    parser.add_argument("--reachable_weight", type=float, default=0.25)
    parser.add_argument("--safe_tiles_weight", type=float, default=0.25)
    parser.add_argument("--blast_distance_weight", type=float, default=0.25)
    parser.add_argument("--return_weight", type=float, default=0.25)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
