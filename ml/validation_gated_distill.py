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
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.model import BomberFeaturesExtractor
from ml.envs.bomber_gym_env import BomberGymEnv
from ml.evaluate_rl_pure import evaluate

PLACE_BOMB = 5


def _policy_logits(policy, observations):
    features = policy.extract_features(observations)
    if isinstance(features, tuple):
        features = features[0]
    latent_pi, _latent_vf = policy.mlp_extractor(features)
    return policy.action_net(latent_pi)


def _load_model(path: str, device: str, seed: int):
    env = Monitor(BomberGymEnv(agent_id=0, opponent_pool=["random", "simple"], max_steps=200, seed=seed))
    policy_kwargs = {
        "features_extractor_class": BomberFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    return PPO.load(path, env=env, device=device, custom_objects={"policy_kwargs": policy_kwargs})


def _freeze_to_action_head(policy):
    for param in policy.parameters():
        param.requires_grad = False
    for param in policy.action_net.parameters():
        param.requires_grad = True
    return [p for p in policy.action_net.parameters() if p.requires_grad]


def _one_update(model, data, rng, args):
    policy = model.policy
    device = policy.device
    params = _freeze_to_action_head(policy)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    bomb_obs = data["bomb_obs"].astype(np.float32)
    escape_obs = data["escape_obs"].astype(np.float32)
    escape_action = data["escape_action"].astype(np.int64)

    bomb_idx = rng.choice(np.arange(len(bomb_obs)), size=min(args.batch_size, len(bomb_obs)), replace=False)
    escape_idx = rng.choice(np.arange(len(escape_obs)), size=min(args.batch_size, len(escape_obs)), replace=False)
    batch_bomb = torch.as_tensor(bomb_obs[bomb_idx], dtype=torch.float32, device=device)
    batch_escape = torch.as_tensor(escape_obs[escape_idx], dtype=torch.float32, device=device)
    batch_action = torch.as_tensor(escape_action[escape_idx], dtype=torch.long, device=device)

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
    return {
        "loss": float(loss.item()),
        "bomb_loss": float(bomb_loss.item()),
        "escape_loss": float(escape_loss.item()),
    }


def _eval_checkpoint(path: str, episodes: int, max_steps: int, seed: int):
    random_result = evaluate(path, "random", episodes, max_steps, seed)
    simple_result = evaluate(path, "simple", episodes, max_steps, seed)
    return {"random": random_result, "simple": simple_result}


def _brief_metrics(eval_result):
    simple = eval_result["simple"]
    random = eval_result["random"]
    return {
        "simple_win": simple["win_rate"],
        "simple_draw": simple["draw_rate"],
        "simple_death": simple["death_rate"],
        "bomb_rate": simple["place_bomb_frequency"],
        "bomb_suicide": simple["bomb_suicide_rate"],
        "death_after_bomb": simple["death_within_7_steps_after_bomb"],
        "post_bomb_survival": simple["post_bomb_survival_steps_avg"],
        "boxes_per_bomb": simple["average_boxes_destroyed_per_bomb"],
        "useful_bombs": simple["useful_bomb_count"],
        "useless_bombs": simple["useless_bomb_count"],
        "random_win": random["win_rate"],
        "random_death": random["death_rate"],
        "invalid": simple["invalid_action_count"] + random["invalid_action_count"],
        "crash": simple["crash_count"] + random["crash_count"],
        "timeout": simple["timeout_count"] + random["timeout_count"],
    }


def _gate(candidate, baseline, args):
    reasons = []
    if candidate["invalid"] or candidate["crash"] or candidate["timeout"]:
        reasons.append("invalid_crash_timeout")
    if candidate["simple_death"] > baseline["simple_death"] + args.max_simple_death_delta:
        reasons.append("simple_death_regressed")
    if candidate["death_after_bomb"] > baseline["death_after_bomb"] + args.max_death_after_bomb_delta:
        reasons.append("death_after_bomb_regressed")
    if candidate["bomb_suicide"] > baseline["bomb_suicide"] + args.max_bomb_suicide_delta:
        reasons.append("bomb_suicide_regressed")
    if candidate["boxes_per_bomb"] < args.min_boxes_per_bomb:
        reasons.append("boxes_per_bomb_low")
    if candidate["random_win"] < args.min_random_win:
        reasons.append("random_win_low")
    return not reasons, reasons


def _confirm_multiseed(path: str, baseline_path: str, seeds: list[int], args):
    rows = []
    candidate_deaths = []
    baseline_deaths = []
    for seed in seeds:
        cand = _brief_metrics(_eval_checkpoint(path, args.confirm_eval_episodes, args.max_steps, seed))
        base = _brief_metrics(_eval_checkpoint(baseline_path, args.confirm_eval_episodes, args.max_steps, seed))
        rows.append({"seed": seed, "candidate": cand, "baseline": base})
        candidate_deaths.append(cand["simple_death"])
        baseline_deaths.append(base["simple_death"])
    return {
        "rows": rows,
        "candidate_simple_death_avg": float(np.mean(candidate_deaths)) if candidate_deaths else 0.0,
        "baseline_simple_death_avg": float(np.mean(baseline_deaths)) if baseline_deaths else 0.0,
    }


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.dataset)
    rng = np.random.default_rng(args.seed)

    baseline_eval = _eval_checkpoint(args.base_policy, args.eval_episodes, args.max_steps, args.eval_seed)
    baseline_metrics = _brief_metrics(baseline_eval)
    current_best_path = str(Path(args.base_policy))
    best_metrics = baseline_metrics
    accepted_any = False
    rejects = 0
    lr = args.lr
    rows = []

    for update_idx in range(1, args.max_updates + 1):
        model = _load_model(current_best_path, args.device, args.seed + update_idx)
        args.lr = lr
        train_metrics = _one_update(model, data, rng, args)
        temp_path = output_dir / f"temp_update_{update_idx:03d}.zip"
        model.save(str(temp_path))
        eval_result = _eval_checkpoint(str(temp_path), args.eval_episodes, args.max_steps, args.eval_seed)
        metrics = _brief_metrics(eval_result)
        accepted, reasons = _gate(metrics, baseline_metrics, args)

        row = {
            "update": update_idx,
            "lr": lr,
            "train": train_metrics,
            "metrics": metrics,
            "accepted": bool(accepted),
            "reasons": reasons,
            "checkpoint": str(temp_path),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

        if accepted:
            accepted_any = True
            rejects = 0
            best_metrics = metrics
            accepted_path = output_dir / f"accepted_update_{update_idx:03d}.zip"
            shutil.copy2(temp_path, accepted_path)
            current_best_path = str(accepted_path)
        else:
            rejects += 1
            lr *= args.lr_decay
            if rejects >= args.patience:
                break

    final_path = Path(args.final_output)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if accepted_any:
        shutil.copy2(current_best_path, final_path)
    else:
        shutil.copy2(args.base_policy, final_path)

    confirmation = None
    if accepted_any and args.confirm_seeds:
        confirmation = _confirm_multiseed(str(final_path), args.base_policy, args.confirm_seeds, args)
        if confirmation["candidate_simple_death_avg"] > confirmation["baseline_simple_death_avg"] + args.max_simple_death_delta:
            shutil.copy2(args.base_policy, final_path)
            accepted_any = False

    summary = {
        "base_policy": args.base_policy,
        "dataset": args.dataset,
        "baseline_metrics": baseline_metrics,
        "updates": rows,
        "accepted_any": bool(accepted_any),
        "best_metrics": best_metrics if accepted_any else baseline_metrics,
        "final_output": str(final_path),
        "confirmation": confirmation,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Validation-gated tiny sequence distillation for pure RL bomb behavior.")
    parser.add_argument("--base_policy", default="ml/checkpoints/rl_agent_pure/ppo_v3_selected_bomb_best.zip")
    parser.add_argument("--dataset", default="ml/datasets/rl_bc_selected_bomb_sequences_v3.npz")
    parser.add_argument("--output_dir", default="ml/checkpoints/rl_agent_pure/validation_gated_distill")
    parser.add_argument("--final_output", default="ml/checkpoints/rl_agent_pure/ppo_validation_gated_distill_best.zip")
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--confirm_eval_episodes", type=int, default=50)
    parser.add_argument("--eval_seed", type=int, default=1400)
    parser.add_argument("--confirm_seeds", nargs="*", type=int, default=[1401, 1402, 1403])
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--max_updates", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--bomb_loss_weight", type=float, default=0.01)
    parser.add_argument("--escape_loss_weight", type=float, default=8.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max_simple_death_delta", type=float, default=0.01)
    parser.add_argument("--max_death_after_bomb_delta", type=int, default=1)
    parser.add_argument("--max_bomb_suicide_delta", type=float, default=0.05)
    parser.add_argument("--min_boxes_per_bomb", type=float, default=1.0)
    parser.add_argument("--min_random_win", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
