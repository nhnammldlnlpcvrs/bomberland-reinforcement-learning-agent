from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd):
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _copy_agent_package(policy_zip: Path, target_dir: Path):
    source_dir = ROOT / "agent" / "rl_agent_pure"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("agent.py", "action_mask.py", "constants.py", "encoder.py", "model.py", "policy.py", "utils.py"):
        shutil.copyfile(source_dir / name, target_dir / name)
    shutil.copyfile(policy_zip, target_dir / "policy.zip")


def _load_eval(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["opponent"]: row for row in rows}


def _select_score(row):
    bomb_rate = float(row.get("place_bomb_frequency", 0.0))
    death_rate = float(row.get("death_rate", 1.0))
    win_rate = float(row.get("win_rate", 0.0))
    boxes_per_bomb = float(row.get("average_boxes_destroyed_per_bomb", 0.0))
    escape_success = float(row.get("bomb_escape_success_rate", 0.0))
    suicide_rate = float(row.get("bomb_suicide_rate", 1.0))
    return win_rate - death_rate + 0.5 * escape_success - 0.5 * suicide_rate + 0.2 * boxes_per_bomb + 0.1 * bomb_rate


def _passes_gate(row, args):
    return (
        float(row.get("place_bomb_frequency", 0.0)) >= args.min_bomb_rate
        and float(row.get("bomb_suicide_rate", 1.0)) <= args.max_bomb_suicide_rate
        and float(row.get("death_rate", 1.0)) <= args.max_simple_death_rate
        and int(row.get("invalid_action_count", 0)) == 0
        and int(row.get("crash_count", 0)) == 0
        and int(row.get("timeout_count", 0)) == 0
    )


def train(args):
    work_dir = ROOT / args.work_dir
    checkpoint_dir = ROOT / args.checkpoint_dir
    log_dir = ROOT / args.log_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    work_policy = work_dir / "policy.zip"
    start_policy = ROOT / args.start_policy
    if not start_policy.exists():
        raise FileNotFoundError(start_policy)
    shutil.copyfile(start_policy, work_policy)

    best_score = -1e9
    best_path = checkpoint_dir / "alternating_best.zip"
    history = []
    for cycle in range(1, args.cycles + 1):
        cycle_seed = args.seed + cycle * 100
        ppo_policy = work_dir / f"policy_cycle_{cycle}_ppo.zip"
        _run([
            sys.executable, "-m", "ml.train_sb3_ppo",
            "--stage", args.stage,
            "--total_timesteps", str(args.ppo_steps),
            "--n_envs", str(args.n_envs),
            "--seed", str(cycle_seed),
            "--max_steps", str(args.max_steps),
            "--n_steps", str(args.n_steps),
            "--batch_size", str(args.batch_size),
            "--opponents", *args.opponents,
            "--resume_bc", str(work_policy.relative_to(ROOT)),
            "--save_path", str(ppo_policy.relative_to(ROOT)),
            "--device", args.device,
            "--learning_rate", str(args.learning_rate),
            "--clip_range", str(args.clip_range),
            "--ent_coef", str(args.ent_coef),
            "--target_kl", str(args.target_kl),
            "--eval_freq", str(args.disable_inner_eval_freq),
            "--checkpoint_freq", str(args.disable_inner_eval_freq),
            "--bc_dataset", args.dataset,
            "--bc_coef", str(args.bc_aux_coef),
            "--bc_train_freq", str(args.bc_aux_train_freq),
            "--bc_bomb_multiplier", str(args.mixed_bomb_weight),
            "--bc_escape_multiplier", str(args.mixed_escape_weight),
            *(["--training_bomb_gate"] if args.training_bomb_gate else []),
        ])

        _run([
            sys.executable, "-m", "ml.pretrain_rl_policy_bc",
            "--dataset", args.dataset,
            "--base_policy", str(ppo_policy.relative_to(ROOT)),
            "--output", str(work_policy.relative_to(ROOT)),
            "--mode", args.bc_refresh_mode,
            "--epochs", str(args.bc_refresh_epochs),
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.bc_learning_rate),
            "--mixed_bomb_weight", str(args.mixed_bomb_weight),
            "--mixed_escape_weight", str(args.mixed_escape_weight),
            "--escape_bomb_weight", str(args.escape_bomb_weight),
            "--escape_move_weight", str(args.escape_move_weight),
            "--class_bomb_weight", str(args.class_bomb_weight),
            "--seed", str(cycle_seed + 1),
            "--device", args.device,
        ])

        eval_agent_dir = work_dir / "eval_agent"
        _copy_agent_package(work_policy, eval_agent_dir)
        eval_path = log_dir / f"alternating_cycle_{cycle}.json"
        _run([
            sys.executable, "-m", "ml.evaluate_rl_pure",
            "--agent_path", str(eval_agent_dir.relative_to(ROOT)),
            "--opponents", *args.eval_opponents,
            "--episodes", str(args.eval_episodes),
            "--max_steps", str(args.max_steps),
            "--seed", str(cycle_seed + 2),
            "--output", str(eval_path.relative_to(ROOT)),
        ])

        rows = _load_eval(eval_path)
        simple = rows.get("simple") or next(iter(rows.values()))
        score = _select_score(simple)
        accepted = _passes_gate(simple, args)
        cycle_path = checkpoint_dir / f"alternating_cycle_{cycle}.zip"
        shutil.copyfile(work_policy, cycle_path)
        if accepted and score > best_score:
            best_score = score
            shutil.copyfile(work_policy, best_path)
        history.append({
            "cycle": cycle,
            "checkpoint": str(cycle_path.relative_to(ROOT)),
            "accepted": accepted,
            "score": score,
            "simple": simple,
        })
        print(json.dumps(history[-1], indent=2), flush=True)

    summary = {
        "start_policy": args.start_policy,
        "dataset": args.dataset,
        "best_checkpoint": str(best_path.relative_to(ROOT)) if best_path.exists() else None,
        "best_score": best_score if best_path.exists() else None,
        "history": history,
    }
    summary_path = log_dir / "alternating_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Short-cycle PPO with post-bomb escape BC refresh.")
    parser.add_argument("--start_policy", default="agent/rl_agent_pure/bc_policy_bomb_escape.zip")
    parser.add_argument("--dataset", default="ml/datasets/rl_bc_bomb_escape.npz")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--ppo_steps", type=int, default=10_000)
    parser.add_argument("--bc_refresh_epochs", type=int, default=1)
    parser.add_argument("--stage", default="stage2")
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--n_steps", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--opponents", nargs="+", default=["random", "simple"])
    parser.add_argument("--eval_opponents", nargs="+", default=["random", "simple"])
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--clip_range", type=float, default=0.05)
    parser.add_argument("--ent_coef", type=float, default=0.02)
    parser.add_argument("--target_kl", type=float, default=0.01)
    parser.add_argument("--bc_learning_rate", type=float, default=3e-5)
    parser.add_argument("--mixed_bomb_weight", type=float, default=1.0)
    parser.add_argument("--mixed_escape_weight", type=float, default=6.0)
    parser.add_argument("--escape_bomb_weight", type=float, default=0.0)
    parser.add_argument("--escape_move_weight", type=float, default=4.0)
    parser.add_argument("--class_bomb_weight", type=float, default=0.0)
    parser.add_argument("--bc_refresh_mode", choices=["all", "bomb_only", "post_bomb_escape", "mixed_bomb_escape"], default="post_bomb_escape")
    parser.add_argument("--bc_aux_coef", type=float, default=0.01)
    parser.add_argument("--bc_aux_train_freq", type=int, default=1024)
    parser.add_argument("--min_bomb_rate", type=float, default=0.002)
    parser.add_argument("--max_bomb_suicide_rate", type=float, default=0.30)
    parser.add_argument("--max_simple_death_rate", type=float, default=0.79)
    parser.add_argument("--training_bomb_gate", action="store_true")
    parser.add_argument("--work_dir", default="ml/checkpoints/rl_agent_pure/alternating_work")
    parser.add_argument("--checkpoint_dir", default="ml/checkpoints/rl_agent_pure")
    parser.add_argument("--log_dir", default="logs/rl_agent_pure_alternating")
    parser.add_argument("--disable_inner_eval_freq", type=int, default=1_000_000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
