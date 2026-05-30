from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT_FILES = ("agent.py", "action_mask.py", "constants.py", "encoder.py", "model.py", "policy.py", "utils.py")


def _run(cmd):
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _copy_agent(policy_zip: Path, target_dir: Path):
    source_dir = ROOT / "agent" / "rl_agent_pure"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in AGENT_FILES:
        shutil.copyfile(source_dir / name, target_dir / name)
    shutil.copyfile(policy_zip, target_dir / "policy.zip")


def _evaluate(agent_dir: Path, args, seed: int, output: Path):
    _run([
        sys.executable, "-m", "ml.evaluate_rl_pure",
        "--agent_path", str(agent_dir.relative_to(ROOT)),
        "--opponents", *args.opponents,
        "--episodes", str(args.eval_episodes),
        "--max_steps", str(args.max_steps),
        "--seed", str(seed),
        "--output", str(output.relative_to(ROOT)),
    ])
    rows = json.loads(output.read_text(encoding="utf-8"))
    return {row["opponent"]: row for row in rows}


def _metric(row, key, default=0.0):
    return float(row.get(key, default))


def _death_after_bomb_rate(row):
    return _metric(row, "death_within_7_steps_after_bomb") / max(1.0, _metric(row, "place_bomb_count"))


def _score(simple):
    return (
        _metric(simple, "win_rate")
        - _metric(simple, "death_rate")
        - 0.5 * _metric(simple, "bomb_suicide_rate")
        - 0.2 * _death_after_bomb_rate(simple)
        + 0.1 * _metric(simple, "average_boxes_destroyed_per_bomb")
    )


def _reject_reasons(candidate_simple, candidate_random, best_simple, best_random, args):
    reasons = []
    bomb_rate = _metric(candidate_simple, "place_bomb_frequency")
    boxes_per_bomb = _metric(candidate_simple, "average_boxes_destroyed_per_bomb")
    if int(candidate_simple.get("invalid_action_count", 0)) or int(candidate_simple.get("crash_count", 0)) or int(candidate_simple.get("timeout_count", 0)):
        reasons.append("simple invalid/crash/timeout")
    if int(candidate_random.get("invalid_action_count", 0)) or int(candidate_random.get("crash_count", 0)) or int(candidate_random.get("timeout_count", 0)):
        reasons.append("random invalid/crash/timeout")
    if not (args.target_bomb_min <= bomb_rate <= args.target_bomb_max):
        reasons.append(f"bomb_rate {bomb_rate:.4f} outside [{args.target_bomb_min:.4f}, {args.target_bomb_max:.4f}]")
    if boxes_per_bomb <= args.min_boxes_per_bomb:
        reasons.append(f"boxes_per_bomb {boxes_per_bomb:.3f} <= {args.min_boxes_per_bomb:.3f}")
    if _metric(candidate_simple, "death_rate") > min(args.max_simple_death, _metric(best_simple, "death_rate")):
        reasons.append("simple death worsened")
    suicide_improved = _metric(candidate_simple, "bomb_suicide_rate") < _metric(best_simple, "bomb_suicide_rate")
    death_after_improved = _metric(candidate_simple, "death_within_7_steps_after_bomb") < _metric(best_simple, "death_within_7_steps_after_bomb")
    if not (suicide_improved or death_after_improved):
        reasons.append("bomb safety did not improve")
    if _metric(candidate_simple, "bomb_suicide_rate") > args.max_bomb_suicide and not death_after_improved:
        reasons.append("bomb suicide above threshold without death-after improvement")
    if _metric(candidate_random, "win_rate") < _metric(best_random, "win_rate") - args.max_random_win_drop:
        reasons.append("random win degraded")
    if _metric(candidate_random, "death_rate") > _metric(best_random, "death_rate") + args.max_random_death_increase:
        reasons.append("random death degraded")
    return reasons


def _summary_row(cycle, seed, path, accepted, reasons, simple, random_row):
    tier = "reject"
    bomb_rate = simple["place_bomb_frequency"]
    if accepted:
        tier = "A"
    elif 0.0005 <= bomb_rate < 0.001 and simple["death_rate"] <= 0.47 and simple["death_within_7_steps_after_bomb"] <= 8 and simple["average_boxes_destroyed_per_bomb"] > 0.5:
        tier = "B"
    elif simple["death_rate"] <= 0.47:
        tier = "C"
    return {
        "cycle": cycle,
        "seed": seed,
        "policy_path": str(path.relative_to(ROOT)),
        "accepted": accepted,
        "tier": tier,
        "reject_reasons": reasons,
        "score": _score(simple),
        "simple": {
            "win_rate": simple["win_rate"],
            "draw_rate": simple["draw_rate"],
            "death_rate": simple["death_rate"],
            "bomb_rate": simple["place_bomb_frequency"],
            "bomb_suicide_rate": simple["bomb_suicide_rate"],
            "death_within_7_steps_after_bomb": simple["death_within_7_steps_after_bomb"],
            "post_bomb_survival_steps_avg": simple["post_bomb_survival_steps_avg"],
            "boxes_per_bomb": simple["average_boxes_destroyed_per_bomb"],
            "place_bomb_count": simple["place_bomb_count"],
        },
        "random": {
            "win_rate": random_row["win_rate"],
            "death_rate": random_row["death_rate"],
            "bomb_rate": random_row["place_bomb_frequency"],
            "bomb_suicide_rate": random_row["bomb_suicide_rate"],
        },
    }


def train(args):
    save_dir = ROOT / args.save_dir
    work_dir = save_dir / "work"
    eval_dir = save_dir / "eval_agents"
    log_dir = ROOT / args.log_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    current_best = ROOT / args.start_policy
    if not current_best.exists():
        raise FileNotFoundError(current_best)
    best_path = save_dir / "ppo_escape_gated_best.zip"
    best_by_death_path = save_dir / "ppo_escape_best_by_death.zip"
    best_by_death_after_bomb_path = save_dir / "ppo_escape_best_by_death_after_bomb.zip"
    best_by_score_even_if_bomb_low_path = save_dir / "ppo_escape_best_by_score_even_if_bomb_low.zip"
    shutil.copyfile(current_best, best_path)
    shutil.copyfile(current_best, best_by_death_path)
    shutil.copyfile(current_best, best_by_death_after_bomb_path)
    shutil.copyfile(current_best, best_by_score_even_if_bomb_low_path)

    best_agent = eval_dir / "best_start"
    _copy_agent(best_path, best_agent)
    baseline_eval_path = log_dir / "baseline_eval.json"
    baseline = _evaluate(best_agent, args, args.seed_base, baseline_eval_path)
    best_simple = baseline.get("simple") or next(iter(baseline.values()))
    best_random = baseline.get("random") or next(iter(baseline.values()))
    best_score = _score(best_simple)
    best_death = _metric(best_simple, "death_rate")
    best_death_after_bomb = _metric(best_simple, "death_within_7_steps_after_bomb")
    best_low_bomb_score = best_score
    history = [{
        "cycle": 0,
        "seed": args.seed_base,
        "policy_path": str(best_path.relative_to(ROOT)),
        "accepted": True,
        "reject_reasons": [],
        "score": best_score,
        "simple": {
            "win_rate": best_simple["win_rate"],
            "draw_rate": best_simple["draw_rate"],
            "death_rate": best_simple["death_rate"],
            "bomb_rate": best_simple["place_bomb_frequency"],
            "bomb_suicide_rate": best_simple["bomb_suicide_rate"],
            "death_within_7_steps_after_bomb": best_simple["death_within_7_steps_after_bomb"],
            "post_bomb_survival_steps_avg": best_simple["post_bomb_survival_steps_avg"],
            "boxes_per_bomb": best_simple["average_boxes_destroyed_per_bomb"],
            "place_bomb_count": best_simple["place_bomb_count"],
        },
        "random": {
            "win_rate": best_random["win_rate"],
            "death_rate": best_random["death_rate"],
            "bomb_rate": best_random["place_bomb_frequency"],
            "bomb_suicide_rate": best_random["bomb_suicide_rate"],
        },
    }]

    seeds = args.seeds or [701 + i for i in range(args.cycles)]
    for cycle, seed in enumerate(seeds[: args.cycles], start=1):
        candidate = work_dir / f"candidate_cycle_{cycle}_seed{seed}.zip"
        _run([
            sys.executable, "-m", "ml.train_sb3_ppo",
            "--stage", args.stage,
            "--total_timesteps", str(args.ppo_steps),
            "--n_envs", str(args.n_envs),
            "--seed", str(seed),
            "--max_steps", str(args.max_steps),
            "--n_steps", str(args.n_steps),
            "--batch_size", str(args.batch_size),
            "--opponents", *args.train_opponents,
            "--resume_bc", str(best_path.relative_to(ROOT)),
            "--save_path", str(candidate.relative_to(ROOT)),
            "--device", args.device,
            "--learning_rate", str(args.learning_rate),
            "--clip_range", str(args.clip_range),
            "--ent_coef", str(args.ent_coef),
            "--target_kl", str(args.target_kl),
            "--eval_freq", str(args.disable_inner_eval_freq),
            "--checkpoint_freq", str(args.disable_inner_eval_freq),
            "--bc_dataset", args.dataset,
            "--bc_coef", str(args.bc_coef),
            "--bc_train_freq", str(args.bc_train_freq),
            "--bc_bomb_multiplier", "0.0",
            "--bc_escape_multiplier", str(args.bc_escape_multiplier),
            "--training_bomb_gate",
        ])
        candidate_agent = eval_dir / f"candidate_cycle_{cycle}_seed{seed}"
        _copy_agent(candidate, candidate_agent)
        eval_path = log_dir / f"cycle_{cycle}_seed{seed}.json"
        rows = _evaluate(candidate_agent, args, seed + 10_000, eval_path)
        simple = rows.get("simple") or next(iter(rows.values()))
        random_row = rows.get("random") or next(iter(rows.values()))
        reasons = _reject_reasons(simple, random_row, best_simple, best_random, args)
        score = _score(simple)
        accepted = not reasons and score >= best_score - args.max_score_regression
        candidate_archive = save_dir / f"ppo_escape_gated_cycle_{cycle}_seed{seed}.zip"
        shutil.copyfile(candidate, candidate_archive)
        if _metric(simple, "death_rate") < best_death:
            best_death = _metric(simple, "death_rate")
            shutil.copyfile(candidate, best_by_death_path)
        if _metric(simple, "death_within_7_steps_after_bomb") < best_death_after_bomb:
            best_death_after_bomb = _metric(simple, "death_within_7_steps_after_bomb")
            shutil.copyfile(candidate, best_by_death_after_bomb_path)
        if score > best_low_bomb_score:
            best_low_bomb_score = score
            shutil.copyfile(candidate, best_by_score_even_if_bomb_low_path)
        if accepted:
            best_score = score
            best_simple = simple
            best_random = random_row
            shutil.copyfile(candidate, best_path)
        row = _summary_row(cycle, seed, candidate_archive, accepted, reasons, simple, random_row)
        history.append(row)
        print(json.dumps(row, indent=2), flush=True)

    summary = {
        "start_policy": args.start_policy,
        "dataset": args.dataset,
        "best_checkpoint": str(best_path.relative_to(ROOT)),
        "best_by_death": str(best_by_death_path.relative_to(ROOT)),
        "best_by_death_after_bomb": str(best_by_death_after_bomb_path.relative_to(ROOT)),
        "best_by_score_even_if_bomb_low": str(best_by_score_even_if_bomb_low_path.relative_to(ROOT)),
        "best_score": best_score,
        "history": history,
    }
    summary_path = log_dir / "gated_escape_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Checkpoint-gated short PPO cycles for post-bomb escape safety.")
    parser.add_argument("--start_policy", default="ml/checkpoints/rl_agent_pure/ppo_successful_escape_aux_10k.zip")
    parser.add_argument("--dataset", default="ml/datasets/rl_bc_successful_escape_only.npz")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--seed_base", type=int, default=610)
    parser.add_argument("--ppo_steps", type=int, default=5_000)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--opponents", nargs="+", default=["random", "simple"])
    parser.add_argument("--train_opponents", nargs="+", default=["random", "simple"])
    parser.add_argument("--save_dir", default="ml/checkpoints/rl_agent_pure/gated_escape")
    parser.add_argument("--log_dir", default="logs/rl_agent_pure_gated_escape")
    parser.add_argument("--target_bomb_min", type=float, default=0.001)
    parser.add_argument("--target_bomb_max", type=float, default=0.005)
    parser.add_argument("--max_bomb_suicide", type=float, default=0.5)
    parser.add_argument("--max_simple_death", type=float, default=0.51)
    parser.add_argument("--min_boxes_per_bomb", type=float, default=0.5)
    parser.add_argument("--max_random_win_drop", type=float, default=0.05)
    parser.add_argument("--max_random_death_increase", type=float, default=0.05)
    parser.add_argument("--max_score_regression", type=float, default=0.02)
    parser.add_argument("--stage", default="stage2")
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--clip_range", type=float, default=0.05)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--target_kl", type=float, default=0.005)
    parser.add_argument("--bc_coef", type=float, default=0.002)
    parser.add_argument("--bc_train_freq", type=int, default=1024)
    parser.add_argument("--bc_escape_multiplier", type=float, default=6.0)
    parser.add_argument("--disable_inner_eval_freq", type=int, default=1_000_000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
