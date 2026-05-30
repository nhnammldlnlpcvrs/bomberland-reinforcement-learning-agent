from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.evaluate_rl_pure import evaluate


STAGES = (
    ("escape_only", "escape_only", 100_000, 40, 1e-4),
    ("bomb_then_escape", "bomb_then_escape", 100_000, 60, 1e-4),
    ("bomb_box_value", "bomb_box_value", 100_000, 80, 5e-5),
    ("mixed", "mixed", 200_000, 120, 5e-5),
)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _aggregate(results: list[dict]) -> dict:
    by_opponent: dict[str, list[dict]] = {}
    for row in results:
        by_opponent.setdefault(str(row["opponent"]), []).append(row)
    summary = {}
    for opponent, rows in by_opponent.items():
        summary[opponent] = {
            "win_rate": _mean([float(row["win_rate"]) for row in rows]),
            "draw_rate": _mean([float(row["draw_rate"]) for row in rows]),
            "death_rate": _mean([float(row["death_rate"]) for row in rows]),
            "average_survival_step": _mean([float(row["average_survival_step"]) for row in rows]),
            "place_bomb_frequency": _mean([float(row["place_bomb_frequency"]) for row in rows]),
            "bomb_suicide_rate": _mean([float(row["bomb_suicide_rate"]) for row in rows]),
            "death_within_7_steps_after_bomb": _mean([float(row["death_within_7_steps_after_bomb"]) for row in rows]),
            "post_bomb_survival_steps_avg": _mean([float(row["post_bomb_survival_steps_avg"]) for row in rows]),
            "average_boxes_destroyed_per_bomb": _mean([float(row["average_boxes_destroyed_per_bomb"]) for row in rows]),
            "useful_bomb_count": _mean([float(row["useful_bomb_count"]) for row in rows]),
            "useless_bomb_count": _mean([float(row["useless_bomb_count"]) for row in rows]),
            "invalid_action_count": float(sum(int(row["invalid_action_count"]) for row in rows)),
            "crash_count": float(sum(int(row["crash_count"]) for row in rows)),
            "timeout_count": float(sum(int(row["timeout_count"]) for row in rows)),
        }
    return summary


def evaluate_checkpoint(path: Path, opponents: list[str], episodes: int, max_steps: int, seeds: list[int]) -> dict:
    rows = []
    for seed in seeds:
        for opponent in opponents:
            rows.append(evaluate(str(path), opponent, episodes, max_steps, seed))
    return {"rows": rows, "summary": _aggregate(rows)}


def _metric(summary: dict, opponent: str, key: str, default: float = 0.0) -> float:
    return float(summary.get(opponent, {}).get(key, default))


def gate_candidate(candidate: dict, previous: dict, args) -> tuple[str, list[str]]:
    reasons = []
    cand = candidate["summary"]
    prev = previous["summary"]

    total_invalid = sum(_metric(cand, opponent, "invalid_action_count") for opponent in args.opponents)
    total_crash = sum(_metric(cand, opponent, "crash_count") for opponent in args.opponents)
    total_timeout = sum(_metric(cand, opponent, "timeout_count") for opponent in args.opponents)
    if total_invalid > 0:
        reasons.append(f"invalid_actions={total_invalid:g}")
    if total_crash > 0:
        reasons.append(f"crashes={total_crash:g}")
    if total_timeout > 0:
        reasons.append(f"timeouts={total_timeout:g}")

    random_win = _metric(cand, "random", "win_rate")
    if "random" in args.opponents and random_win < args.min_random_win:
        reasons.append(f"random_win {random_win:.3f} < {args.min_random_win:.3f}")

    simple_death = _metric(cand, "simple", "death_rate")
    prev_simple_death = _metric(prev, "simple", "death_rate")
    if "simple" in args.opponents and simple_death > prev_simple_death + args.max_simple_death_regression:
        reasons.append(
            f"simple_death {simple_death:.3f} > previous {prev_simple_death:.3f} + {args.max_simple_death_regression:.3f}"
        )

    death_after = _metric(cand, "simple", "death_within_7_steps_after_bomb")
    prev_death_after = _metric(prev, "simple", "death_within_7_steps_after_bomb")
    if death_after > prev_death_after + args.max_death_after_bomb_regression:
        reasons.append(
            f"death_after_bomb {death_after:.2f} > previous {prev_death_after:.2f} + {args.max_death_after_bomb_regression:.2f}"
        )

    bomb_suicide = _metric(cand, "simple", "bomb_suicide_rate")
    prev_bomb_suicide = _metric(prev, "simple", "bomb_suicide_rate")
    if bomb_suicide > max(args.max_bomb_suicide_rate, prev_bomb_suicide + args.max_bomb_suicide_regression):
        reasons.append(
            f"bomb_suicide {bomb_suicide:.3f} exceeds allowed max/regression from {prev_bomb_suicide:.3f}"
        )

    bomb_rate = _metric(cand, "simple", "place_bomb_frequency")
    boxes_per_bomb = _metric(cand, "simple", "average_boxes_destroyed_per_bomb")
    if bomb_rate > 0 and boxes_per_bomb < args.min_boxes_per_bomb_when_bombing:
        reasons.append(f"boxes_per_bomb {boxes_per_bomb:.3f} < {args.min_boxes_per_bomb_when_bombing:.3f}")

    if reasons:
        bomb_improved = (
            death_after <= prev_death_after
            and boxes_per_bomb >= max(args.min_boxes_per_bomb_when_bombing, _metric(prev, "simple", "average_boxes_destroyed_per_bomb"))
            and bomb_rate >= _metric(prev, "simple", "place_bomb_frequency")
        )
        if bomb_improved and simple_death <= prev_simple_death + args.research_death_regression:
            return "research_candidate", reasons
        return "rejected", reasons
    return "accepted", ["passed_all_gates"]


def _stage_param(stage_name: str, args, default_value, escape_attr: str):
    if stage_name == "escape_only":
        value = getattr(args, escape_attr)
        if value is not None:
            return value
    return default_value


def train_stage(stage_name: str, mode: str, start_policy: Path, save_path: Path, steps: int, max_steps: int, lr: float, args) -> list[str]:
    lr = float(_stage_param(stage_name, args, lr, "escape_learning_rate"))
    clip_range = float(_stage_param(stage_name, args, args.clip_range, "escape_clip_range"))
    target_kl = float(_stage_param(stage_name, args, args.target_kl, "escape_target_kl"))
    ent_coef = float(_stage_param(stage_name, args, args.ent_coef, "escape_ent_coef"))
    retain_full_game_ratio = float(args.escape_retain_full_game_ratio if stage_name == "escape_only" else args.retain_full_game_ratio)
    mode_counts_path = args.output_dir / f"{stage_name}_mode_counts.json"
    command = [
        sys.executable,
        "-m",
        "ml.train_curriculum_ppo",
        "--mode",
        mode,
        "--start_policy",
        str(start_policy),
        "--total_timesteps",
        str(steps),
        "--n_envs",
        str(args.n_envs),
        "--max_steps",
        str(max_steps),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--n_steps",
        str(args.n_steps),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(lr),
        "--clip_range",
        str(clip_range),
        "--target_kl",
        str(target_kl),
        "--ent_coef",
        str(ent_coef),
        "--save_path",
        str(save_path),
        "--checkpoint_dir",
        str(args.output_dir / "tmp_checkpoints" / stage_name),
        "--retain_full_game_ratio",
        str(retain_full_game_ratio),
        "--mode_counts_output",
        str(mode_counts_path),
    ]
    if args.retain_policy and args.retain_kl_coef > 0:
        command.extend([
            "--retain_policy",
            str(args.retain_policy),
            "--retain_kl_coef",
            str(args.retain_kl_coef),
            "--retain_kl_samples",
            str(args.retain_kl_samples),
            "--retain_kl_batch_size",
            str(args.retain_kl_batch_size),
            "--retain_kl_train_freq",
            str(args.retain_kl_train_freq),
        ])
    if args.opponents:
        command.extend(["--opponents", *args.opponents])
    if args.mix_schedule:
        command.extend(["--mix_schedule", args.mix_schedule])
    if not args.training_bomb_gate:
        command.append("--no_training_bomb_gate")
    print("TRAIN", " ".join(command))
    subprocess.run(command, check=True, cwd=ROOT)
    return command


def copy_best(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def main():
    parser = argparse.ArgumentParser(description="Run staged bomb-escape curriculum training with normal-env gates.")
    parser.add_argument("--start_policy", default="ml/checkpoints/rl_agent_pure/ppo_bomb_selector_best.zip")
    parser.add_argument("--output_dir", type=Path, default=Path("ml/checkpoints/rl_agent_pure/curriculum"))
    parser.add_argument("--best_path", type=Path, default=Path("ml/checkpoints/rl_agent_pure/ppo_curriculum_best.zip"))
    parser.add_argument("--report_path", type=Path, default=Path("logs/curriculum_pipeline_report.json"))
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--eval_seeds", nargs="+", type=int, default=[4200, 4201, 4202])
    parser.add_argument("--max_steps_eval", type=int, default=500)
    parser.add_argument("--opponents", nargs="+", default=["random", "simple"])
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--stop_on_regression", action="store_true")
    parser.add_argument("--reduced", action="store_true", help="Use 25k/25k/25k/50k instead of the full schedule.")
    parser.add_argument("--smoke", action="store_true", help="Use tiny stage steps and eval episodes for pipeline verification.")
    parser.add_argument("--escape_steps", type=int, default=None)
    parser.add_argument("--bomb_then_escape_steps", type=int, default=None)
    parser.add_argument("--bomb_box_value_steps", type=int, default=None)
    parser.add_argument("--mixed_steps", type=int, default=None)
    parser.add_argument("--n_steps", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--clip_range", type=float, default=0.1)
    parser.add_argument("--target_kl", type=float, default=0.01)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--mix_schedule", default=None)
    parser.add_argument("--retain_full_game_ratio", type=float, default=0.0)
    parser.add_argument("--escape_retain_full_game_ratio", type=float, default=0.3)
    parser.add_argument("--escape_learning_rate", type=float, default=5e-5)
    parser.add_argument("--escape_clip_range", type=float, default=0.05)
    parser.add_argument("--escape_target_kl", type=float, default=0.005)
    parser.add_argument("--escape_ent_coef", type=float, default=0.005)
    parser.add_argument("--retain_policy", default=None)
    parser.add_argument("--retain_kl_coef", type=float, default=0.0)
    parser.add_argument("--retain_kl_samples", type=int, default=2048)
    parser.add_argument("--retain_kl_batch_size", type=int, default=256)
    parser.add_argument("--retain_kl_train_freq", type=int, default=1024)
    parser.add_argument("--training_bomb_gate", action="store_true", default=True)
    parser.add_argument("--no_training_bomb_gate", action="store_false", dest="training_bomb_gate")
    parser.add_argument("--min_random_win", type=float, default=0.95)
    parser.add_argument("--max_simple_death_regression", type=float, default=0.05)
    parser.add_argument("--research_death_regression", type=float, default=0.10)
    parser.add_argument("--max_death_after_bomb_regression", type=float, default=3.0)
    parser.add_argument("--max_bomb_suicide_regression", type=float, default=0.25)
    parser.add_argument("--max_bomb_suicide_rate", type=float, default=0.95)
    parser.add_argument("--min_boxes_per_bomb_when_bombing", type=float, default=0.5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    start_policy = Path(args.start_policy)
    if not start_policy.exists():
        raise FileNotFoundError(f"start_policy not found: {start_policy}")
    if args.retain_policy is None:
        args.retain_policy = str(start_policy)

    stage_steps = {
        "escape_only": args.escape_steps,
        "bomb_then_escape": args.bomb_then_escape_steps,
        "bomb_box_value": args.bomb_box_value_steps,
        "mixed": args.mixed_steps,
    }
    if args.smoke:
        stage_steps.update({"escape_only": 512, "bomb_then_escape": 512, "bomb_box_value": 512, "mixed": 512})
        args.eval_episodes = min(args.eval_episodes, 5)
        args.eval_seeds = args.eval_seeds[:1]
        args.n_envs = min(args.n_envs, 1)
    elif args.reduced:
        stage_steps.update({"escape_only": 25_000, "bomb_then_escape": 25_000, "bomb_box_value": 25_000, "mixed": 50_000})

    print(f"Evaluating baseline: {start_policy}")
    baseline_eval = evaluate_checkpoint(start_policy, args.opponents, args.eval_episodes, args.max_steps_eval, args.eval_seeds)
    current_best = start_policy
    previous_eval = baseline_eval
    copy_best(current_best, args.best_path)

    report = {
        "start_policy": str(start_policy),
        "best_path": str(args.best_path),
        "opponents": args.opponents,
        "eval_episodes": args.eval_episodes,
        "eval_seeds": args.eval_seeds,
        "baseline": baseline_eval,
        "stages": [],
    }

    for stage_name, mode, default_steps, default_max_steps, default_lr in STAGES:
        steps = int(stage_steps.get(stage_name) or default_steps)
        save_path = args.output_dir / f"{stage_name}.zip"
        stage_record = {
            "stage": stage_name,
            "mode": mode,
            "steps": steps,
            "start_policy": str(current_best),
            "checkpoint": str(save_path),
            "decision": "not_run",
            "reasons": [],
            "retain_full_game_ratio": args.escape_retain_full_game_ratio if stage_name == "escape_only" else args.retain_full_game_ratio,
        }
        if args.skip_existing and save_path.exists():
            stage_record["command"] = ["skip_existing"]
            print(f"SKIP existing stage checkpoint: {save_path}")
        else:
            stage_record["command"] = train_stage(stage_name, mode, current_best, save_path, steps, default_max_steps, default_lr, args)
        counts_path = args.output_dir / f"{stage_name}_mode_counts.json"
        if counts_path.exists():
            stage_record["sampled_mode_counts"] = json.loads(counts_path.read_text(encoding="utf-8"))
        print(f"Evaluating stage checkpoint: {save_path}")
        stage_eval = evaluate_checkpoint(save_path, args.opponents, args.eval_episodes, args.max_steps_eval, args.eval_seeds)
        decision, reasons = gate_candidate(stage_eval, previous_eval, args)
        stage_record["evaluation"] = stage_eval
        stage_record["decision"] = decision
        stage_record["reasons"] = reasons
        report["stages"].append(stage_record)
        args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"STAGE {stage_name}: {decision} :: {'; '.join(reasons)}")
        if decision == "accepted":
            current_best = save_path
            previous_eval = stage_eval
            copy_best(current_best, args.best_path)
        elif args.stop_on_regression:
            print("Stopping on regression as requested.")
            break

    report["selected_best"] = str(args.best_path)
    report["selected_source"] = str(current_best)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected curriculum best: {args.best_path}")
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
