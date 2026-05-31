from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from ml.train_bomberland_strong.checkpoint_gate import update_best_files


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def latest_checkpoint(save_dir: Path) -> Path | None:
    latest = save_dir / "latest.zip"
    if latest.exists():
        return latest
    checkpoints = sorted((save_dir / "checkpoints").glob("*.zip")) if (save_dir / "checkpoints").exists() else []
    return checkpoints[-1] if checkpoints else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle-friendly chunked launcher for Bomberland strong PPO.")
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3"], default="stage1")
    parser.add_argument("--resume", default="ml/checkpoints/rl_agent_temporal/ppo_framestack4_lr1e4_ent002_100k.zip")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--chunk_timesteps", type=int, default=100_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save_dir", default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--eval_opponents", nargs="+", default=["random", "simple"])
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--clip_range", type=float, default=0.1)
    parser.add_argument("--target_kl", type=float, default=0.01)
    args = parser.parse_args()

    default_root = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("ml/checkpoints")
    save_dir = Path(args.save_dir) if args.save_dir else default_root / f"bomberland_strong_{args.stage}"
    save_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = save_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    env_info = {
        "platform": platform.platform(),
        "python": sys.version,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": args.device,
        "cwd": os.getcwd(),
    }
    (logs_dir / "environment.json").write_text(json.dumps(env_info, indent=2), encoding="utf-8")
    print(json.dumps(env_info, indent=2))

    completed = 0
    resume = latest_checkpoint(save_dir) or (Path(args.resume) if args.resume else None)
    csv_path = logs_dir / "chunks.csv"
    while completed < args.total_timesteps:
        chunk = min(args.chunk_timesteps, args.total_timesteps - completed)
        train_cmd = [
            sys.executable, "-m", "ml.train_bomberland_strong.train",
            "--stage", args.stage,
            "--total_timesteps", str(chunk),
            "--n_envs", str(args.n_envs),
            "--seed", str(args.seed + completed),
            "--max_steps", str(args.max_steps),
            "--frame_stack", str(args.frame_stack),
            "--device", args.device,
            "--save_dir", str(save_dir),
            "--learning_rate", str(args.learning_rate),
            "--ent_coef", str(args.ent_coef),
            "--clip_range", str(args.clip_range),
            "--target_kl", str(args.target_kl),
        ]
        if resume and resume.exists():
            train_cmd.extend(["--resume", str(resume)])
        run(train_cmd)
        completed += chunk
        candidate = save_dir / "latest.zip"
        eval_json = logs_dir / f"eval_after_{completed}.json"
        eval_cmd = [
            sys.executable, "-m", "ml.train_bomberland_strong.evaluate",
            "--agent_path", str(candidate),
            "--frame_stack", str(args.frame_stack),
            "--opponents", *args.eval_opponents,
            "--episodes", str(args.eval_episodes),
            "--max_steps", str(args.max_steps),
            "--seed", str(args.seed + completed),
            "--output", str(eval_json),
        ]
        run(eval_cmd)
        results = json.loads(eval_json.read_text(encoding="utf-8"))
        decision = update_best_files(str(candidate), results, str(save_dir))
        (logs_dir / f"gate_after_{completed}.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["timesteps", "accepted", "score", "reasons"])
            if fh.tell() == 0:
                writer.writeheader()
            writer.writerow({
                "timesteps": completed,
                "accepted": decision["accepted"],
                "score": decision["score"],
                "reasons": ";".join(decision["reasons"]),
            })
        resume = candidate
    if not (save_dir / "best_overall.zip").exists() and (save_dir / "latest.zip").exists():
        shutil.copy2(save_dir / "latest.zip", save_dir / "latest_unaccepted.zip")


if __name__ == "__main__":
    main()
