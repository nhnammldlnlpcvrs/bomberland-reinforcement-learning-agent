from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from sb3_contrib import RecurrentPPO
except ImportError as exc:  # pragma: no cover - dependency-gated script
    raise SystemExit(
        "sb3-contrib is required for RecurrentPPO. Install requirements.txt first "
        "or run: pip install sb3-contrib"
    ) from exc

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv

from agent.rl_agent_pure.model import BomberFeaturesExtractor
from ml.envs.bomber_gym_env import BomberGymEnv
from ml.selfplay_pool import SelfPlayPool


def make_env(args, rank: int):
    pool = SelfPlayPool(
        args.checkpoint_dir,
        args.stage,
        seed=args.seed + rank,
        opponents=args.opponents or None,
    ).opponents()
    return Monitor(BomberGymEnv(
        agent_id=args.agent_id,
        opponent_pool=pool,
        max_steps=args.max_steps,
        seed=args.seed + rank,
        training_bomb_gate=args.training_bomb_gate,
    ))


def main():
    parser = argparse.ArgumentParser(description="Train research-only Bomberland RecurrentPPO.")
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3", "stage4"], default="stage1")
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=9700)
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--checkpoint_dir", default="ml/checkpoints/rl_agent_recurrent")
    parser.add_argument("--tensorboard_log", default="logs/tensorboard/rl_recurrent")
    parser.add_argument("--save_path", default="ml/checkpoints/rl_agent_recurrent/recurrent_ppo_stage1.zip")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--opponents", nargs="*", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.02)
    parser.add_argument("--clip_range", type=float, default=0.1)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=None)
    parser.add_argument("--checkpoint_freq", type=int, default=25_000)
    parser.add_argument("--training_bomb_gate", action="store_true")
    parser.add_argument("--progress_bar", action="store_true")
    args = parser.parse_args()

    env = DummyVecEnv([lambda rank=rank: make_env(args, rank) for rank in range(args.n_envs)])
    policy_kwargs = {
        "features_extractor_class": BomberFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    config = {
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "ent_coef": args.ent_coef,
        "clip_range": args.clip_range,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "target_kl": args.target_kl,
    }
    if args.resume:
        model = RecurrentPPO.load(args.resume, env=env, device=args.device, custom_objects={"policy_kwargs": policy_kwargs})
        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
        model.ent_coef = args.ent_coef
        model.clip_range = get_schedule_fn(args.clip_range)
        model.target_kl = args.target_kl
    else:
        model = RecurrentPPO(
            "CnnLstmPolicy",
            env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=args.tensorboard_log,
            seed=args.seed,
            verbose=1,
            device=args.device,
            **config,
        )

    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // max(1, args.n_envs)),
            save_path=args.checkpoint_dir,
            name_prefix="recurrent_ppo",
        ))
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=args.progress_bar)
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.save_path)
    archive_path = Path(args.checkpoint_dir) / f"recurrent_ppo_{args.stage}_seed{args.seed}_steps{args.total_timesteps}.zip"
    if Path(args.save_path).resolve() != archive_path.resolve():
        shutil.copyfile(args.save_path, archive_path)
    print(f"Saved recurrent policy: {args.save_path}")
    print(f"Archived checkpoint: {archive_path}")


if __name__ == "__main__":
    main()
