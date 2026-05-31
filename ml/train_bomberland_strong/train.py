from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv

from agent.rl_agent_pure.model import BomberFeaturesExtractor
from ml.envs.bomber_gym_env import BomberGymEnv
from ml.envs.frame_stack_wrapper import FrameStackObservationWrapper
from ml.train_bomberland_strong.selfplay_pool import StrongSelfPlayPool


class ManifestCallback(BaseCallback):
    def __init__(self, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        manifest = {
            "num_timesteps": int(self.num_timesteps),
            "save_dir": str(self.save_dir),
        }
        (self.save_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def make_env(args, rank: int):
    pool = StrongSelfPlayPool(args.save_dir, args.stage, seed=args.seed + rank, extra_opponents=args.opponents).opponents()
    env = BomberGymEnv(
        agent_id=0,
        opponent_pool=pool,
        max_steps=args.max_steps,
        seed=args.seed + rank,
        training_bomb_gate=args.training_bomb_gate,
    )
    if args.frame_stack > 1:
        env = FrameStackObservationWrapper(env, args.frame_stack)
    return Monitor(env)


def build_model(args, env):
    policy_kwargs = {
        "features_extractor_class": BomberFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    if args.resume:
        model = PPO.load(
            args.resume,
            env=env,
            device=args.device,
            custom_objects={"policy_kwargs": policy_kwargs},
        )
        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
        model.ent_coef = args.ent_coef
        model.clip_range = get_schedule_fn(args.clip_range)
        model.target_kl = args.target_kl
        return model
    return PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        tensorboard_log=str(Path(args.save_dir) / "tensorboard"),
        seed=args.seed,
        verbose=1,
        device=args.device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Strong Kaggle Bomberland PPO training entrypoint.")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default="")
    parser.add_argument("--save_dir", default="ml/checkpoints/bomberland_strong")
    parser.add_argument("--checkpoint_freq", type=int, default=50_000)
    parser.add_argument("--eval_freq", type=int, default=50_000, help="Reserved for launcher-level eval/gating.")
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3"], default="stage1")
    parser.add_argument("--opponents", nargs="*", default=None)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--clip_range", type=float, default=0.1)
    parser.add_argument("--target_kl", type=float, default=0.01)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--training_bomb_gate", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "python": sys.version,
        "platform": platform.platform(),
        "stage": args.stage,
        "save_dir": str(save_dir),
        "resume": args.resume,
        "frame_stack": args.frame_stack,
        "n_envs": args.n_envs,
    }, indent=2))

    env = DummyVecEnv([lambda rank=rank: make_env(args, rank) for rank in range(args.n_envs)])
    model = build_model(args, env)
    callbacks = [ManifestCallback(str(save_dir))]
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // max(1, args.n_envs)),
            save_path=str(save_dir / "checkpoints"),
            name_prefix=f"ppo_{args.stage}",
        ))
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    latest = save_dir / "latest.zip"
    model.save(latest)
    stage_path = save_dir / f"{args.stage}_seed{args.seed}_steps{args.total_timesteps}.zip"
    shutil.copy2(latest, stage_path)
    print(f"Saved latest: {latest}")
    print(f"Saved stage checkpoint: {stage_path}")


if __name__ == "__main__":
    main()
