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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.model import BomberFeaturesExtractor
from ml.envs.bomber_curriculum_env import CURRICULUM_MODES, BomberCurriculumEnv
from ml.envs.frame_stack_wrapper import FrameStackObservationWrapper
from ml.envs.bomber_gym_env import BomberGymEnv
from ml.selfplay_pool import SelfPlayPool


class FullGameRetainKLCallback(BaseCallback):
    """Small training-only KL penalty to preserve the baseline policy on normal states."""

    def __init__(
        self,
        teacher_path: str,
        coef: float,
        agent_id: int,
        seed: int,
        max_steps: int,
        sample_count: int = 2048,
        batch_size: int = 256,
        train_freq: int = 1024,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.teacher_path = teacher_path
        self.coef = float(coef)
        self.agent_id = int(agent_id)
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.sample_count = int(sample_count)
        self.batch_size = int(batch_size)
        self.train_freq = int(train_freq)
        self.obs = None
        self.teacher_probs = None
        self.teacher = None

    def _policy_logits(self, policy, observations):
        features = policy.extract_features(observations)
        if isinstance(features, tuple):
            features = features[0]
        latent_pi, _latent_vf = policy.mlp_extractor(features)
        return policy.action_net(latent_pi)

    def _on_training_start(self):
        if not self.teacher_path or self.coef <= 0:
            return
        self.teacher = PPO.load(self.teacher_path, device=self.model.policy.device)
        retain_env = BomberCurriculumEnv(
            mode="full_game_mix",
            agent_id=self.agent_id,
            opponent_pool=["random", "simple"],
            max_steps=self.max_steps,
            seed=self.seed,
            training_bomb_gate=False,
        )
        observations = []
        obs, _info = retain_env.reset(seed=self.seed)
        while len(observations) < self.sample_count:
            observations.append(obs.astype(np.float32))
            action, _state = self.teacher.predict(obs[None, ...], deterministic=True)
            obs, _reward, terminated, truncated, _info = retain_env.step(int(np.asarray(action).reshape(-1)[0]))
            if terminated or truncated:
                obs, _info = retain_env.reset(seed=self.seed + len(observations))
        self.obs = np.asarray(observations, dtype=np.float32)
        probs = []
        device = self.model.policy.device
        self.teacher.policy.eval()
        with torch.no_grad():
            for start in range(0, len(self.obs), 512):
                batch = torch.as_tensor(self.obs[start:start + 512], dtype=torch.float32, device=device)
                logits = self._policy_logits(self.teacher.policy, batch)
                probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        self.teacher_probs = np.concatenate(probs, axis=0).astype(np.float32)
        if self.verbose:
            print(f"retain_kl_samples={len(self.obs)} coef={self.coef}")

    def _on_step(self):
        if self.obs is None or self.teacher_probs is None or self.train_freq <= 0 or self.n_calls % self.train_freq != 0:
            return True
        idx = np.random.choice(len(self.obs), size=min(self.batch_size, len(self.obs)), replace=True)
        device = self.model.policy.device
        obs = torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device)
        teacher_probs = torch.as_tensor(self.teacher_probs[idx], dtype=torch.float32, device=device)
        logits = self._policy_logits(self.model.policy, obs)
        log_probs = torch.log_softmax(logits, dim=1)
        teacher_log_probs = torch.log(torch.clamp(teacher_probs, min=1e-8))
        loss = (teacher_probs * (teacher_log_probs - log_probs)).sum(dim=1).mean() * self.coef
        self.model.policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 0.5)
        self.model.policy.optimizer.step()
        if self.verbose:
            print(f"retain_kl_loss={float(loss.item()):.6f}")
        return True


def make_curriculum_env(args, rank: int):
    pool = SelfPlayPool(args.checkpoint_dir, args.stage, seed=args.seed + rank, opponents=args.opponents).opponents()
    env = BomberCurriculumEnv(
        mode=args.mode,
        agent_id=args.agent_id,
        opponent_pool=pool,
        max_steps=args.max_steps,
        seed=args.seed + rank,
        max_reset_attempts=args.max_reset_attempts,
        mix_schedule=args.mix_schedule,
        retain_full_game_ratio=args.retain_full_game_ratio,
        training_bomb_gate=args.training_bomb_gate,
    )
    if args.frame_stack > 1:
        env = FrameStackObservationWrapper(env, args.frame_stack)
    return Monitor(env)


def main():
    parser = argparse.ArgumentParser(description="Train pure RL PPO on bomb-escape curriculum scenarios.")
    parser.add_argument("--mode", choices=CURRICULUM_MODES, default="escape_only")
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3", "stage4"], default="stage1")
    parser.add_argument("--start_policy", default=None, help="Optional PPO zip to continue from.")
    parser.add_argument("--total_timesteps", type=int, default=50_000)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--max_reset_attempts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save_path", default="ml/checkpoints/rl_agent_pure/ppo_curriculum.zip")
    parser.add_argument("--archive_dir", default="ml/checkpoints/rl_agent_pure")
    parser.add_argument("--checkpoint_dir", default="ml/checkpoints/rl_agent_pure/curriculum_tmp")
    parser.add_argument("--tensorboard_log", default="logs/tensorboard/rl_curriculum")
    parser.add_argument("--opponents", nargs="*", default=["random", "simple"])
    parser.add_argument("--mix_schedule", default=None, help="For mixed mode, e.g. escape_only:0.25,bomb_then_escape:0.30,bomb_box_value:0.35,full_game_mix:0.10")
    parser.add_argument("--retain_full_game_ratio", type=float, default=0.0, help="For a non-mixed curriculum mode, sample full_game_mix episodes with this probability.")
    parser.add_argument("--retain_policy", default=None, help="Optional PPO zip used as teacher for full-game KL retain regularization.")
    parser.add_argument("--retain_kl_coef", type=float, default=0.0)
    parser.add_argument("--retain_kl_samples", type=int, default=2048)
    parser.add_argument("--retain_kl_batch_size", type=int, default=256)
    parser.add_argument("--retain_kl_train_freq", type=int, default=1024)
    parser.add_argument("--mode_counts_output", default=None)
    parser.add_argument("--training_bomb_gate", action="store_true", default=True)
    parser.add_argument("--no_training_bomb_gate", action="store_false", dest="training_bomb_gate")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--n_steps", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.1)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=0.01)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--checkpoint_freq", type=int, default=25_000)
    parser.add_argument("--check_env", action="store_true")
    parser.add_argument("--progress_bar", action="store_true")
    parser.add_argument("--frame_stack", type=int, default=1)
    args = parser.parse_args()
    if args.frame_stack > 1 and args.archive_dir == "ml/checkpoints/rl_agent_pure":
        args.archive_dir = "ml/checkpoints/rl_agent_temporal"

    if args.check_env:
        check_env(BomberCurriculumEnv(mode=args.mode, agent_id=args.agent_id, max_steps=args.max_steps, seed=args.seed), warn=True)

    env = DummyVecEnv([lambda rank=rank: make_curriculum_env(args, rank) for rank in range(args.n_envs)])
    eval_env_raw = BomberCurriculumEnv(
        mode=args.mode,
        agent_id=args.agent_id,
        opponent_pool=SelfPlayPool(args.checkpoint_dir, args.stage, seed=args.seed + 999, opponents=args.opponents).opponents(),
        max_steps=args.max_steps,
        seed=args.seed + 999,
        max_reset_attempts=args.max_reset_attempts,
        mix_schedule=args.mix_schedule,
        retain_full_game_ratio=args.retain_full_game_ratio,
        training_bomb_gate=args.training_bomb_gate,
    )
    if args.frame_stack > 1:
        eval_env_raw = FrameStackObservationWrapper(eval_env_raw, args.frame_stack)
    eval_env = Monitor(eval_env_raw)
    policy_kwargs = {
        "features_extractor_class": BomberFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    ppo_kwargs = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "target_kl": args.target_kl,
    }

    if args.start_policy:
        model = PPO.load(args.start_policy, env=env, device=args.device, custom_objects={"policy_kwargs": policy_kwargs})
        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
        model.clip_range = get_schedule_fn(args.clip_range)
        model.ent_coef = args.ent_coef
        model.target_kl = args.target_kl
    else:
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=args.tensorboard_log,
            seed=args.seed,
            verbose=1,
            device=args.device,
            **ppo_kwargs,
        )

    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // max(1, args.n_envs)),
            save_path=args.checkpoint_dir,
            name_prefix=f"ppo_curriculum_{args.mode}",
        ))
    if args.eval_freq > 0:
        callbacks.append(EvalCallback(
            eval_env,
            best_model_save_path=args.checkpoint_dir,
            log_path="logs/rl_curriculum_eval",
            eval_freq=max(1, args.eval_freq // max(1, args.n_envs)),
            deterministic=True,
        ))
    if args.retain_policy and args.retain_kl_coef > 0:
        callbacks.append(FullGameRetainKLCallback(
            teacher_path=args.retain_policy,
            coef=args.retain_kl_coef,
            agent_id=args.agent_id,
            seed=args.seed + 7000,
            max_steps=args.max_steps,
            sample_count=args.retain_kl_samples,
            batch_size=args.retain_kl_batch_size,
            train_freq=args.retain_kl_train_freq,
            verbose=1,
        ))
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=args.progress_bar)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    archive_dir = Path(args.archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"ppo_curriculum_{args.mode}_seed{args.seed}_steps{args.total_timesteps}.zip"
    if archive_path.resolve() != save_path.resolve():
        shutil.copyfile(save_path, archive_path)
    mode_counts = {}
    for idx, wrapped in enumerate(env.envs):
        inner = getattr(wrapped, "env", wrapped)
        counts = getattr(inner, "mode_counts", None)
        if counts:
            for key, value in counts.items():
                mode_counts[key] = mode_counts.get(key, 0) + int(value)
    if args.mode_counts_output:
        output_path = Path(args.mode_counts_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(mode_counts, indent=2), encoding="utf-8")
    print(f"Saved curriculum policy: {save_path}")
    print(f"Archived checkpoint: {archive_path}")
    print(f"Sampled mode counts: {mode_counts}")
    print("Evaluate in the normal full-game env with ml.evaluate_rl_pure before considering this checkpoint useful.")


if __name__ == "__main__":
    main()
