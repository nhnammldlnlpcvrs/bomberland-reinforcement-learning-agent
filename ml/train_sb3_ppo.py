from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv

from agent.rl_agent_pure.constants import PPO_CONFIG
from agent.rl_agent_pure.model import BomberFeaturesExtractor
from ml.envs.bomber_gym_env import BomberGymEnv
from ml.envs.frame_stack_wrapper import FrameStackObservationWrapper
from ml.selfplay_pool import SelfPlayPool


class BombAwareEvalCallback(BaseCallback):
    def __init__(self, eval_freq, save_path, opponents, episodes, max_steps, seed, min_bomb_rate, max_death_rate, verbose=1):
        super().__init__(verbose=verbose)
        self.eval_freq = int(eval_freq)
        self.save_path = Path(save_path)
        self.opponents = opponents
        self.episodes = int(episodes)
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.min_bomb_rate = float(min_bomb_rate)
        self.max_death_rate = float(max_death_rate)
        self.best_score = -1e9

    def _on_step(self):
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        from ml.evaluate_rl_pure import evaluate

        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "candidate"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(candidate_dir / "policy.zip")
            agent_dir = ROOT / "agent" / "rl_agent_pure"
            for name in ("agent.py", "action_mask.py", "constants.py", "encoder.py", "model.py", "policy.py", "utils.py"):
                shutil.copyfile(agent_dir / name, candidate_dir / name)
            results = [
                evaluate(str(candidate_dir), opponent, self.episodes, self.max_steps, self.seed + idx * 1000)
                for idx, opponent in enumerate(self.opponents)
            ]
        bomb_rate = float(np.mean([row["place_bomb_frequency"] for row in results]))
        death_rate = float(np.mean([row["death_rate"] for row in results]))
        win_rate = float(np.mean([row["win_rate"] for row in results]))
        useful_bomb_rate = float(np.mean([
            row["useful_bomb_count"] / max(1, row["place_bomb_count"]) for row in results
        ]))
        boxes_per_bomb = float(np.mean([row["average_boxes_destroyed_per_bomb"] for row in results]))
        score = win_rate - death_rate + 0.5 * useful_bomb_rate + 0.2 * boxes_per_bomb
        if self.verbose:
            print(
                "bomb_eval "
                f"steps={self.num_timesteps} score={score:.3f} win={win_rate:.3f} "
                f"death={death_rate:.3f} bomb_rate={bomb_rate:.4f} "
                f"useful={useful_bomb_rate:.3f} boxes_per_bomb={boxes_per_bomb:.3f}"
            )
        if bomb_rate >= self.min_bomb_rate and death_rate <= self.max_death_rate and score > self.best_score:
            self.best_score = score
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.save_path)
            if self.verbose:
                print(f"Saved bomb-aware checkpoint: {self.save_path}")
        return True


class BCAuxiliaryCallback(BaseCallback):
    def __init__(self, dataset, coef=0.0, batch_size=256, train_freq=2048, bomb_multiplier=2.0, escape_multiplier=6.0, verbose=0):
        super().__init__(verbose=verbose)
        self.dataset = dataset
        self.coef = float(coef)
        self.batch_size = int(batch_size)
        self.train_freq = int(train_freq)
        self.bomb_multiplier = float(bomb_multiplier)
        self.escape_multiplier = float(escape_multiplier)
        self.obs = None
        self.actions = None
        self.weights = None

    def _on_training_start(self):
        if not self.dataset or self.coef <= 0:
            return
        data = np.load(self.dataset)
        self.obs = data["observations"].astype(np.float32)
        self.actions = data["actions"].astype(np.int64)
        weights = data["sample_weight"].astype(np.float32) if "sample_weight" in data else np.ones(len(self.actions), dtype=np.float32)
        weights = weights.copy()
        weights[self.actions == 5] *= self.bomb_multiplier
        if "is_post_bomb_escape" in data:
            weights[data["is_post_bomb_escape"].astype(bool)] *= self.escape_multiplier
        elif "post_bomb_escape" in data:
            weights[data["post_bomb_escape"].astype(bool)] *= self.escape_multiplier
        self.weights = weights / max(1e-6, weights.mean())

    def _policy_logits(self, observations):
        policy = self.model.policy
        features = policy.extract_features(observations)
        if isinstance(features, tuple):
            features = features[0]
        latent_pi, _latent_vf = policy.mlp_extractor(features)
        return policy.action_net(latent_pi)

    def _on_step(self):
        if self.obs is None or self.train_freq <= 0 or self.n_calls % self.train_freq != 0:
            return True
        idx = np.random.choice(len(self.actions), size=self.batch_size, replace=True, p=self.weights / self.weights.sum())
        device = self.model.policy.device
        obs = torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.actions[idx], dtype=torch.long, device=device)
        logits = self._policy_logits(obs)
        loss = F.cross_entropy(logits, actions) * self.coef
        self.model.policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 0.5)
        self.model.policy.optimizer.step()
        if self.verbose:
            print(f"bc_aux_loss={float(loss.item()):.4f}")
        return True


class BombLogitRegularizationCallback(BaseCallback):
    def __init__(
        self,
        teacher_path,
        dataset,
        coef=0.0,
        batch_size=256,
        train_freq=1024,
        min_teacher_prob=0.01,
        max_samples=4096,
        verbose=0,
    ):
        super().__init__(verbose=verbose)
        self.teacher_path = teacher_path
        self.dataset = dataset
        self.coef = float(coef)
        self.batch_size = int(batch_size)
        self.train_freq = int(train_freq)
        self.min_teacher_prob = float(min_teacher_prob)
        self.max_samples = int(max_samples)
        self.obs = None
        self.teacher_bomb_logits = None

    def _policy_logits(self, policy, observations):
        features = policy.extract_features(observations)
        if isinstance(features, tuple):
            features = features[0]
        latent_pi, _latent_vf = policy.mlp_extractor(features)
        return policy.action_net(latent_pi)

    def _on_training_start(self):
        if not self.teacher_path or not self.dataset or self.coef <= 0:
            return
        data = np.load(self.dataset)
        observations = data["observations"].astype(np.float32)
        teacher = PPO.load(self.teacher_path, device=self.model.policy.device)
        teacher.policy.eval()
        device = self.model.policy.device
        probs = []
        logits = []
        with torch.no_grad():
            for start in range(0, len(observations), 1024):
                obs = torch.as_tensor(observations[start:start + 1024], dtype=torch.float32, device=device)
                batch_logits = self._policy_logits(teacher.policy, obs)
                batch_probs = torch.softmax(batch_logits, dim=1)[:, 5]
                probs.append(batch_probs.detach().cpu().numpy())
                logits.append(batch_logits[:, 5].detach().cpu().numpy())
        probs = np.concatenate(probs)
        logits = np.concatenate(logits)
        selected = probs >= self.min_teacher_prob
        if not np.any(selected):
            keep = min(self.max_samples, len(probs))
            selected_idx = np.argsort(probs)[-keep:]
        else:
            selected_idx = np.flatnonzero(selected)
            if len(selected_idx) > self.max_samples:
                top = np.argsort(probs[selected_idx])[-self.max_samples:]
                selected_idx = selected_idx[top]
        self.obs = observations[selected_idx]
        self.teacher_bomb_logits = logits[selected_idx].astype(np.float32)
        if self.verbose:
            print(
                "bomb_logit_regularizer "
                f"samples={len(selected_idx)} mean_teacher_prob={float(probs[selected_idx].mean()):.5f}"
            )

    def _on_step(self):
        if self.obs is None or self.train_freq <= 0 or self.n_calls % self.train_freq != 0:
            return True
        idx = np.random.choice(len(self.obs), size=min(self.batch_size, len(self.obs)), replace=True)
        device = self.model.policy.device
        obs = torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device)
        target = torch.as_tensor(self.teacher_bomb_logits[idx], dtype=torch.float32, device=device)
        logits = self._policy_logits(self.model.policy, obs)[:, 5]
        loss = F.mse_loss(logits, target) * self.coef
        self.model.policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 0.5)
        self.model.policy.optimizer.step()
        if self.verbose:
            print(f"bomb_logit_reg_loss={float(loss.item()):.6f}")
        return True


def make_env(args, rank):
    custom_opponents = args.opponents or None
    pool = SelfPlayPool(args.checkpoint_dir, args.stage, seed=args.seed + rank, opponents=custom_opponents).opponents()
    env = BomberGymEnv(
        agent_id=args.agent_id,
        opponent_pool=pool,
        max_steps=args.max_steps,
        seed=args.seed + rank,
        training_bomb_gate=args.training_bomb_gate,
    )
    if args.frame_stack > 1:
        env = FrameStackObservationWrapper(env, args.frame_stack)
    return Monitor(env)


def main():
    parser = argparse.ArgumentParser(description="Train pure RL Bomberland PPO with SB3.")
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3", "stage4"], default="stage1")
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--checkpoint_dir", default="ml/checkpoints/rl_pure")
    parser.add_argument("--archive_dir", default="ml/checkpoints/rl_agent_pure")
    parser.add_argument("--tensorboard_log", default="logs/tensorboard/rl_pure")
    parser.add_argument("--save_path", default="agent/rl_agent_pure/policy.zip")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume_bc", default=None, help="Load a BC-pretrained PPO zip before PPO fine-tuning.")
    parser.add_argument("--opponents", nargs="*", default=None, help="Override opponent pool, e.g. random simple checkpoints.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_freq", type=int, default=25_000)
    parser.add_argument("--checkpoint_freq", type=int, default=50_000)
    parser.add_argument("--progress_bar", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--target_kl", type=float, default=None)
    parser.add_argument("--ent_coef", type=float, default=None)
    parser.add_argument("--clip_range", type=float, default=None)
    parser.add_argument("--min_bomb_rate", type=float, default=0.0)
    parser.add_argument("--max_death_rate", type=float, default=1.0)
    parser.add_argument("--bomb_eval_episodes", type=int, default=10)
    parser.add_argument("--bomb_eval_opponents", nargs="*", default=None)
    parser.add_argument("--bomb_aware_save_path", default="ml/checkpoints/rl_agent_pure/best_bomb_aware.zip")
    parser.add_argument("--bc_dataset", default=None)
    parser.add_argument("--bc_coef", type=float, default=0.0)
    parser.add_argument("--bc_train_freq", type=int, default=2048)
    parser.add_argument("--bc_bomb_multiplier", type=float, default=2.0)
    parser.add_argument("--bc_escape_multiplier", type=float, default=6.0)
    parser.add_argument("--bomb_kl_teacher", default=None, help="Teacher PPO zip for PLACE_BOMB logit preservation.")
    parser.add_argument("--bomb_kl_dataset", default=None, help="Observation dataset used for PLACE_BOMB logit preservation.")
    parser.add_argument("--bomb_kl_coef", type=float, default=0.0)
    parser.add_argument("--bomb_kl_train_freq", type=int, default=1024)
    parser.add_argument("--bomb_kl_batch_size", type=int, default=256)
    parser.add_argument("--bomb_kl_min_teacher_prob", type=float, default=0.01)
    parser.add_argument("--bomb_kl_max_samples", type=int, default=4096)
    parser.add_argument("--training_bomb_gate", action="store_true", help="Training-only gate for clearly self-destructive or valueless bomb actions.")
    parser.add_argument("--frame_stack", type=int, default=1, help="Stack last K encoded observations as [K*C, 13, 13].")
    args = parser.parse_args()
    if args.frame_stack > 1 and args.archive_dir == "ml/checkpoints/rl_agent_pure":
        args.archive_dir = "ml/checkpoints/rl_agent_temporal"

    env = DummyVecEnv([lambda rank=rank: make_env(args, rank) for rank in range(args.n_envs)])
    eval_env_raw = BomberGymEnv(
        agent_id=args.agent_id,
        opponent_pool=SelfPlayPool(args.checkpoint_dir, args.stage, seed=args.seed + 999, opponents=args.opponents).opponents(),
        max_steps=args.max_steps,
        seed=args.seed + 999,
        training_bomb_gate=False,
    )
    if args.frame_stack > 1:
        eval_env_raw = FrameStackObservationWrapper(eval_env_raw, args.frame_stack)
    eval_env = Monitor(eval_env_raw)
    policy_kwargs = {
        "features_extractor_class": BomberFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    ppo_config = dict(PPO_CONFIG)
    if args.n_steps is not None:
        ppo_config["n_steps"] = args.n_steps
    if args.batch_size is not None:
        ppo_config["batch_size"] = args.batch_size
    for key in ("learning_rate", "target_kl", "ent_coef", "clip_range"):
        value = getattr(args, key)
        if value is not None:
            ppo_config[key] = value
    resume_path = args.resume_bc or args.resume
    if resume_path:
        model = PPO.load(
            resume_path,
            env=env,
            device=args.device,
            custom_objects={"policy_kwargs": policy_kwargs},
        )
        model.learning_rate = ppo_config["learning_rate"]
        model.lr_schedule = get_schedule_fn(ppo_config["learning_rate"])
        model.ent_coef = ppo_config["ent_coef"]
        model.clip_range = get_schedule_fn(ppo_config["clip_range"])
        model.target_kl = ppo_config.get("target_kl")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=args.tensorboard_log,
            seed=args.seed,
            verbose=1,
            device=args.device,
            **ppo_config,
        )

    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // max(1, args.n_envs)),
            save_path=args.checkpoint_dir,
            name_prefix="ppo_rl_pure",
        ))
    if args.eval_freq > 0:
        callbacks.append(EvalCallback(
            eval_env,
            best_model_save_path=args.checkpoint_dir,
            log_path="logs/rl_pure_eval",
            eval_freq=max(1, args.eval_freq // max(1, args.n_envs)),
            deterministic=True,
        ))
    if args.min_bomb_rate > 0:
        callbacks.append(BombAwareEvalCallback(
            eval_freq=max(1, args.eval_freq // max(1, args.n_envs)),
            save_path=args.bomb_aware_save_path,
            opponents=args.bomb_eval_opponents or args.opponents or ["random", "simple"],
            episodes=args.bomb_eval_episodes,
            max_steps=args.max_steps,
            seed=args.seed + 5000,
            min_bomb_rate=args.min_bomb_rate,
            max_death_rate=args.max_death_rate,
        ))
    if args.bc_dataset and args.bc_coef > 0:
        callbacks.append(BCAuxiliaryCallback(
            dataset=args.bc_dataset,
            coef=args.bc_coef,
            batch_size=args.batch_size or 256,
            train_freq=args.bc_train_freq,
            bomb_multiplier=args.bc_bomb_multiplier,
            escape_multiplier=args.bc_escape_multiplier,
            verbose=0,
        ))
    if args.bomb_kl_teacher and args.bomb_kl_dataset and args.bomb_kl_coef > 0:
        callbacks.append(BombLogitRegularizationCallback(
            teacher_path=args.bomb_kl_teacher,
            dataset=args.bomb_kl_dataset,
            coef=args.bomb_kl_coef,
            batch_size=args.bomb_kl_batch_size,
            train_freq=args.bomb_kl_train_freq,
            min_teacher_prob=args.bomb_kl_min_teacher_prob,
            max_samples=args.bomb_kl_max_samples,
            verbose=1,
        ))
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=args.progress_bar)
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.save_path)
    archive_dir = Path(args.archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"ppo_{args.stage}_seed{args.seed}_steps{args.total_timesteps}.zip"
    shutil.copyfile(args.save_path, archive_path)
    print(f"Saved policy: {args.save_path}")
    print(f"Archived checkpoint: {archive_path}")


if __name__ == "__main__":
    main()
