from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from sb3_contrib import RecurrentPPO
except ImportError as exc:  # pragma: no cover
    raise SystemExit("sb3-contrib is required. Install requirements.txt first.") from exc

from agent.rl_agent_pure.constants import NUM_ACTIONS, PLACE_BOMB
from agent.rl_agent_pure.model import BomberFeaturesExtractor
from ml.envs.bomber_gym_env import BomberGymEnv


def _make_env(seed: int):
    return Monitor(BomberGymEnv(agent_id=0, opponent_pool=["random", "simple"], max_steps=500, seed=seed))


def _make_model(args):
    env = DummyVecEnv([lambda: _make_env(args.seed)])
    policy_kwargs = {
        "features_extractor_class": BomberFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    if args.base_policy and Path(args.base_policy).exists():
        return RecurrentPPO.load(args.base_policy, env=env, device=args.device, custom_objects={"policy_kwargs": policy_kwargs})
    return RecurrentPPO(
        "CnnLstmPolicy",
        env,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        verbose=0,
        device=args.device,
        n_steps=512,
        batch_size=256,
        n_epochs=4,
        learning_rate=args.learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        clip_range=0.1,
        max_grad_norm=0.5,
    )


def _initial_lstm_state(policy, n_seq: int, device):
    shape = policy.lstm_hidden_state_shape
    h = torch.zeros((shape[0], n_seq, shape[2]), dtype=torch.float32, device=device)
    c = torch.zeros((shape[0], n_seq, shape[2]), dtype=torch.float32, device=device)
    return (h, c)


def _distribution(policy, obs, episode_starts, n_seq: int):
    states = _initial_lstm_state(policy, n_seq, obs.device)
    return policy.get_distribution(obs, states, episode_starts)[0]


def _batch_loss(policy, obs_seq, actions_seq, mask_seq, weights_seq, episode_start_seq=None, class_weights=None):
    n_seq, seq_len = actions_seq.shape
    # sb3-contrib recurrent policies expect flattened data in time-major order:
    # [t0_seq0, t0_seq1, ..., t1_seq0, t1_seq1, ...].  Batch-major flattening
    # silently scrambles hidden-state flow and prevents BC from fitting.
    obs = obs_seq.transpose(0, 1).reshape(n_seq * seq_len, *obs_seq.shape[2:])
    actions = actions_seq.transpose(0, 1).reshape(n_seq * seq_len)
    mask = mask_seq.transpose(0, 1).reshape(n_seq * seq_len)
    weights = weights_seq.transpose(0, 1).reshape(n_seq * seq_len)
    if class_weights is not None:
        weights = weights * class_weights[actions]
    if episode_start_seq is None:
        episode_starts = torch.zeros(n_seq, seq_len, dtype=torch.float32, device=obs.device)
        episode_starts[:, 0] = 1.0
    else:
        episode_starts = episode_start_seq.float()
    dist = _distribution(policy, obs, episode_starts.transpose(0, 1).reshape(-1), n_seq)
    log_prob = dist.log_prob(actions)
    loss = -(log_prob * weights * mask).sum() / torch.clamp((weights * mask).sum(), min=1.0)
    pred = dist.distribution.probs.argmax(dim=1)
    return loss, pred.detach(), actions.detach(), mask.detach()


def _metrics(policy, observations, actions, mask, weights, episode_starts=None, batch_size=16, device="cpu"):
    policy.eval()
    counts = np.zeros((NUM_ACTIONS, NUM_ACTIONS), dtype=np.int64)
    total = correct = 0
    pred_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(actions), batch_size):
            obs = torch.as_tensor(observations[start:start + batch_size], dtype=torch.float32, device=device)
            act = torch.as_tensor(actions[start:start + batch_size], dtype=torch.long, device=device)
            msk = torch.as_tensor(mask[start:start + batch_size], dtype=torch.float32, device=device)
            w = torch.as_tensor(weights[start:start + batch_size], dtype=torch.float32, device=device)
            starts = None
            if episode_starts is not None:
                starts = torch.as_tensor(episode_starts[start:start + batch_size], dtype=torch.float32, device=device)
            _loss, pred, true, flat_mask = _batch_loss(policy, obs, act, msk, w, episode_start_seq=starts)
            pred_np = pred.cpu().numpy()
            true_np = true.cpu().numpy()
            mask_np = flat_mask.cpu().numpy().astype(bool)
            for t, p in zip(true_np[mask_np], pred_np[mask_np]):
                counts[int(t), int(p)] += 1
            total += int(mask_np.sum())
            correct += int((pred_np[mask_np] == true_np[mask_np]).sum())
            pred_counts += np.bincount(pred_np[mask_np], minlength=NUM_ACTIONS)
    per_action = {}
    for action in range(NUM_ACTIONS):
        per_action[str(action)] = float(counts[action, action] / max(1, counts[action].sum()))
    pred_bomb = counts[:, PLACE_BOMB].sum()
    true_bomb = counts[PLACE_BOMB].sum()
    return {
        "accuracy": float(correct / max(1, total)),
        "per_action_accuracy": per_action,
        "place_bomb_precision": float(counts[PLACE_BOMB, PLACE_BOMB] / max(1, pred_bomb)),
        "place_bomb_recall": float(counts[PLACE_BOMB, PLACE_BOMB] / max(1, true_bomb)),
        "predicted_place_bomb_frequency": float(pred_counts[PLACE_BOMB] / max(1, pred_counts.sum())),
        "predicted_action_distribution": {str(i): int(v) for i, v in enumerate(pred_counts)},
        "confusion_matrix": counts.tolist(),
    }


def _mode_loss_mask(actions, valid_mask, mode):
    loss_mask = valid_mask.astype(np.float32).copy()
    if mode == "movement":
        loss_mask[actions == PLACE_BOMB] = 0.0
    elif mode == "bomb_light":
        pass
    elif mode != "all":
        raise ValueError(f"Unsupported BC mode: {mode}")
    return loss_mask


def _make_chunks(observations, actions, valid_mask, episode_starts, weights, mode, seq_len, burn_in, overfit_subset=0):
    episode_indices = np.arange(len(actions))
    if overfit_subset and overfit_subset > 0:
        episode_indices = episode_indices[:min(overfit_subset, len(episode_indices))]
    obs_chunks = []
    action_chunks = []
    mask_chunks = []
    weight_chunks = []
    start_chunks = []
    source_episodes = []
    for ep_idx in episode_indices:
        length = int(valid_mask[ep_idx].sum())
        if length <= 0:
            continue
        starts = list(range(0, length, max(1, seq_len - burn_in)))
        for start in starts:
            end = min(length, start + seq_len)
            actual = end - start
            obs = np.zeros((seq_len, *observations.shape[2:]), dtype=np.float32)
            act = np.zeros((seq_len,), dtype=np.int64)
            mask = np.zeros((seq_len,), dtype=np.float32)
            w = np.ones((seq_len,), dtype=np.float32)
            ep_start = np.zeros((seq_len,), dtype=np.float32)
            obs[:actual] = observations[ep_idx, start:end]
            act[:actual] = actions[ep_idx, start:end]
            base_mask = _mode_loss_mask(actions[ep_idx, start:end], valid_mask[ep_idx, start:end], mode)
            if burn_in > 0:
                base_mask[:min(burn_in, actual)] = 0.0
            mask[:actual] = base_mask
            w[:actual] = weights[ep_idx, start:end]
            ep_start[:actual] = episode_starts[ep_idx, start:end]
            if start > 0:
                ep_start[0] = 0.0
            if mask.sum() <= 0:
                continue
            obs_chunks.append(obs)
            action_chunks.append(act)
            mask_chunks.append(mask)
            weight_chunks.append(w)
            start_chunks.append(ep_start)
            source_episodes.append(ep_idx)
    if not obs_chunks:
        raise ValueError("No recurrent BC chunks were created. Check seq_len/burn_in/mode settings.")
    return (
        np.asarray(obs_chunks, dtype=np.float32),
        np.asarray(action_chunks, dtype=np.int64),
        np.asarray(mask_chunks, dtype=np.float32),
        np.asarray(weight_chunks, dtype=np.float32),
        np.asarray(start_chunks, dtype=np.float32),
        np.asarray(source_episodes, dtype=np.int32),
    )


def train(args):
    data = np.load(args.dataset, allow_pickle=True)
    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    mask = data["valid_mask"].astype(np.float32)
    is_bomb = data["is_bomb_action"].astype(bool) if "is_bomb_action" in data else actions == PLACE_BOMB
    is_escape = data["is_post_bomb_escape"].astype(bool) if "is_post_bomb_escape" in data else np.zeros_like(is_bomb)
    episode_starts = data["episode_starts"].astype(np.float32) if "episode_starts" in data else np.zeros_like(mask)
    episode_starts[:, 0] = 1.0
    weights = np.ones_like(mask, dtype=np.float32)
    weights[is_bomb] *= args.bomb_weight
    weights[is_escape] *= args.escape_weight

    chunks_obs, chunks_actions, chunks_mask, chunks_weights, chunks_starts, chunk_episodes = _make_chunks(
        observations,
        actions,
        mask,
        episode_starts,
        weights,
        args.mode,
        args.seq_len,
        args.burn_in if args.loss_after_burnin else 0,
        args.overfit_subset,
    )
    class_weights_np = np.ones(NUM_ACTIONS, dtype=np.float32)
    if args.class_balance:
        flat_actions = chunks_actions[chunks_mask.astype(bool)]
        counts = np.bincount(flat_actions, minlength=NUM_ACTIONS).astype(np.float32)
        mean_count = counts[counts > 0].mean()
        class_weights_np = np.sqrt(mean_count / np.maximum(counts, 1.0))
        class_weights_np = np.clip(class_weights_np, args.min_class_weight, args.max_class_weight)

    rng = np.random.default_rng(args.seed)
    if args.episode_split:
        unique_eps = rng.permutation(np.unique(chunk_episodes))
        split_ep = int(len(unique_eps) * (1.0 - args.val_fraction))
        train_eps = set(int(v) for v in unique_eps[:split_ep])
        train_idx = np.asarray([idx for idx, ep in enumerate(chunk_episodes) if int(ep) in train_eps], dtype=np.int64)
        val_idx = np.asarray([idx for idx, ep in enumerate(chunk_episodes) if int(ep) not in train_eps], dtype=np.int64)
    else:
        indices = rng.permutation(len(chunks_actions))
        split = int(len(indices) * (1.0 - args.val_fraction))
        train_idx = indices[:split]
        val_idx = indices[split:]
    if len(val_idx) == 0:
        val_idx = train_idx.copy()

    model = _make_model(args)
    policy = model.policy
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    device = policy.device
    class_weights = torch.as_tensor(class_weights_np, dtype=torch.float32, device=device)

    history = []
    for epoch in range(args.epochs):
        losses = []
        rng.shuffle(train_idx)
        for start in range(0, len(train_idx), args.batch_size):
            batch_idx = train_idx[start:start + args.batch_size]
            obs = torch.as_tensor(chunks_obs[batch_idx], dtype=torch.float32, device=device)
            act = torch.as_tensor(chunks_actions[batch_idx], dtype=torch.long, device=device)
            msk = torch.as_tensor(chunks_mask[batch_idx], dtype=torch.float32, device=device)
            w = torch.as_tensor(chunks_weights[batch_idx], dtype=torch.float32, device=device)
            starts = torch.as_tensor(chunks_starts[batch_idx], dtype=torch.float32, device=device)
            loss, _pred, _true, _flat_mask = _batch_loss(policy, obs, act, msk, w, episode_start_seq=starts, class_weights=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
        metric = _metrics(
            policy,
            chunks_obs[val_idx],
            chunks_actions[val_idx],
            chunks_mask[val_idx],
            chunks_weights[val_idx],
            episode_starts=chunks_starts[val_idx],
            batch_size=args.batch_size,
            device=device,
        )
        train_metric = {}
        if args.report_train_metrics:
            train_metric = {
                f"train_{key}": value for key, value in _metrics(
                    policy,
                    chunks_obs[train_idx],
                    chunks_actions[train_idx],
                    chunks_mask[train_idx],
                    chunks_weights[train_idx],
                    episode_starts=chunks_starts[train_idx],
                    batch_size=args.batch_size,
                    device=device,
                ).items()
            }
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **metric, **train_metric}
        history.append(row)
        print(json.dumps(row))
        policy.train()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    flat_actions = chunks_actions[chunks_mask.astype(bool)]
    summary = {
        "dataset": args.dataset,
        "episodes": int(len(actions)),
        "chunks": int(len(chunks_actions)),
        "train_chunks": int(len(train_idx)),
        "val_chunks": int(len(val_idx)),
        "steps": int(chunks_mask.sum()),
        "action_distribution": {str(i): int(v) for i, v in enumerate(np.bincount(flat_actions, minlength=NUM_ACTIONS))},
        "bomb_fraction": float((flat_actions == PLACE_BOMB).mean()) if len(flat_actions) else 0.0,
        "mode": args.mode,
        "seq_len": int(args.seq_len),
        "burn_in": int(args.burn_in if args.loss_after_burnin else 0),
        "episode_split": bool(args.episode_split),
        "overfit_subset": int(args.overfit_subset),
        "bomb_weight": float(args.bomb_weight),
        "escape_weight": float(args.escape_weight),
        "class_balance": bool(args.class_balance),
        "class_weights": {str(i): float(v) for i, v in enumerate(class_weights_np)},
        "output": args.output,
        "final": history[-1] if history else {},
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Supervised sequence BC warm-start for RecurrentPPO.")
    parser.add_argument("--dataset", default="ml/datasets/recurrent_bc_online_robust.npz")
    parser.add_argument("--output", default="ml/checkpoints/rl_agent_recurrent/recurrent_bc_online_robust.zip")
    parser.add_argument("--base_policy", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--burn_in", type=int, default=16)
    parser.add_argument("--loss_after_burnin", action="store_true")
    parser.add_argument("--episode_split", action="store_true")
    parser.add_argument("--overfit_subset", type=int, default=0)
    parser.add_argument("--mode", choices=["all", "movement", "bomb_light"], default="all")
    parser.add_argument("--bomb_weight", type=float, default=1.0)
    parser.add_argument("--escape_weight", type=float, default=1.0)
    parser.add_argument("--class_balance", action="store_true")
    parser.add_argument("--min_class_weight", type=float, default=0.5)
    parser.add_argument("--max_class_weight", type=float, default=3.0)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--report_train_metrics", action="store_true")
    parser.add_argument("--seed", type=int, default=9801)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
