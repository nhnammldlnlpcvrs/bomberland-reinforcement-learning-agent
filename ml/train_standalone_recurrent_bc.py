from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.constants import NUM_ACTIONS, PLACE_BOMB
from agent.rl_agent_recurrent.standalone_model import StandaloneBomberCnnLstm


class ChunkDataset(Dataset):
    def __init__(self, obs, actions, mask, source_episodes):
        self.obs = obs
        self.actions = actions
        self.mask = mask
        self.source_episodes = source_episodes

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.mask[idx]),
            int(self.source_episodes[idx]),
        )


def _mode_mask(actions, valid_mask, mode):
    mask = valid_mask.astype(np.float32).copy()
    if mode == "movement":
        mask[actions == PLACE_BOMB] = 0.0
    elif mode in {"all", "bomb_light"}:
        pass
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return mask


def make_chunks(data, seq_len, burn_in, mode, overfit_subset=0):
    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    valid = data["valid_mask"].astype(bool)
    episode_indices = np.arange(len(actions))
    if overfit_subset and overfit_subset > 0:
        episode_indices = episode_indices[:min(overfit_subset, len(episode_indices))]
    obs_chunks = []
    action_chunks = []
    mask_chunks = []
    source_episodes = []
    stride = max(1, seq_len - burn_in)
    for ep_idx in episode_indices:
        length = int(valid[ep_idx].sum())
        if length <= 0:
            continue
        for start in range(0, length, stride):
            end = min(length, start + seq_len)
            actual = end - start
            obs = np.zeros((seq_len, *observations.shape[2:]), dtype=np.float32)
            act = np.zeros((seq_len,), dtype=np.int64)
            mask = np.zeros((seq_len,), dtype=np.float32)
            obs[:actual] = observations[ep_idx, start:end]
            act[:actual] = actions[ep_idx, start:end]
            base_mask = _mode_mask(actions[ep_idx, start:end], valid[ep_idx, start:end], mode)
            if burn_in > 0:
                base_mask[:min(burn_in, actual)] = 0.0
            mask[:actual] = base_mask
            if mask.sum() <= 0:
                continue
            obs_chunks.append(obs)
            action_chunks.append(act)
            mask_chunks.append(mask)
            source_episodes.append(ep_idx)
    if not obs_chunks:
        raise ValueError("No chunks created")
    return (
        np.asarray(obs_chunks, dtype=np.float32),
        np.asarray(action_chunks, dtype=np.int64),
        np.asarray(mask_chunks, dtype=np.float32),
        np.asarray(source_episodes, dtype=np.int32),
    )


def split_chunks(source_episodes, val_fraction, seed, episode_split):
    rng = np.random.default_rng(seed)
    if episode_split:
        eps = rng.permutation(np.unique(source_episodes))
        split = int(len(eps) * (1.0 - val_fraction))
        train_eps = set(int(v) for v in eps[:split])
        train_idx = np.asarray([i for i, ep in enumerate(source_episodes) if int(ep) in train_eps], dtype=np.int64)
        val_idx = np.asarray([i for i, ep in enumerate(source_episodes) if int(ep) not in train_eps], dtype=np.int64)
    else:
        idx = rng.permutation(len(source_episodes))
        split = int(len(idx) * (1.0 - val_fraction))
        train_idx = idx[:split]
        val_idx = idx[split:]
    if len(val_idx) == 0:
        val_idx = train_idx.copy()
    return train_idx, val_idx


def action_distribution(actions, mask):
    values = actions[mask.astype(bool)]
    return {str(i): int(v) for i, v in enumerate(np.bincount(values.reshape(-1), minlength=NUM_ACTIONS))}


def class_weights_for(actions, mask, max_class_weight):
    values = actions[mask.astype(bool)]
    counts = np.bincount(values.reshape(-1), minlength=NUM_ACTIONS).astype(np.float32)
    mean = counts[counts > 0].mean() if np.any(counts > 0) else 1.0
    weights = np.sqrt(mean / np.maximum(counts, 1.0))
    return np.clip(weights, 0.5, max_class_weight).astype(np.float32)


def masked_ce_loss(logits, actions, mask, class_weights=None):
    batch, seq_len, num_actions = logits.shape
    flat_logits = logits.reshape(batch * seq_len, num_actions)
    flat_actions = actions.reshape(batch * seq_len)
    flat_mask = mask.reshape(batch * seq_len)
    ce = F.cross_entropy(flat_logits, flat_actions, weight=class_weights, reduction="none")
    return (ce * flat_mask).sum() / torch.clamp(flat_mask.sum(), min=1.0)


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    confusion = np.zeros((NUM_ACTIONS, NUM_ACTIONS), dtype=np.int64)
    pred_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    losses = []
    for obs, actions, mask, _eps in loader:
        obs = obs.to(device=device, dtype=torch.float32)
        actions = actions.to(device=device, dtype=torch.long)
        mask = mask.to(device=device, dtype=torch.float32)
        logits, _ = model(obs)
        losses.append(float(masked_ce_loss(logits, actions, mask).item()))
        pred = logits.argmax(dim=-1).detach().cpu().numpy()
        true = actions.detach().cpu().numpy()
        valid = mask.detach().cpu().numpy().astype(bool)
        for t, p in zip(true[valid], pred[valid]):
            confusion[int(t), int(p)] += 1
        pred_counts += np.bincount(pred[valid], minlength=NUM_ACTIONS)
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    per_action = {
        str(i): float(confusion[i, i] / max(1, confusion[i].sum()))
        for i in range(NUM_ACTIONS)
    }
    pred_bomb = int(confusion[:, PLACE_BOMB].sum())
    true_bomb = int(confusion[PLACE_BOMB].sum())
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(correct / max(1, total)),
        "per_action_accuracy": per_action,
        "place_bomb_precision": float(confusion[PLACE_BOMB, PLACE_BOMB] / max(1, pred_bomb)),
        "place_bomb_recall": float(confusion[PLACE_BOMB, PLACE_BOMB] / max(1, true_bomb)),
        "predicted_place_bomb_frequency": float(pred_counts[PLACE_BOMB] / max(1, pred_counts.sum())),
        "predicted_action_distribution": {str(i): int(v) for i, v in enumerate(pred_counts)},
        "confusion_matrix": confusion.tolist(),
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = np.load(args.dataset, allow_pickle=True)
    obs, actions, mask, source_eps = make_chunks(
        data,
        seq_len=args.seq_len,
        burn_in=args.burn_in,
        mode=args.mode,
        overfit_subset=args.overfit_subset,
    )
    train_idx, val_idx = split_chunks(source_eps, args.val_fraction, args.seed, args.episode_split)
    train_ds = ChunkDataset(obs[train_idx], actions[train_idx], mask[train_idx], source_eps[train_idx])
    val_ds = ChunkDataset(obs[val_idx], actions[val_idx], mask[val_idx], source_eps[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    metric_train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    model = StandaloneBomberCnnLstm(
        in_channels=int(obs.shape[2]),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    weights = None
    if args.class_balance:
        weights = torch.as_tensor(class_weights_for(actions[train_idx], mask[train_idx], args.max_class_weight), device=device)
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_obs, batch_actions, batch_mask, _eps in train_loader:
            batch_obs = batch_obs.to(device=device, dtype=torch.float32)
            batch_actions = batch_actions.to(device=device, dtype=torch.long)
            batch_mask = batch_mask.to(device=device, dtype=torch.float32)
            logits, _ = model(batch_obs)
            loss = masked_ce_loss(logits, batch_actions, batch_mask, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
        train_metrics = evaluate_model(model, metric_train_loader, device)
        val_metrics = evaluate_model(model, val_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_bomb_recall": train_metrics["place_bomb_recall"],
            "val_bomb_recall": val_metrics["place_bomb_recall"],
            "train_predicted_action_distribution": train_metrics["predicted_action_distribution"],
            "val_predicted_action_distribution": val_metrics["predicted_action_distribution"],
        }
        history.append(row)
        print(json.dumps(row))
    final_train = evaluate_model(model, metric_train_loader, device)
    final_val = evaluate_model(model, val_loader, device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "in_channels": int(obs.shape[2]),
            "embedding_dim": int(args.embedding_dim),
            "hidden_size": int(args.hidden_size),
            "num_actions": NUM_ACTIONS,
            "seq_len": int(args.seq_len),
            "burn_in": int(args.burn_in),
        },
        "dataset": args.dataset,
    }
    torch.save(checkpoint, output)
    summary = {
        "dataset": args.dataset,
        "output": str(output),
        "chunks": int(len(actions)),
        "train_chunks": int(len(train_idx)),
        "val_chunks": int(len(val_idx)),
        "seq_len": int(args.seq_len),
        "burn_in": int(args.burn_in),
        "overfit_subset": int(args.overfit_subset),
        "mode": args.mode,
        "class_balance": bool(args.class_balance),
        "train_action_distribution": action_distribution(actions[train_idx], mask[train_idx]),
        "val_action_distribution": action_distribution(actions[val_idx], mask[val_idx]),
        "final_train": final_train,
        "final_val": final_val,
        "history": history,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Standalone PyTorch CNN-LSTM BC for Bomberland.")
    parser.add_argument("--dataset", default="ml/datasets/recurrent_bc_online_robust.npz")
    parser.add_argument("--output", default="ml/checkpoints/rl_agent_recurrent/standalone_bc_full.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--burn_in", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--class_balance", action="store_true")
    parser.add_argument("--max_class_weight", type=float, default=10.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--overfit_subset", type=int, default=0)
    parser.add_argument("--episode_split", action="store_true")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--mode", choices=["all", "movement", "bomb_light"], default="all")
    parser.add_argument("--seed", type=int, default=9950)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
