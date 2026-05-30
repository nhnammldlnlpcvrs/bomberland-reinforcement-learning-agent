"""Train the lightweight BC advisor used by agent/hybrid_agent_rl."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent" / "hybrid_agent_rl"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import HybridBCNet

ACTION_NAMES = ["STOP", "LEFT", "RIGHT", "UP", "DOWN", "BOMB"]


def _split(n, val_frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    return perm[n_val:], perm[:n_val]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    counts = {i: 0 for i in range(6)}
    pred_counts = {i: 0 for i in range(6)}
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=-1)
        total += int(y.numel())
        correct += int((pred == y).sum().item())
        for i in range(6):
            counts[i] += int((y == i).sum().item())
            pred_counts[i] += int((pred == i).sum().item())
    return correct / max(1, total), counts, pred_counts


def train(args):
    data = np.load(args.dataset, allow_pickle=True)
    x = data["observations"].astype(np.float32)
    y = data["actions"].astype(np.int64)
    if len(y) == 0:
        raise ValueError("dataset has no samples")

    train_idx, val_idx = _split(len(y), args.val_split, args.seed)
    train_ds = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]))
    val_ds = TensorDataset(torch.from_numpy(x[val_idx]), torch.from_numpy(y[val_idx]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False)

    device = torch.device("cpu")
    model = HybridBCNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = -1.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    raw_counts = {i: int((y == i).sum()) for i in range(6)}
    print(f"samples={len(y)} train={len(train_idx)} val={len(val_idx)}")
    print("action distribution:", {ACTION_NAMES[k]: v for k, v in raw_counts.items()})

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(yb.numel())
            total += int(yb.numel())
            correct += int((logits.argmax(dim=-1) == yb).sum().item())

        val_acc, _counts, pred_counts = evaluate(model, val_loader, device)
        train_acc = correct / max(1, total)
        avg_loss = total_loss / max(1, total)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"loss={avg_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": {
                        "dataset": args.dataset,
                        "best_val_acc": best_acc,
                        "action_distribution": raw_counts,
                    },
                },
                output,
            )

    best = torch.load(output, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    val_acc, counts, pred_counts = evaluate(model, val_loader, device)
    print(f"best val_acc={val_acc:.3f}")
    print("val target distribution:", {ACTION_NAMES[k]: counts[k] for k in range(6)})
    print("val pred distribution:", {ACTION_NAMES[k]: pred_counts[k] for k in range(6)})
    print(f"saved model to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ml/datasets/hybrid_agent_rl_bc.npz")
    parser.add_argument("--output", default="agent/hybrid_agent_rl/model.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
