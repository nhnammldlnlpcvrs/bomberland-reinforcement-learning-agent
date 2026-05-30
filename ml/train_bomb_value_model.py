from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class BombValueNet(nn.Module):
    def __init__(self, in_channels=19, scalar_dim=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + scalar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs, scalars):
        return self.head(torch.cat([self.cnn(obs), scalars], dim=1)).squeeze(1)


def _metrics(logits, labels, threshold):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y = labels.detach().cpu().numpy().astype(bool)
    pred = probs >= threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    tn = int((~pred & ~y).sum())
    fn = int((~pred & y).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    acc = (tp + tn) / max(1, len(y))
    return {"accuracy": acc, "precision": precision, "recall": recall, "false_positive_rate": fpr, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _metrics_from_probs(probs, labels, threshold):
    y = labels.astype(bool)
    pred = probs >= threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    tn = int((~pred & ~y).sum())
    fn = int((~pred & y).sum())
    return {
        "accuracy": (tp + tn) / max(1, len(y)),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "false_positive_rate": fp / max(1, fp + tn),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _choose_threshold(probs, labels, target_precision):
    y = labels.astype(bool)
    best = (0.99, 0.0, 0.0)
    for threshold in np.linspace(0.05, 0.99, 95):
        pred = probs >= threshold
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision >= target_precision and recall >= best[2]:
            best = (float(threshold), float(precision), float(recall))
    return {"threshold": best[0], "precision": best[1], "recall": best[2]}


def _threshold_sweep(probs, labels, sources=None):
    rows = []
    for threshold in np.arange(0.5, 0.91, 0.05):
        row = {"threshold": float(round(threshold, 2)), **_metrics_from_probs(probs, labels, threshold)}
        if sources is not None:
            for source_id, name in ((3, "rollout_hard"), (4, "counterfactual"), (5, "legacy_hard")):
                hard_mask = sources == source_id
                if hard_mask.any():
                    row[f"{name}_false_positive_rate"] = float((probs[hard_mask] >= threshold).mean())
                    row[f"{name}_selected"] = int((probs[hard_mask] >= threshold).sum())
                    row[f"{name}_total"] = int(hard_mask.sum())
        rows.append(row)
    return rows


def _weighted_bce_loss(logits, labels, weights, pos_weight=None, focal_gamma=0.0):
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=pos_weight)
    if focal_gamma > 0:
        probs = torch.sigmoid(logits)
        pt = torch.where(labels > 0.5, probs, 1.0 - probs).clamp(1e-6, 1.0)
        loss = loss * ((1.0 - pt) ** focal_gamma)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _split_metrics_by_source(probs, labels, sources, threshold):
    out = {}
    for source_id, name in ((1, "positive"), (2, "easy_negative"), (3, "rollout_hard_negative"), (4, "counterfactual_negative"), (5, "legacy_hard_negative")):
        mask = sources == source_id
        if not mask.any():
            continue
        row = _metrics_from_probs(probs[mask], labels[mask], threshold)
        row["count"] = int(mask.sum())
        out[name] = row
    return out


def train(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    data = np.load(args.dataset)
    obs = data["observations"].astype(np.float32)
    scalars = data["scalar_features"].astype(np.float32)
    labels = data["labels"].astype(np.float32)
    sources = data["sample_type"].astype(np.int8) if "sample_type" in data.files else data["source"].astype(np.int8) if "source" in data.files else np.zeros(len(labels), dtype=np.int8)
    sample_weights = data["sample_weight"].astype(np.float32) if "sample_weight" in data.files else np.ones(len(labels), dtype=np.float32)
    hard_mask_all = np.isin(sources, [3, 4, 5])
    sample_weights[hard_mask_all & (labels == 0)] *= args.hard_negative_weight
    idx = rng.permutation(len(labels))
    split = int(len(idx) * (1.0 - args.val_fraction))
    train_idx, val_idx = idx[:split], idx[split:]
    train_ds = TensorDataset(
        torch.from_numpy(obs[train_idx]),
        torch.from_numpy(scalars[train_idx]),
        torch.from_numpy(labels[train_idx]),
        torch.from_numpy(sample_weights[train_idx]),
    )
    val_obs = torch.from_numpy(obs[val_idx])
    val_scalars = torch.from_numpy(scalars[val_idx])
    val_labels = torch.from_numpy(labels[val_idx])
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    device = torch.device(args.device)
    model = BombValueNet(in_channels=obs.shape[1], scalar_dim=scalars.shape[1]).to(device)
    pos_weight = None
    if args.use_pos_weight:
        pos_weight = torch.tensor([(labels[train_idx] == 0).sum() / max(1, (labels[train_idx] == 1).sum())], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_obs, batch_scalars, batch_labels, batch_weights in loader:
            batch_obs = batch_obs.to(device)
            batch_scalars = batch_scalars.to(device)
            batch_labels = batch_labels.to(device)
            batch_weights = batch_weights.to(device)
            logits = model(batch_obs, batch_scalars)
            loss = _weighted_bce_loss(logits, batch_labels, batch_weights, pos_weight=pos_weight, focal_gamma=args.focal_gamma if args.focal_loss else 0.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_logits = model(val_obs.to(device), val_scalars.to(device))
            val_weights = torch.as_tensor(sample_weights[val_idx], dtype=torch.float32, device=device)
            val_loss = _weighted_bce_loss(val_logits, val_labels.to(device), val_weights, pos_weight=pos_weight, focal_gamma=args.focal_gamma if args.focal_loss else 0.0).item()
            metric = _metrics(val_logits.cpu(), val_labels, args.threshold)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": float(val_loss), **metric}
        history.append(row)
        print(json.dumps(row), flush=True)
    model.eval()
    with torch.no_grad():
        val_logits = model(val_obs.to(device), val_scalars.to(device)).cpu()
    probs = torch.sigmoid(val_logits).numpy()
    threshold_info = _choose_threshold(probs, labels[val_idx], args.target_precision)
    sweep = _threshold_sweep(probs, labels[val_idx], sources[val_idx])
    hard_mask = sources[val_idx] == 4
    hard_negative_metrics = {}
    for source_id, name in ((3, "rollout_hard_negative"), (4, "counterfactual_negative"), (5, "legacy_hard_negative")):
        mask = sources[val_idx] == source_id
        if mask.any():
            hard_negative_metrics[name] = {
                "count": int(mask.sum()),
                "false_positive_rate_at_0_5": float((probs[mask] >= 0.5).mean()),
                "false_positive_rate_at_chosen_threshold": float((probs[mask] >= threshold_info["threshold"]).mean()),
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "threshold": threshold_info["threshold"],
        "in_channels": int(obs.shape[1]),
        "scalar_dim": int(scalars.shape[1]),
    }, output)
    summary = {
        "dataset": args.dataset,
        "output": str(output),
        "samples": int(len(labels)),
        "positive_fraction": float(labels.mean()),
        "threshold_info": threshold_info,
        "threshold_sweep": sweep,
        "hard_negative_metrics": hard_negative_metrics,
        "split_metrics_at_chosen_threshold": _split_metrics_by_source(probs, labels[val_idx], sources[val_idx], threshold_info["threshold"]),
        "final": history[-1],
        "history": history,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ml/datasets/bomb_value_dataset.npz")
    parser.add_argument("--output", default="ml/checkpoints/rl_agent_pure/bomb_value_model.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target_precision", type=float, default=0.8)
    parser.add_argument("--use_pos_weight", action="store_true")
    parser.add_argument("--hard_negative_weight", type=float, default=1.0)
    parser.add_argument("--focal_loss", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
