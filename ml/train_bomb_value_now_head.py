from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_recurrent.modular_model import ModularBomberCnnLstm


class ValueNowDataset(Dataset):
    def __init__(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.obs = data["observations"].astype(np.float32)
        self.mask = data["valid_mask"].astype(bool)
        self.labels = data["label_value_now"].astype(np.float32)
        self.source = data["source_type"].astype(np.int16)
        self.boxes = data["expected_boxes"].astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.mask[idx]),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            torch.tensor(int(self.source[idx]), dtype=torch.long),
            torch.tensor(self.boxes[idx], dtype=torch.float32),
        )


def load_model(path: str, in_channels: int, device: torch.device) -> ModularBomberCnnLstm:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    model = ModularBomberCnnLstm(
        in_channels=int(config.get("in_channels", in_channels)),
        embedding_dim=int(config.get("embedding_dim", 128)),
        hidden_size=int(config.get("hidden_size", 128)),
        num_lstm_layers=int(config.get("num_lstm_layers", 1)),
        dropout=float(config.get("dropout", 0.0)),
        layer_norm=bool(config.get("layer_norm", False)),
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    return model


def freeze_except_value_head(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.bomb_value_head.parameters():
        param.requires_grad = True


def first_valid_score(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    first = mask.float().argmax(dim=1)
    return logits[torch.arange(logits.shape[0], device=logits.device), first]


def weighted_focal_bce(logits, labels, weights, gamma):
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if gamma > 0:
        probs = torch.sigmoid(logits)
        pt = torch.where(labels > 0.5, probs, 1.0 - probs)
        loss = loss * torch.pow(1.0 - pt, gamma)
    return (loss * weights).mean()


def train_epoch(model, loader, optimizer, args, device):
    model.train()
    losses = []
    bces = []
    ranks = []
    for obs, mask, labels, source, _boxes in loader:
        obs = obs.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.bool)
        obs = obs[:, :1]
        mask = mask[:, :1]
        labels = labels.to(device=device, dtype=torch.float32)
        source = source.to(device=device, dtype=torch.long)
        logits = first_valid_score(model(obs)[0]["bomb_value_logit"], mask)
        weights = torch.ones_like(labels)
        weights = torch.where(labels > 0.5, torch.full_like(weights, args.positive_weight), weights)
        weights = torch.where(source == 1, torch.full_like(weights, args.safe_zero_weight), weights)
        weights = torch.where(source == 2, torch.full_like(weights, args.unsafe_weight), weights)
        weights = torch.where(source == 3, torch.full_like(weights, args.onpolicy_zero_weight), weights)
        weights = torch.where(source == 4, torch.full_like(weights, args.onpolicy_death_weight), weights)
        bce = weighted_focal_bce(logits, labels, weights, args.focal_gamma)
        pos = logits[labels > 0.5]
        safe_zero = logits[source == 1]
        if len(pos) and len(safe_zero):
            count = min(len(pos), len(safe_zero))
            rank = F.relu(args.margin - (pos[:count] - safe_zero[:count])).mean()
        else:
            rank = torch.zeros((), device=device)
        loss = args.bce_loss_weight * bce + args.ranking_loss_weight * rank
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.bomb_value_head.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
        bces.append(float(bce.item()))
        ranks.append(float(rank.item()))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "bce_loss": float(np.mean(bces)) if bces else 0.0,
        "ranking_loss": float(np.mean(ranks)) if ranks else 0.0,
    }


def pr_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if labels.size == 0 or labels.sum() == 0:
        return 0.0
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / max(1, int(y.sum()))
    return float(np.trapz(np.r_[1.0, precision], np.r_[0.0, recall]))


@torch.no_grad()
def evaluate(model, loader, thresholds, device):
    model.eval()
    scores = []
    labels = []
    source = []
    boxes = []
    for obs, mask, batch_labels, batch_source, batch_boxes in loader:
        obs = obs.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.bool)
        obs = obs[:, :1]
        mask = mask[:, :1]
        logits = first_valid_score(model(obs)[0]["bomb_value_logit"], mask)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch_labels.numpy())
        source.append(batch_source.numpy())
        boxes.append(batch_boxes.numpy())
    scores = np.concatenate(scores) if scores else np.zeros(0)
    labels = np.concatenate(labels) if labels else np.zeros(0)
    source = np.concatenate(source) if source else np.zeros(0, dtype=np.int16)
    boxes = np.concatenate(boxes) if boxes else np.zeros(0)
    rows = []
    for threshold in thresholds:
        pred = scores >= threshold
        pos = labels > 0.5
        safe_zero = source == 1
        unsafe = source == 2
        onpolicy_zero = source == 3
        onpolicy_death = source == 4
        tp = int(np.sum(pred & pos))
        fp = int(np.sum(pred & ~pos))
        fn = int(np.sum(~pred & pos))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(2 * precision * recall / max(1e-8, precision + recall)),
                "safe_zero_value_fpr": float(np.sum(pred & safe_zero) / max(1, np.sum(safe_zero))),
                "unsafe_fpr": float(np.sum(pred & unsafe) / max(1, np.sum(unsafe))),
                "onpolicy_zero_value_fpr": float(np.sum(pred & onpolicy_zero) / max(1, np.sum(onpolicy_zero))),
                "death_after_bomb_fpr": float(np.sum(pred & onpolicy_death) / max(1, np.sum(onpolicy_death))),
                "predicted_positive_count": int(np.sum(pred)),
                "expected_boxes_mean_predicted": float(boxes[pred].mean()) if np.any(pred) else 0.0,
            }
        )
    return {
        "pr_auc": pr_auc(labels, scores),
        "score_mean_positive": float(scores[labels > 0.5].mean()) if np.any(labels > 0.5) else 0.0,
        "score_mean_safe_zero": float(scores[source == 1].mean()) if np.any(source == 1) else 0.0,
        "score_mean_unsafe": float(scores[source == 2].mean()) if np.any(source == 2) else 0.0,
        "rows": rows,
    }


def save_checkpoint(path, model, args, metrics):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "in_channels": int(model.in_channels),
            "embedding_dim": int(model.embedding_dim),
            "hidden_size": int(model.hidden_size),
            "num_lstm_layers": int(model.num_lstm_layers),
            "dropout": float(model.dropout),
            "layer_norm": bool(model.layer_norm_enabled),
        },
        "stage": "bomb_value_now",
        "metrics": metrics,
    }
    torch.save(checkpoint, output)
    output.with_suffix(".json").write_text(json.dumps(checkpoint, indent=2, default=str), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train offline bomb_value_now head for modular recurrent BC.")
    parser.add_argument("--dataset", default="ml/datasets/bomb_value_now_dataset.npz")
    parser.add_argument("--init_checkpoint", default="ml/checkpoints/rl_agent_recurrent/modular_bomb_calibrated_escape_refined.pt")
    parser.add_argument("--output", default="ml/checkpoints/rl_agent_recurrent/modular_bomb_value_now.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--safe_zero_weight", type=float, default=4.0)
    parser.add_argument("--unsafe_weight", type=float, default=2.0)
    parser.add_argument("--positive_weight", type=float, default=1.0)
    parser.add_argument("--onpolicy_zero_weight", type=float, default=8.0)
    parser.add_argument("--onpolicy_death_weight", type=float, default=10.0)
    parser.add_argument("--focal_gamma", type=float, default=1.0)
    parser.add_argument("--bce_loss_weight", type=float, default=1.0)
    parser.add_argument("--ranking_loss_weight", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--seed", type=int, default=9990)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ds = ValueNowDataset(args.dataset)
    val_size = max(1, int(len(ds) * args.val_fraction))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    model = load_model(args.init_checkpoint, int(ds.obs.shape[2]), device)
    freeze_except_value_head(model)
    optimizer = torch.optim.AdamW(model.bomb_value_head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    thresholds = [float(t) for t in args.thresholds.split(",")]
    history = []
    for epoch in range(args.epochs):
        losses = train_epoch(model, train_loader, optimizer, args, device)
        metrics = evaluate(model, val_loader, thresholds, device)
        row = {"epoch": epoch + 1, **losses, "val": metrics}
        history.append(row)
        print(json.dumps(row))
    final = evaluate(model, val_loader, thresholds, device)
    report = {
        "history": history,
        "final": final,
        "train_samples": int(train_size),
        "val_samples": int(val_size),
    }
    save_checkpoint(args.output, model, args, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
