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


class BombRankingDataset(Dataset):
    def __init__(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.pos_obs = data["pos_observations"].astype(np.float32)
        self.pos_mask = data["pos_valid_mask"].astype(bool)
        self.neg_obs = data["neg_observations"].astype(np.float32)
        self.neg_mask = data["neg_valid_mask"].astype(bool)
        self.neg_source = data["neg_source"].astype(np.int16)
        self.pair_pos_idx = data["pair_pos_idx"].astype(np.int64)
        self.pair_neg_idx = data["pair_neg_idx"].astype(np.int64)

    def __len__(self):
        return len(self.pair_pos_idx)

    def __getitem__(self, idx):
        pos_idx = self.pair_pos_idx[idx]
        neg_idx = self.pair_neg_idx[idx]
        return (
            torch.from_numpy(self.pos_obs[pos_idx]),
            torch.from_numpy(self.pos_mask[pos_idx]),
            torch.from_numpy(self.neg_obs[neg_idx]),
            torch.from_numpy(self.neg_mask[neg_idx]),
            int(self.neg_source[neg_idx]),
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


def freeze_except_bomb_head(model: ModularBomberCnnLstm) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.bomb_head.parameters():
        param.requires_grad = True


def first_valid_score(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Ranking samples are context sequences. Current data uses length 1 for bomb
    # contexts, but first-valid keeps the code robust to longer windows.
    first = mask.float().argmax(dim=1)
    return logits[torch.arange(logits.shape[0], device=logits.device), first]


def bce_with_optional_focal(logits, targets, weights, gamma: float):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if gamma > 0:
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        loss = loss * torch.pow(1.0 - pt, gamma)
    return (loss * weights).mean()


def train_epoch(model, loader, optimizer, args, device):
    model.train()
    losses = []
    bce_losses = []
    rank_losses = []
    for pos_obs, pos_mask, neg_obs, neg_mask, neg_source in loader:
        pos_obs = pos_obs.to(device=device, dtype=torch.float32)
        neg_obs = neg_obs.to(device=device, dtype=torch.float32)
        pos_mask = pos_mask.to(device=device, dtype=torch.bool)
        neg_mask = neg_mask.to(device=device, dtype=torch.bool)
        neg_source = neg_source.to(device=device, dtype=torch.long)

        pos_logits = first_valid_score(model(pos_obs)[0]["bomb_logit"], pos_mask)
        neg_logits = first_valid_score(model(neg_obs)[0]["bomb_logit"], neg_mask)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        targets = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
        neg_weights = torch.where(
            neg_source == 2,
            torch.full_like(neg_logits, args.hard_negative_weight),
            torch.ones_like(neg_logits),
        )
        weights = torch.cat([torch.ones_like(pos_logits), neg_weights], dim=0)
        bce = bce_with_optional_focal(logits, targets, weights, args.focal_gamma)
        rank = F.relu(args.margin - (pos_logits - neg_logits)).mean()
        loss = args.bce_loss_weight * bce + args.ranking_loss_weight * rank

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.bomb_head.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
        bce_losses.append(float(bce.item()))
        rank_losses.append(float(rank.item()))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "bce_loss": float(np.mean(bce_losses)) if bce_losses else 0.0,
        "ranking_loss": float(np.mean(rank_losses)) if rank_losses else 0.0,
    }


@torch.no_grad()
def collect_scores(model, loader, device):
    model.eval()
    pos_scores = []
    neg_scores = []
    neg_sources = []
    for pos_obs, pos_mask, neg_obs, neg_mask, neg_source in loader:
        pos_obs = pos_obs.to(device=device, dtype=torch.float32)
        neg_obs = neg_obs.to(device=device, dtype=torch.float32)
        pos_mask = pos_mask.to(device=device, dtype=torch.bool)
        neg_mask = neg_mask.to(device=device, dtype=torch.bool)
        pos = torch.sigmoid(first_valid_score(model(pos_obs)[0]["bomb_logit"], pos_mask)).cpu().numpy()
        neg = torch.sigmoid(first_valid_score(model(neg_obs)[0]["bomb_logit"], neg_mask)).cpu().numpy()
        pos_scores.append(pos)
        neg_scores.append(neg)
        neg_sources.append(neg_source.numpy())
    return (
        np.concatenate(pos_scores) if pos_scores else np.zeros(0),
        np.concatenate(neg_scores) if neg_scores else np.zeros(0),
        np.concatenate(neg_sources) if neg_sources else np.zeros(0, dtype=np.int16),
    )


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = labels[order]
    positives = max(1, int(y.sum()))
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / positives
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapz(precision, recall))


def threshold_metrics(pos_scores, neg_scores, neg_sources, thresholds):
    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)]).astype(np.int64)
    scores = np.concatenate([pos_scores, neg_scores])
    rows = []
    for threshold in thresholds:
        tp = int(np.sum(pos_scores >= threshold))
        fn = int(np.sum(pos_scores < threshold))
        fp = int(np.sum(neg_scores >= threshold))
        tn = int(np.sum(neg_scores < threshold))
        hard_mask = neg_sources == 2
        hard_fp = int(np.sum(neg_scores[hard_mask] >= threshold))
        hard_total = int(np.sum(hard_mask))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(tp / max(1, tp + fp)),
                "recall": float(tp / max(1, tp + fn)),
                "f1": float((2 * tp) / max(1, 2 * tp + fp + fn)),
                "unsafe_false_bomb_rate": float(fp / max(1, fp + tn)),
                "hard_negative_false_bomb_rate": float(hard_fp / max(1, hard_total)),
                "predicted_bomb_count": int(tp + fp),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return {"pr_auc": pr_auc(labels, scores), "rows": rows}


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
        "stage": "bomb_calibrated",
        "metrics": metrics,
    }
    torch.save(checkpoint, output)
    output.with_suffix(".json").write_text(json.dumps(checkpoint, indent=2, default=str), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train calibrated modular bomb head with BCE + pairwise ranking.")
    parser.add_argument("--dataset", default="ml/datasets/recurrent_bomb_ranking_dataset.npz")
    parser.add_argument("--init_checkpoint", default="ml/checkpoints/rl_agent_recurrent/modular_movement_long.pt")
    parser.add_argument("--output", default="ml/checkpoints/rl_agent_recurrent/modular_bomb_calibrated.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--bce_loss_weight", type=float, default=1.0)
    parser.add_argument("--ranking_loss_weight", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--hard_negative_weight", type=float, default=3.0)
    parser.add_argument("--focal_gamma", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=9980)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.8,0.9")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ds = BombRankingDataset(args.dataset)
    val_size = max(1, int(len(ds) * args.val_fraction))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    model = load_model(args.init_checkpoint, int(ds.pos_obs.shape[2]), device)
    freeze_except_bomb_head(model)
    optimizer = torch.optim.AdamW(model.bomb_head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    thresholds = [float(t) for t in args.thresholds.split(",")]
    history = []
    for epoch in range(args.epochs):
        losses = train_epoch(model, train_loader, optimizer, args, device)
        pos_scores, neg_scores, neg_sources = collect_scores(model, val_loader, device)
        metrics = threshold_metrics(pos_scores, neg_scores, neg_sources, thresholds)
        row = {"epoch": epoch + 1, **losses, "val_pr_auc": metrics["pr_auc"], "threshold_rows": metrics["rows"]}
        history.append(row)
        print(json.dumps(row))
    pos_scores, neg_scores, neg_sources = collect_scores(model, val_loader, device)
    final_metrics = {
        "history": history,
        "final": threshold_metrics(pos_scores, neg_scores, neg_sources, thresholds),
        "train_pairs": int(train_size),
        "val_pairs": int(val_size),
        "pos_score_mean": float(pos_scores.mean()) if len(pos_scores) else 0.0,
        "neg_score_mean": float(neg_scores.mean()) if len(neg_scores) else 0.0,
    }
    save_checkpoint(args.output, model, args, final_metrics)
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
