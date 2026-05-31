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

from agent.rl_agent_recurrent.bomb_outcome_model import BombOutcomeCnnLstm


class BombOutcomeDataset(Dataset):
    def __init__(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.obs = data["observations"].astype(np.float32)
        self.mask = data["valid_mask"].astype(bool)
        self.boxes = data["expected_boxes_destroyed"].astype(np.float32)
        self.death = data["death_after_bomb"].astype(np.float32)
        self.zero = data["zero_value"].astype(np.float32)
        self.survived = data["survived_after_bomb"].astype(np.float32)
        self.escape = data["has_escape_after_bomb"].astype(np.float32)
        self.trapped = data["trapped_if_bomb"].astype(np.float32)
        self.reachable = data["reachable_delta"].astype(np.float32)
        self.source = data["source_type"].astype(np.int16)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.mask[idx]),
            torch.tensor(self.boxes[idx], dtype=torch.float32),
            torch.tensor(self.death[idx], dtype=torch.float32),
            torch.tensor(self.zero[idx], dtype=torch.float32),
            torch.tensor(self.survived[idx], dtype=torch.float32),
            torch.tensor(self.escape[idx], dtype=torch.float32),
            torch.tensor(self.reachable[idx], dtype=torch.float32),
            torch.tensor(int(self.source[idx]), dtype=torch.long),
        )


def load_model(path: str, in_channels: int, device: torch.device) -> BombOutcomeCnnLstm:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    model = BombOutcomeCnnLstm(
        in_channels=int(config.get("in_channels", in_channels)),
        embedding_dim=int(config.get("embedding_dim", 128)),
        hidden_size=int(config.get("hidden_size", 128)),
        num_lstm_layers=int(config.get("num_lstm_layers", 1)),
        dropout=float(config.get("dropout", 0.0)),
        layer_norm=bool(config.get("layer_norm", False)),
    ).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=False)
    return model


def freeze_except_outcome_heads(model: BombOutcomeCnnLstm) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for head in [
        model.box_value_head,
        model.death_risk_head,
        model.zero_value_head,
        model.escape_success_head,
        model.reachable_delta_head,
    ]:
        for param in head.parameters():
            param.requires_grad = True


def first_valid_score(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    first = mask.float().argmax(dim=1)
    return values[torch.arange(values.shape[0], device=values.device), first]


def bce_with_weights(logits, labels, weights, gamma: float):
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if gamma > 0:
        probs = torch.sigmoid(logits)
        pt = torch.where(labels > 0.5, probs, 1.0 - probs)
        loss = loss * torch.pow(1.0 - pt, gamma)
    return (loss * weights).mean()


def sample_weights(source, args):
    weights = torch.ones_like(source, dtype=torch.float32)
    weights = torch.where(source == 0, torch.full_like(weights, args.useful_weight, dtype=torch.float32), weights)
    weights = torch.where(source == 1, torch.full_like(weights, args.zero_weight, dtype=torch.float32), weights)
    weights = torch.where(source == 2, torch.full_like(weights, args.death_weight, dtype=torch.float32), weights)
    weights = torch.where(source == 3, torch.full_like(weights, args.safe_nonbomb_weight, dtype=torch.float32), weights)
    weights = torch.where(source == 4, torch.full_like(weights, args.counterfactual_weight, dtype=torch.float32), weights)
    return weights


def train_epoch(model, loader, optimizer, args, device):
    model.train()
    totals = []
    for obs, mask, boxes, death, zero, survived, escape, reachable, source in loader:
        obs = obs.to(device=device, dtype=torch.float32)[:, :1]
        mask = mask.to(device=device, dtype=torch.bool)[:, :1]
        boxes = boxes.to(device=device)
        death = death.to(device=device)
        zero = zero.to(device=device)
        survived = survived.to(device=device)
        escape = escape.to(device=device)
        reachable = reachable.to(device=device)
        source = source.to(device=device)
        weights = sample_weights(source, args).to(device)

        out, _ = model(obs)
        pred_boxes = first_valid_score(out["box_value"], mask)
        pred_death = first_valid_score(out["death_risk_logit"], mask)
        pred_zero = first_valid_score(out["zero_value_logit"], mask)
        pred_survival = first_valid_score(out["escape_success_logit"], mask)
        pred_reachable = first_valid_score(out["reachable_delta"], mask)

        box_loss = (F.huber_loss(pred_boxes, boxes, reduction="none") * weights).mean()
        death_loss = bce_with_weights(pred_death, death, weights, args.focal_gamma)
        zero_loss = bce_with_weights(pred_zero, zero, weights, args.focal_gamma)
        survival_loss = bce_with_weights(pred_survival, survived * escape, weights, args.focal_gamma)
        reachable_loss = (F.huber_loss(pred_reachable, reachable, reduction="none") * weights).mean()

        useful_score = pred_boxes + torch.sigmoid(pred_survival) - torch.sigmoid(pred_death) - torch.sigmoid(pred_zero)
        pos = useful_score[source == 0]
        neg = useful_score[(source == 1) | (source == 2)]
        if len(pos) and len(neg):
            count = min(len(pos), len(neg))
            rank_loss = F.relu(args.ranking_margin - (pos[:count] - neg[:count])).mean()
        else:
            rank_loss = torch.zeros((), device=device)

        loss = (
            args.box_loss_weight * box_loss
            + args.death_loss_weight * death_loss
            + args.zero_loss_weight * zero_loss
            + args.survival_loss_weight * survival_loss
            + args.reachable_loss_weight * reachable_loss
            + args.ranking_loss_weight * rank_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()
        totals.append(float(loss.item()))
    return {"loss": float(np.mean(totals)) if totals else 0.0}


def pr_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if labels.size == 0 or int(labels.sum()) == 0:
        return 0.0
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / max(1, int(y.sum()))
    return float(np.trapz(np.r_[1.0, precision], np.r_[0.0, recall]))


def binary_rows(labels, scores, thresholds):
    rows = []
    labels = np.asarray(labels) > 0.5
    scores = np.asarray(scores)
    for threshold in thresholds:
        pred = scores >= threshold
        tp = int(np.sum(pred & labels))
        fp = int(np.sum(pred & ~labels))
        fn = int(np.sum(~pred & labels))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(2 * precision * recall / max(1e-8, precision + recall)),
                "predicted_positive_rate": float(np.mean(pred)) if len(pred) else 0.0,
            }
        )
    return rows


@torch.no_grad()
def evaluate(model, loader, thresholds, device):
    model.eval()
    boxes = []
    pred_boxes = []
    death = []
    pred_death = []
    zero = []
    pred_zero = []
    survived = []
    pred_survival = []
    reachable = []
    pred_reachable = []
    source = []
    for obs, mask, batch_boxes, batch_death, batch_zero, batch_survived, _escape, batch_reachable, batch_source in loader:
        obs = obs.to(device=device, dtype=torch.float32)[:, :1]
        mask = mask.to(device=device, dtype=torch.bool)[:, :1]
        out, _ = model(obs)
        boxes.append(batch_boxes.numpy())
        pred_boxes.append(first_valid_score(out["box_value"], mask).cpu().numpy())
        death.append(batch_death.numpy())
        pred_death.append(torch.sigmoid(first_valid_score(out["death_risk_logit"], mask)).cpu().numpy())
        zero.append(batch_zero.numpy())
        pred_zero.append(torch.sigmoid(first_valid_score(out["zero_value_logit"], mask)).cpu().numpy())
        survived.append(batch_survived.numpy())
        pred_survival.append(torch.sigmoid(first_valid_score(out["escape_success_logit"], mask)).cpu().numpy())
        reachable.append(batch_reachable.numpy())
        pred_reachable.append(first_valid_score(out["reachable_delta"], mask).cpu().numpy())
        source.append(batch_source.numpy())
    boxes = np.concatenate(boxes) if boxes else np.zeros(0)
    pred_boxes = np.concatenate(pred_boxes) if pred_boxes else np.zeros(0)
    death = np.concatenate(death) if death else np.zeros(0)
    pred_death = np.concatenate(pred_death) if pred_death else np.zeros(0)
    zero = np.concatenate(zero) if zero else np.zeros(0)
    pred_zero = np.concatenate(pred_zero) if pred_zero else np.zeros(0)
    survived = np.concatenate(survived) if survived else np.zeros(0)
    pred_survival = np.concatenate(pred_survival) if pred_survival else np.zeros(0)
    reachable = np.concatenate(reachable) if reachable else np.zeros(0)
    pred_reachable = np.concatenate(pred_reachable) if pred_reachable else np.zeros(0)
    source = np.concatenate(source) if source else np.zeros(0, dtype=np.int16)
    useful = source == 0
    zero_sources = source == 1
    death_sources = source == 2
    return {
        "box_mae": float(np.mean(np.abs(pred_boxes - boxes))) if len(boxes) else 0.0,
        "box_mae_useful": float(np.mean(np.abs(pred_boxes[useful] - boxes[useful]))) if np.any(useful) else 0.0,
        "reachable_mae": float(np.mean(np.abs(pred_reachable - reachable))) if len(reachable) else 0.0,
        "death_pr_auc": pr_auc(death, pred_death),
        "zero_pr_auc": pr_auc(zero, pred_zero),
        "survival_pr_auc": pr_auc(survived, pred_survival),
        "death_rows": binary_rows(death, pred_death, thresholds),
        "zero_rows": binary_rows(zero, pred_zero, thresholds),
        "survival_rows": binary_rows(survived, pred_survival, thresholds),
        "score_means": {
            "pred_box_useful": float(pred_boxes[useful].mean()) if np.any(useful) else 0.0,
            "pred_box_zero": float(pred_boxes[zero_sources].mean()) if np.any(zero_sources) else 0.0,
            "pred_death_death_source": float(pred_death[death_sources].mean()) if np.any(death_sources) else 0.0,
            "pred_zero_zero_source": float(pred_zero[zero_sources].mean()) if np.any(zero_sources) else 0.0,
        },
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
        "stage": "bomb_outcome_model",
        "metrics": metrics,
    }
    torch.save(checkpoint, output)
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train offline multi-outcome bomb model.")
    parser.add_argument("--dataset", default="ml/datasets/bomb_outcome_dataset.npz")
    parser.add_argument("--init_checkpoint", default="ml/checkpoints/rl_agent_recurrent/modular_bomb_value_now.pt")
    parser.add_argument("--output", default="ml/checkpoints/rl_agent_recurrent/bomb_outcome_model.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--thresholds", default="0.3,0.5,0.7")
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--useful_weight", type=float, default=4.0)
    parser.add_argument("--zero_weight", type=float, default=4.0)
    parser.add_argument("--death_weight", type=float, default=6.0)
    parser.add_argument("--safe_nonbomb_weight", type=float, default=1.0)
    parser.add_argument("--counterfactual_weight", type=float, default=2.0)
    parser.add_argument("--box_loss_weight", type=float, default=1.0)
    parser.add_argument("--death_loss_weight", type=float, default=1.5)
    parser.add_argument("--zero_loss_weight", type=float, default=1.2)
    parser.add_argument("--survival_loss_weight", type=float, default=1.0)
    parser.add_argument("--reachable_loss_weight", type=float, default=0.2)
    parser.add_argument("--ranking_loss_weight", type=float, default=0.5)
    parser.add_argument("--ranking_margin", type=float, default=0.5)
    parser.add_argument("--focal_gamma", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=10100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ds = BombOutcomeDataset(args.dataset)
    val_size = max(1, int(len(ds) * args.val_fraction))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    model = load_model(args.init_checkpoint, int(ds.obs.shape[2]), device)
    if args.freeze_backbone:
        freeze_except_outcome_heads(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    history = []
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, args, device)
        val_metrics = evaluate(model, val_loader, thresholds, device)
        row = {"epoch": epoch + 1, **train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row))
    final = evaluate(model, val_loader, thresholds, device)
    report = {
        "train_samples": int(train_size),
        "val_samples": int(val_size),
        "history": history,
        "final": final,
    }
    save_checkpoint(args.output, model, args, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
