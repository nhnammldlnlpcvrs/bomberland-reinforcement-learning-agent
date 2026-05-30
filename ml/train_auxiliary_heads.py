from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.aux_model import BomberAuxModel

try:
    from sklearn.metrics import average_precision_score
except Exception:  # pragma: no cover
    average_precision_score = None


def _standardize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(values.mean())
    std = float(values.std() + 1e-6)
    return ((values - mean) / std).astype(np.float32), mean, std


def _binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels = targets.detach().cpu().numpy() >= 0.5
    out = {"positive_rate": float(labels.mean())}
    if average_precision_score is not None and labels.any():
        out["pr_auc"] = float(average_precision_score(labels.astype(np.int32), probs))
    for threshold in (0.3, 0.5, 0.7, 0.9):
        pred = probs >= threshold
        tp = int(np.logical_and(pred, labels).sum())
        fp = int(np.logical_and(pred, ~labels).sum())
        fn = int(np.logical_and(~pred, labels).sum())
        tn = int(np.logical_and(~pred, ~labels).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
        out[f"precision@{threshold}"] = precision
        out[f"recall@{threshold}"] = recall
        out[f"accuracy@{threshold}"] = accuracy
    out["precision"] = out["precision@0.5"]
    out["recall"] = out["recall@0.5"]
    out["accuracy"] = out["accuracy@0.5"]
    return out


def focal_bce_with_logits(logits, targets, pos_weight=None, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    return ((1.0 - pt).pow(gamma) * bce).mean()


def train(args):
    data = np.load(args.dataset)
    obs = data["observations"].astype(np.float32)
    death = data["death_within_7"].astype(np.float32)
    escape = data["escaped_blast"].astype(np.float32)
    escape_available = data["has_escape_path_now"].astype(np.float32) if "has_escape_path_now" in data else np.zeros(len(obs), dtype=np.float32)
    bomb_escape_available = data["has_escape_after_bomb"].astype(np.float32) if "has_escape_after_bomb" in data else np.zeros(len(obs), dtype=np.float32)
    trapped_if_bomb = data["trapped_if_bomb"].astype(np.float32) if "trapped_if_bomb" in data else np.zeros(len(obs), dtype=np.float32)
    future_blast = data["in_future_blast"].astype(np.float32) if "in_future_blast" in data else np.zeros(len(obs), dtype=np.float32)
    boxes, box_mean, box_std = _standardize(data["boxes_destroyed_future"].astype(np.float32))
    reachable, reach_mean, reach_std = _standardize(data["reachable_area_delta"].astype(np.float32))
    safe_tiles, safe_tiles_mean, safe_tiles_std = _standardize(
        data["safe_tiles_after_bomb_count"].astype(np.float32) if "safe_tiles_after_bomb_count" in data else np.zeros(len(obs), dtype=np.float32)
    )
    blast_distance_raw = data["blast_corridor_distance"].astype(np.float32) if "blast_corridor_distance" in data else np.full(len(obs), -1.0, dtype=np.float32)
    blast_distance_raw = np.where(blast_distance_raw < 0, 14.0, blast_distance_raw)
    blast_distance, blast_distance_mean, blast_distance_std = _standardize(blast_distance_raw)
    returns, ret_mean, ret_std = _standardize(data["discounted_returns"].astype(np.float32))

    if "train_split" in data:
        train_idx = np.flatnonzero(data["train_split"].astype(np.int8) > 0)
        val_idx = np.flatnonzero(data["train_split"].astype(np.int8) <= 0)
    else:
        rng = np.random.default_rng(args.seed)
        indices = np.arange(len(obs))
        rng.shuffle(indices)
        split = int(len(indices) * (1.0 - args.val_fraction))
        train_idx, val_idx = indices[:split], indices[split:]

    tensors = [
        torch.as_tensor(obs, dtype=torch.float32),
        torch.as_tensor(death, dtype=torch.float32),
        torch.as_tensor(escape, dtype=torch.float32),
        torch.as_tensor(escape_available, dtype=torch.float32),
        torch.as_tensor(bomb_escape_available, dtype=torch.float32),
        torch.as_tensor(trapped_if_bomb, dtype=torch.float32),
        torch.as_tensor(future_blast, dtype=torch.float32),
        torch.as_tensor(boxes, dtype=torch.float32),
        torch.as_tensor(reachable, dtype=torch.float32),
        torch.as_tensor(safe_tiles, dtype=torch.float32),
        torch.as_tensor(blast_distance, dtype=torch.float32),
        torch.as_tensor(returns, dtype=torch.float32),
    ]
    train_ds = TensorDataset(*(tensor[train_idx] for tensor in tensors))
    val_ds = TensorDataset(*(tensor[val_idx] for tensor in tensors))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = BomberAuxModel(features_dim=args.features_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    pos_death = torch.tensor([(len(train_idx) - death[train_idx].sum()) / max(1.0, death[train_idx].sum())], device=device)
    pos_escape = torch.tensor([(len(train_idx) - escape[train_idx].sum()) / max(1.0, escape[train_idx].sum())], device=device)
    pos_escape_available = torch.tensor([(len(train_idx) - escape_available[train_idx].sum()) / max(1.0, escape_available[train_idx].sum())], device=device)
    pos_bomb_escape_available = torch.tensor([(len(train_idx) - bomb_escape_available[train_idx].sum()) / max(1.0, bomb_escape_available[train_idx].sum())], device=device)
    pos_trapped = torch.tensor([(len(train_idx) - trapped_if_bomb[train_idx].sum()) / max(1.0, trapped_if_bomb[train_idx].sum())], device=device)
    pos_future_blast = torch.tensor([(len(train_idx) - future_blast[train_idx].sum()) / max(1.0, future_blast[train_idx].sum())], device=device)
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = [item.to(device) for item in batch]
            (
                b_obs,
                b_death,
                b_escape,
                b_escape_available,
                b_bomb_escape_available,
                b_trapped,
                b_future_blast,
                b_boxes,
                b_reach,
                b_safe_tiles,
                b_blast_distance,
                b_returns,
            ) = batch
            out = model(b_obs)
            if args.focal_loss:
                death_loss = focal_bce_with_logits(out["death_logit"], b_death, pos_weight=pos_death, gamma=args.focal_gamma)
                escape_loss = focal_bce_with_logits(out["escape_logit"], b_escape, pos_weight=pos_escape, gamma=args.focal_gamma)
                escape_available_loss = focal_bce_with_logits(out["escape_available_logit"], b_escape_available, pos_weight=pos_escape_available, gamma=args.focal_gamma)
                bomb_escape_available_loss = focal_bce_with_logits(out["bomb_escape_available_logit"], b_bomb_escape_available, pos_weight=pos_bomb_escape_available, gamma=args.focal_gamma)
                trapped_loss = focal_bce_with_logits(out["trapped_if_bomb_logit"], b_trapped, pos_weight=pos_trapped, gamma=args.focal_gamma)
                future_blast_loss = focal_bce_with_logits(out["future_blast_logit"], b_future_blast, pos_weight=pos_future_blast, gamma=args.focal_gamma)
            else:
                death_loss = F.binary_cross_entropy_with_logits(out["death_logit"], b_death, pos_weight=pos_death)
                escape_loss = F.binary_cross_entropy_with_logits(out["escape_logit"], b_escape, pos_weight=pos_escape)
                escape_available_loss = F.binary_cross_entropy_with_logits(out["escape_available_logit"], b_escape_available, pos_weight=pos_escape_available)
                bomb_escape_available_loss = F.binary_cross_entropy_with_logits(out["bomb_escape_available_logit"], b_bomb_escape_available, pos_weight=pos_bomb_escape_available)
                trapped_loss = F.binary_cross_entropy_with_logits(out["trapped_if_bomb_logit"], b_trapped, pos_weight=pos_trapped)
                future_blast_loss = F.binary_cross_entropy_with_logits(out["future_blast_logit"], b_future_blast, pos_weight=pos_future_blast)
            box_loss = F.smooth_l1_loss(out["box_value"], b_boxes)
            reach_loss = F.smooth_l1_loss(out["reachable_delta"], b_reach)
            safe_tiles_loss = F.smooth_l1_loss(out["safe_tiles_after_bomb"], b_safe_tiles)
            blast_distance_loss = F.smooth_l1_loss(out["blast_corridor_distance"], b_blast_distance)
            return_loss = F.smooth_l1_loss(out["return"], b_returns)
            loss = (
                args.death_weight * death_loss
                + args.escape_weight * escape_loss
                + args.escape_available_weight * escape_available_loss
                + args.bomb_escape_available_weight * bomb_escape_available_loss
                + args.trapped_weight * trapped_loss
                + args.future_blast_weight * future_blast_loss
                + args.box_weight * box_loss
                + args.reachable_weight * reach_loss
                + args.safe_tiles_weight * safe_tiles_loss
                + args.blast_distance_weight * blast_distance_loss
                + args.return_weight * return_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(b_obs)

        metrics = evaluate_model(model, val_loader, device)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = train_loss / max(1, len(train_ds))
        history.append(metrics)
        print(json.dumps(metrics))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "features_dim": args.features_dim,
            "normalization": {
                "boxes_mean": box_mean,
                "boxes_std": box_std,
                "reachable_mean": reach_mean,
                "reachable_std": reach_std,
                "safe_tiles_mean": safe_tiles_mean,
                "safe_tiles_std": safe_tiles_std,
                "blast_distance_mean": blast_distance_mean,
                "blast_distance_std": blast_distance_std,
                "return_mean": ret_mean,
                "return_std": ret_std,
            },
            "metrics": history[-1] if history else {},
            "history": history,
        },
        output,
    )
    output.with_suffix(".json").write_text(json.dumps({"history": history, "final": history[-1] if history else {}}, indent=2), encoding="utf-8")
    thresholds = {
        "death": _recommend_threshold(history[-1]["death"] if history else {}),
        "escape": _recommend_threshold(history[-1]["escape"] if history else {}, prefer_recall=True),
        "escape_available": _recommend_threshold(history[-1]["escape_available"] if history else {}, prefer_recall=True),
        "bomb_escape_available": _recommend_threshold(history[-1]["bomb_escape_available"] if history else {}, prefer_recall=True),
        "trapped_if_bomb": _recommend_threshold(history[-1]["trapped_if_bomb"] if history else {}),
        "future_blast": _recommend_threshold(history[-1]["future_blast"] if history else {}),
    }
    Path(args.thresholds_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.thresholds_output).write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    print(f"Saved auxiliary model: {output}")


def _recommend_threshold(metrics: dict, prefer_recall: bool = False) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in (0.3, 0.5, 0.7, 0.9):
        precision = float(metrics.get(f"precision@{threshold}", 0.0))
        recall = float(metrics.get(f"recall@{threshold}", 0.0))
        score = (0.5 * precision + recall) if prefer_recall else (precision + 0.5 * recall)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def evaluate_model(model, loader, device):
    model.eval()
    all_death_logits = []
    all_escape_logits = []
    all_escape_available_logits = []
    all_bomb_escape_available_logits = []
    all_trapped_logits = []
    all_future_blast_logits = []
    all_death = []
    all_escape = []
    all_escape_available = []
    all_bomb_escape_available = []
    all_trapped = []
    all_future_blast = []
    box_abs = []
    reach_abs = []
    safe_tiles_abs = []
    blast_distance_abs = []
    ret_abs = []
    val_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = [item.to(device) for item in batch]
            (
                b_obs,
                b_death,
                b_escape,
                b_escape_available,
                b_bomb_escape_available,
                b_trapped,
                b_future_blast,
                b_boxes,
                b_reach,
                b_safe_tiles,
                b_blast_distance,
                b_returns,
            ) = batch
            out = model(b_obs)
            val_loss += float(
                F.binary_cross_entropy_with_logits(out["death_logit"], b_death)
                + F.binary_cross_entropy_with_logits(out["escape_logit"], b_escape)
                + F.binary_cross_entropy_with_logits(out["escape_available_logit"], b_escape_available)
                + F.binary_cross_entropy_with_logits(out["bomb_escape_available_logit"], b_bomb_escape_available)
                + F.binary_cross_entropy_with_logits(out["trapped_if_bomb_logit"], b_trapped)
                + F.binary_cross_entropy_with_logits(out["future_blast_logit"], b_future_blast)
                + F.smooth_l1_loss(out["box_value"], b_boxes)
                + F.smooth_l1_loss(out["reachable_delta"], b_reach)
                + F.smooth_l1_loss(out["safe_tiles_after_bomb"], b_safe_tiles)
                + F.smooth_l1_loss(out["blast_corridor_distance"], b_blast_distance)
                + F.smooth_l1_loss(out["return"], b_returns)
            ) * len(b_obs)
            count += len(b_obs)
            all_death_logits.append(out["death_logit"].cpu())
            all_escape_logits.append(out["escape_logit"].cpu())
            all_escape_available_logits.append(out["escape_available_logit"].cpu())
            all_bomb_escape_available_logits.append(out["bomb_escape_available_logit"].cpu())
            all_trapped_logits.append(out["trapped_if_bomb_logit"].cpu())
            all_future_blast_logits.append(out["future_blast_logit"].cpu())
            all_death.append(b_death.cpu())
            all_escape.append(b_escape.cpu())
            all_escape_available.append(b_escape_available.cpu())
            all_bomb_escape_available.append(b_bomb_escape_available.cpu())
            all_trapped.append(b_trapped.cpu())
            all_future_blast.append(b_future_blast.cpu())
            box_abs.append(torch.abs(out["box_value"] - b_boxes).cpu())
            reach_abs.append(torch.abs(out["reachable_delta"] - b_reach).cpu())
            safe_tiles_abs.append(torch.abs(out["safe_tiles_after_bomb"] - b_safe_tiles).cpu())
            blast_distance_abs.append(torch.abs(out["blast_corridor_distance"] - b_blast_distance).cpu())
            ret_abs.append(torch.abs(out["return"] - b_returns).cpu())
    death_metrics = _binary_metrics(torch.cat(all_death_logits), torch.cat(all_death))
    escape_metrics = _binary_metrics(torch.cat(all_escape_logits), torch.cat(all_escape))
    return {
        "val_loss": val_loss / max(1, count),
        "death": death_metrics,
        "escape": escape_metrics,
        "escape_available": _binary_metrics(torch.cat(all_escape_available_logits), torch.cat(all_escape_available)),
        "bomb_escape_available": _binary_metrics(torch.cat(all_bomb_escape_available_logits), torch.cat(all_bomb_escape_available)),
        "trapped_if_bomb": _binary_metrics(torch.cat(all_trapped_logits), torch.cat(all_trapped)),
        "future_blast": _binary_metrics(torch.cat(all_future_blast_logits), torch.cat(all_future_blast)),
        "box_value_mae_norm": float(torch.cat(box_abs).mean()),
        "reachable_delta_mae_norm": float(torch.cat(reach_abs).mean()),
        "safe_tiles_after_bomb_mae_norm": float(torch.cat(safe_tiles_abs).mean()),
        "blast_corridor_distance_mae_norm": float(torch.cat(blast_distance_abs).mean()),
        "return_mae_norm": float(torch.cat(ret_abs).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Train standalone auxiliary heads on curriculum rollouts.")
    parser.add_argument("--dataset", default="ml/datasets/curriculum_rollouts.npz")
    parser.add_argument("--output", default="ml/checkpoints/rl_agent_pure/aux_curriculum_model.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--features_dim", type=int, default=256)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=6200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--death_weight", type=float, default=1.0)
    parser.add_argument("--escape_weight", type=float, default=1.0)
    parser.add_argument("--box_weight", type=float, default=0.5)
    parser.add_argument("--reachable_weight", type=float, default=0.25)
    parser.add_argument("--return_weight", type=float, default=0.25)
    parser.add_argument("--escape_available_weight", type=float, default=1.0)
    parser.add_argument("--bomb_escape_available_weight", type=float, default=1.0)
    parser.add_argument("--trapped_weight", type=float, default=1.0)
    parser.add_argument("--future_blast_weight", type=float, default=1.0)
    parser.add_argument("--safe_tiles_weight", type=float, default=0.25)
    parser.add_argument("--blast_distance_weight", type=float, default=0.25)
    parser.add_argument("--focal_loss", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--thresholds_output", default="ml/checkpoints/rl_agent_pure/aux_thresholds_v2.json")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
