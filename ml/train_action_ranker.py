"""Train a neural prior that ranks only safe heuristic actions."""

import argparse
import json
from pathlib import Path

import numpy as np

from ml.augment_dataset import _augment_masks, _augment_observations, _remap_action_array
from ml.dataset_builder import str2bool
from ml.models.simple_cnn_policy import TORCH_AVAILABLE, build_model
from ml.train_imitation import ACTION_NAMES


if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
else:  # pragma: no cover - depends on environment
    torch = None
    DataLoader = None
    TensorDataset = None


def _load_metadata(data):
    raw = data.get("metadata_json")
    if raw is None:
        return {}
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {}


def load_ranking_dataset(path, max_samples=None, seed=42):
    data = np.load(path, allow_pickle=False)
    observations = data["observations"].astype(np.float32)
    targets = data["target_actions"].astype(np.int64)
    masks = data["safe_action_masks"].astype(bool)
    metadata = _load_metadata(data)
    if max_samples is not None and len(targets) > max_samples:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(np.arange(len(targets)), size=int(max_samples), replace=False))
        observations = observations[idx]
        targets = targets[idx]
        masks = masks[idx]
    return observations, targets, masks, metadata


def augment_ranking_arrays(observations, targets, masks, modes=("hflip", "vflip", "rot180")):
    obs_parts = [observations]
    target_parts = [targets]
    mask_parts = [masks]
    for mode in modes:
        obs_parts.append(_augment_observations(observations, mode))
        target_parts.append(_remap_action_array(targets, mode))
        mask_parts.append(_augment_masks(masks, mode))
    return (
        np.concatenate(obs_parts, axis=0).astype(np.float32),
        np.concatenate(target_parts, axis=0).astype(np.int64),
        np.concatenate(mask_parts, axis=0).astype(bool),
    )


def split_indices(total, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)
    val_size = max(1, int(total * val_ratio))
    return indices[val_size:], indices[:val_size]


def masked_cross_entropy(logits, targets, masks):
    masked_logits = logits.masked_fill(~masks, -1e9)
    return torch.nn.functional.cross_entropy(masked_logits, targets)


def plain_cross_entropy(logits, targets, masks):
    del masks
    return torch.nn.functional.cross_entropy(logits, targets)


def masked_label_smoothing_loss(logits, targets, masks, smoothing=0.05):
    masked_logits = logits.masked_fill(~masks, -1e9)
    log_probs = torch.nn.functional.log_softmax(masked_logits, dim=1)
    valid_counts = masks.sum(dim=1).clamp(min=1).float()
    smooth = torch.zeros_like(logits)
    smooth = smooth.masked_fill(masks, 1.0)
    smooth = smooth / valid_counts.unsqueeze(1)
    hard = torch.zeros_like(logits)
    hard.scatter_(1, targets.unsqueeze(1), 1.0)
    single_action = valid_counts <= 1
    target_dist = (1.0 - float(smoothing)) * hard + float(smoothing) * smooth
    target_dist[single_action] = hard[single_action]
    return -torch.sum(target_dist * log_probs, dim=1).mean()


def ranking_loss(logits, targets, masks, loss_mode="masked_ce", label_smoothing=0.05):
    if loss_mode == "ce":
        return plain_cross_entropy(logits, targets, masks)
    if loss_mode == "masked_ce":
        return masked_cross_entropy(logits, targets, masks)
    if loss_mode == "label_smoothing":
        return masked_label_smoothing_loss(logits, targets, masks, smoothing=label_smoothing)
    raise ValueError(f"unknown loss_mode: {loss_mode}")


def behavior_regularization_loss(logits, targets, masks, target_bomb_pred_min=0.03,
                                 target_bomb_pred_max=0.10,
                                 target_stop_pred_max=0.30,
                                 stop_margin=0.05):
    masked_logits = logits.masked_fill(~masks, -1e9)
    probs = torch.softmax(masked_logits, dim=1)
    stop_mean = probs[:, 0].mean()
    bomb_mean = probs[:, 5].mean()
    movement = probs[:, 1:5].mean(dim=0)
    movement_total = movement.sum().clamp(min=1e-6)
    direction_share = movement / movement_total
    max_direction = direction_share.max()
    penalty = torch.relu(stop_mean - float(target_stop_pred_max)) ** 2
    penalty = penalty + torch.relu(float(target_bomb_pred_min) - bomb_mean) ** 2
    penalty = penalty + torch.relu(bomb_mean - float(target_bomb_pred_max)) ** 2
    penalty = penalty + torch.relu(max_direction - 0.50) ** 2
    non_stop = (targets != 0) & masks[:, 0]
    if torch.any(non_stop):
        stop_logits = logits[non_stop, 0]
        target_logits = logits[non_stop].gather(1, targets[non_stop].unsqueeze(1)).squeeze(1)
        penalty = penalty + torch.mean(torch.relu(stop_logits - target_logits + float(stop_margin)) ** 2)
    return penalty


def behavior_score_from_counts(metrics, target_bomb_pred_min=0.03,
                               target_bomb_pred_max=0.10,
                               target_stop_pred_max=0.30):
    pred_counts = np.asarray(metrics["prediction_counts"], dtype=np.float64)
    total = max(1.0, float(pred_counts.sum()))
    pred_pct = pred_counts / total
    movement = pred_counts[1:5]
    movement_total = max(1.0, float(movement.sum()))
    max_direction = float(movement.max() / movement_total) if movement_total else 0.0

    penalties = 0.0
    warnings = []
    bomb_pct = float(pred_pct[5])
    stop_pct = float(pred_pct[0])
    if bomb_pct < target_bomb_pred_min:
        penalties += min(0.30, target_bomb_pred_min - bomb_pct)
        warnings.append("BOMB_BELOW_RANGE")
    if bomb_pct > target_bomb_pred_max:
        penalties += min(0.40, bomb_pct - target_bomb_pred_max)
        warnings.append("BOMB_ABOVE_RANGE")
    if stop_pct > target_stop_pred_max:
        penalties += min(0.35, stop_pct - target_stop_pred_max)
        warnings.append("STOP_OVERUSE")
    if max_direction > 0.50:
        penalties += min(0.35, max_direction - 0.50)
        warnings.append("DIRECTION_COLLAPSE")

    score = (
        float(metrics["top2_safe_agreement"])
        + 0.1 * float(metrics["entropy_normalized"])
        - penalties
    )
    return {
        "behavior_score": float(score),
        "penalties": float(penalties),
        "warnings": warnings,
        "stop_pred_pct": float(stop_pct * 100.0),
        "bomb_pred_pct": float(bomb_pct * 100.0),
        "max_direction_pct": float(max_direction * 100.0),
    }


def ranking_metrics(logits, targets, masks):
    masked_logits = logits.masked_fill(~masks, -1e9)
    probs = torch.softmax(masked_logits, dim=1)
    preds = masked_logits.argmax(dim=1)
    top2 = masked_logits.topk(k=min(2, masked_logits.shape[1]), dim=1).indices
    valid_counts = masks.sum(dim=1).clamp(min=1)
    entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-9)), dim=1)
    max_entropy = torch.log(valid_counts.float()).clamp(min=1e-9)
    normalized_entropy = (entropy / max_entropy).mean().item()

    pred_counts = torch.bincount(preds.cpu(), minlength=len(ACTION_NAMES)).numpy()
    return {
        "ranking_accuracy": float((preds == targets).float().mean().item()),
        "top2_safe_agreement": float((top2 == targets.unsqueeze(1)).any(dim=1).float().mean().item()),
        "entropy_normalized": float(normalized_entropy),
        "prediction_counts": pred_counts,
        "bomb_prediction_pct": float((preds == 5).float().mean().item() * 100.0),
    }


def evaluate_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_seen = 0
    all_logits = []
    all_targets = []
    all_masks = []
    with torch.no_grad():
        for obs, targets, masks in loader:
            obs = obs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            logits = model(obs)
            loss = masked_cross_entropy(logits, targets, masks)
            total_loss += float(loss.item()) * int(targets.shape[0])
            total_seen += int(targets.shape[0])
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
            all_masks.append(masks.cpu())
    metrics = ranking_metrics(torch.cat(all_logits), torch.cat(all_targets), torch.cat(all_masks))
    metrics["loss"] = total_loss / max(1, total_seen)
    return metrics


def train_action_ranker(args):
    if not TORCH_AVAILABLE:
        print("PyTorch not installed, skipping action ranker training.")
        return None

    observations, targets, masks, metadata = load_ranking_dataset(
        args.dataset,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if args.augment_symmetry:
        observations, targets, masks = augment_ranking_arrays(observations, targets, masks)
        metadata = dict(metadata)
        metadata["train_augment_symmetry"] = True
    train_idx, val_idx = split_indices(len(targets), seed=args.seed)
    device = torch.device("cpu")

    x_train = torch.from_numpy(observations[train_idx])
    y_train = torch.from_numpy(targets[train_idx])
    m_train = torch.from_numpy(masks[train_idx])
    x_val = torch.from_numpy(observations[val_idx])
    y_val = torch.from_numpy(targets[val_idx])
    m_val = torch.from_numpy(masks[val_idx])

    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    train_loader = DataLoader(
        TensorDataset(x_train, y_train, m_train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val, m_val), batch_size=args.batch_size)

    torch.manual_seed(int(args.seed))
    model = build_model(input_channels=12, num_actions=len(ACTION_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_score = -float("inf")
    best_metrics = None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"dataset: {args.dataset}")
    print(f"samples: {len(targets)} train={len(train_idx)} val={len(val_idx)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_seen = 0
        for obs, batch_targets, batch_masks in train_loader:
            obs = obs.to(device)
            batch_targets = batch_targets.to(device)
            batch_masks = batch_masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(obs)
            loss = ranking_loss(
                logits,
                batch_targets,
                batch_masks,
                loss_mode=args.loss_mode,
                label_smoothing=args.label_smoothing,
            )
            if args.behavior_reg_weight > 0:
                loss = loss + float(args.behavior_reg_weight) * behavior_regularization_loss(
                    logits,
                    batch_targets,
                    batch_masks,
                    target_bomb_pred_min=args.target_bomb_pred_min,
                    target_bomb_pred_max=args.target_bomb_pred_max,
                    target_stop_pred_max=args.target_stop_pred_max,
                    stop_margin=args.stop_margin,
                )
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * int(batch_targets.shape[0])
            train_seen += int(batch_targets.shape[0])

        metrics = evaluate_model(model, val_loader, device)
        behavior = behavior_score_from_counts(
            metrics,
            target_bomb_pred_min=args.target_bomb_pred_min,
            target_bomb_pred_max=args.target_bomb_pred_max,
            target_stop_pred_max=args.target_stop_pred_max,
        )
        metrics.update(behavior)
        avg_train_loss = train_loss / max(1, train_seen)
        print(
            f"epoch {epoch}: train_loss={avg_train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} rank_acc={metrics['ranking_accuracy']:.4f} "
            f"top2_safe={metrics['top2_safe_agreement']:.4f} entropy={metrics['entropy_normalized']:.4f} "
            f"behavior_score={metrics['behavior_score']:.4f} warnings={','.join(metrics['warnings']) or 'none'}"
        )
        if metrics["behavior_score"] > best_score:
            best_score = metrics["behavior_score"]
            best_metrics = dict(metrics)
            if hasattr(best_metrics.get("prediction_counts"), "tolist"):
                best_metrics["prediction_counts"] = best_metrics["prediction_counts"].tolist()
            best_metrics["epoch"] = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_channels": 12,
                    "num_actions": len(ACTION_NAMES),
                    "action_names": list(ACTION_NAMES),
                    "dataset_path": str(args.dataset),
                    "metadata": metadata,
                    "best_metrics": best_metrics,
                    "mode": "safe_action_ranker",
                    "training_options": {
                        "loss_mode": args.loss_mode,
                        "label_smoothing": args.label_smoothing,
                        "target_bomb_pred_min": args.target_bomb_pred_min,
                        "target_bomb_pred_max": args.target_bomb_pred_max,
                        "target_stop_pred_max": args.target_stop_pred_max,
                        "behavior_reg_weight": args.behavior_reg_weight,
                        "stop_margin": args.stop_margin,
                        "augment_symmetry": args.augment_symmetry,
                    },
                },
                output,
            )

    print(f"saved best checkpoint: {output}")
    if best_metrics:
        print(f"best_metrics={best_metrics}")
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train a safe-action neural prior ranker.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment_symmetry", type=str2bool, default=False)
    parser.add_argument("--loss_mode", choices=("ce", "masked_ce", "label_smoothing"), default="masked_ce")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--target_bomb_pred_min", type=float, default=0.03)
    parser.add_argument("--target_bomb_pred_max", type=float, default=0.10)
    parser.add_argument("--target_stop_pred_max", type=float, default=0.30)
    parser.add_argument("--behavior_reg_weight", type=float, default=1.0)
    parser.add_argument("--stop_margin", type=float, default=0.05)
    args = parser.parse_args()
    train_action_ranker(args)


if __name__ == "__main__":
    main()
