"""Train a neural prior that ranks only safe heuristic actions."""

import argparse
import json
from pathlib import Path

import numpy as np

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


def split_indices(total, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)
    val_size = max(1, int(total * val_ratio))
    return indices[val_size:], indices[:val_size]


def masked_cross_entropy(logits, targets, masks):
    masked_logits = logits.masked_fill(~masks, -1e9)
    return torch.nn.functional.cross_entropy(masked_logits, targets)


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

    best_loss = float("inf")
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
            loss = masked_cross_entropy(logits, batch_targets, batch_masks)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * int(batch_targets.shape[0])
            train_seen += int(batch_targets.shape[0])

        metrics = evaluate_model(model, val_loader, device)
        avg_train_loss = train_loss / max(1, train_seen)
        print(
            f"epoch {epoch}: train_loss={avg_train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} rank_acc={metrics['ranking_accuracy']:.4f} "
            f"top2_safe={metrics['top2_safe_agreement']:.4f} entropy={metrics['entropy_normalized']:.4f}"
        )
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
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
    args = parser.parse_args()
    train_action_ranker(args)


if __name__ == "__main__":
    main()
