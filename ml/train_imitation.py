"""Train a tiny supervised imitation policy from replay datasets."""

import argparse
import json
from pathlib import Path

import numpy as np

from ml.features import CHANNEL_NAMES
from ml.models.simple_cnn_policy import TORCH_AVAILABLE, build_model


if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
else:  # pragma: no cover - depends on local env
    torch = None
    DataLoader = None
    TensorDataset = None
    WeightedRandomSampler = None


ACTION_NAMES = ["STOP", "LEFT", "RIGHT", "UP", "DOWN", "PLACE_BOMB"]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _load_metadata(npz):
    raw = npz.get("metadata_json")
    if raw is None:
        return {}
    if hasattr(raw, "item"):
        raw = raw.item()
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {}


def _load_dataset(path):
    npz = np.load(path, allow_pickle=False)
    observations = npz["observations"].astype(np.float32)
    actions = npz["actions"].astype(np.int64)
    outcomes = npz["outcomes"].astype(str) if "outcomes" in npz else np.array(["unknown"] * len(actions))
    metadata = _load_metadata(npz)
    return observations, actions, outcomes, metadata


def _filter_dataset(observations, actions, outcomes, wins_only=False,
                    exclude_draws=False, max_samples=None):
    mask = np.ones(len(actions), dtype=bool)
    if wins_only:
        mask &= outcomes == "win"
    if exclude_draws:
        mask &= outcomes != "draw"
    indices = np.flatnonzero(mask)
    if max_samples is not None and len(indices) > max_samples:
        rng = np.random.default_rng(42)
        indices = np.sort(rng.choice(indices, size=int(max_samples), replace=False))
    return observations[indices], actions[indices], outcomes[indices]


def _action_counts(actions):
    return np.bincount(actions, minlength=len(ACTION_NAMES))


def _print_action_distribution(actions, prefix=""):
    counts = _action_counts(actions)
    total = max(1, int(counts.sum()))
    print(f"{prefix}action distribution:")
    for idx, count in enumerate(counts):
        pct = 100.0 * float(count) / float(total)
        print(f"  {idx} {ACTION_NAMES[idx]}: {int(count)} ({pct:.1f}%)")
    if counts.max() > total * 0.45:
        print("WARNING: dominant action class is very large.")
    if counts[5] < total * 0.02:
        print("WARNING: PLACE_BOMB is underrepresented.")
    return counts


def _split_indices(total, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)
    val_size = max(1, int(total * val_ratio))
    return indices[val_size:], indices[:val_size]


def _metrics(logits, targets, loss_fn):
    loss = loss_fn(logits, targets).item()
    preds = logits.argmax(dim=1)
    accuracy = (preds == targets).float().mean().item()
    top2 = logits.topk(k=min(2, logits.shape[1]), dim=1).indices
    top2_accuracy = (top2 == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    return loss, accuracy, top2_accuracy


def _evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_top2 = 0
    total_seen = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = loss_fn(logits, labels)
            batch_size = int(labels.shape[0])
            preds = logits.argmax(dim=1)
            top2 = logits.topk(k=min(2, logits.shape[1]), dim=1).indices
            total_loss += float(loss.item()) * batch_size
            total_correct += int((preds == labels).sum().item())
            total_top2 += int((top2 == labels.unsqueeze(1)).any(dim=1).sum().item())
            total_seen += batch_size
    denom = max(1, total_seen)
    return total_loss / denom, total_correct / denom, total_top2 / denom


def train_imitation(args):
    """Train a tiny imitation model, saving the best validation checkpoint."""
    if not TORCH_AVAILABLE:
        print("PyTorch not installed, skipping training.")
        return None

    observations, actions, outcomes, metadata = _load_dataset(args.dataset)
    observations, actions, outcomes = _filter_dataset(
        observations,
        actions,
        outcomes,
        wins_only=args.wins_only,
        exclude_draws=args.exclude_draws,
        max_samples=args.max_samples,
    )
    if len(actions) < 10:
        raise RuntimeError("Not enough samples after filtering to train.")

    if observations.shape[1:] != (12, 13, 13):
        raise RuntimeError(f"Expected observations shaped (N, 12, 13, 13), got {observations.shape}")

    print(f"dataset: {args.dataset}")
    print(f"samples: {len(actions)}")
    print(f"observation shape: {observations.shape}")
    print(f"action shape: {actions.shape}")
    counts = _print_action_distribution(actions)

    train_idx, val_idx = _split_indices(len(actions), seed=args.seed)
    x_train = torch.from_numpy(observations[train_idx])
    y_train = torch.from_numpy(actions[train_idx])
    x_val = torch.from_numpy(observations[val_idx])
    y_val = torch.from_numpy(actions[val_idx])

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    class_counts = torch.tensor(_action_counts(actions), dtype=torch.float32)
    class_weights = class_counts.sum() / (len(ACTION_NAMES) * torch.clamp(class_counts, min=1.0))
    loss_weights = None if args.balanced_actions else class_weights
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights)

    sampler = None
    shuffle = True
    if args.balanced_actions:
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        shuffle = False
        print("imbalance handling: weighted sampler")
    else:
        print("imbalance handling: weighted cross-entropy loss")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cpu")
    model = build_model(input_channels=12, num_actions=len(ACTION_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_metrics = None
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_seen = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = int(labels.shape[0])
            running_loss += float(loss.item()) * batch_size
            running_seen += batch_size

        train_loss = running_loss / max(1, running_seen)
        val_loss, val_acc, val_top2 = _evaluate(model, val_loader, loss_fn, device)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} top2_acc={val_top2:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "top2_accuracy": val_top2,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_channels": 12,
                    "num_actions": len(ACTION_NAMES),
                    "channel_names": list(CHANNEL_NAMES),
                    "action_names": list(ACTION_NAMES),
                    "action_counts": counts.tolist(),
                    "dataset_path": str(args.dataset),
                    "metadata": metadata,
                    "best_metrics": best_metrics,
                    "filters": {
                        "wins_only": args.wins_only,
                        "exclude_draws": args.exclude_draws,
                        "max_samples": args.max_samples,
                        "balanced_actions": args.balanced_actions,
                    },
                },
                output_path,
            )

    print(f"saved best checkpoint: {output_path}")
    if best_metrics:
        print(
            "best validation: "
            f"epoch={best_metrics['epoch']} "
            f"loss={best_metrics['val_loss']:.4f} "
            f"acc={best_metrics['val_accuracy']:.4f} "
            f"top2={best_metrics['top2_accuracy']:.4f}"
        )
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train a tiny Bomberland imitation policy.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wins_only", type=str2bool, default=False)
    parser.add_argument("--exclude_draws", type=str2bool, default=False)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--balanced_actions", type=str2bool, default=False)
    args = parser.parse_args()
    train_imitation(args)


if __name__ == "__main__":
    main()
