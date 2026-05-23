"""Evaluate a tiny imitation policy checkpoint on a replay dataset."""

import argparse

import numpy as np

from ml.models.simple_cnn_policy import TORCH_AVAILABLE, load_checkpoint
from ml.train_imitation import ACTION_NAMES


if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
else:  # pragma: no cover - depends on local env
    torch = None
    DataLoader = None
    TensorDataset = None


def _load_dataset(path, max_samples=None):
    npz = np.load(path, allow_pickle=False)
    observations = npz["observations"].astype(np.float32)
    actions = npz["actions"].astype(np.int64)
    if max_samples is not None and len(actions) > max_samples:
        observations = observations[:max_samples]
        actions = actions[:max_samples]
    return observations, actions


def _print_confusion(confusion):
    header = "actual\\pred " + " ".join(f"{idx:>7}" for idx in range(len(ACTION_NAMES)))
    print(header)
    for idx, row in enumerate(confusion):
        values = " ".join(f"{int(value):>7}" for value in row)
        print(f"{idx:>4} {ACTION_NAMES[idx]:<10} {values}")


def evaluate(args):
    if not TORCH_AVAILABLE:
        print("PyTorch not installed, skipping evaluation.")
        return None

    observations, actions = _load_dataset(args.dataset, max_samples=args.max_samples)
    model, checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    dataset = TensorDataset(torch.from_numpy(observations), torch.from_numpy(actions))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    confusion = np.zeros((len(ACTION_NAMES), len(ACTION_NAMES)), dtype=np.int64)
    prediction_counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    total = 0
    correct = 0
    top2_correct = 0

    with torch.no_grad():
        for features, labels in loader:
            logits = model(features)
            preds = logits.argmax(dim=1)
            top2 = logits.topk(k=min(2, logits.shape[1]), dim=1).indices
            for actual, pred in zip(labels.numpy(), preds.numpy()):
                if 0 <= actual < len(ACTION_NAMES) and 0 <= pred < len(ACTION_NAMES):
                    confusion[int(actual), int(pred)] += 1
                    prediction_counts[int(pred)] += 1
            correct += int((preds == labels).sum().item())
            top2_correct += int((top2 == labels.unsqueeze(1)).any(dim=1).sum().item())
            total += int(labels.shape[0])

    denom = max(1, total)
    accuracy = correct / denom
    top2_accuracy = top2_correct / denom
    print(f"checkpoint: {args.checkpoint}")
    print(f"dataset: {args.dataset}")
    print(f"samples: {total}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"top2_accuracy: {top2_accuracy:.4f}")
    best_metrics = checkpoint.get("best_metrics")
    if best_metrics:
        print(f"checkpoint best_metrics: {best_metrics}")

    print("confusion matrix:")
    _print_confusion(confusion)

    print("per-action accuracy:")
    for idx, name in enumerate(ACTION_NAMES):
        row_total = int(confusion[idx].sum())
        row_correct = int(confusion[idx, idx])
        value = row_correct / max(1, row_total)
        print(f"  {idx} {name}: {value:.4f} ({row_correct}/{row_total})")

    print("prediction distribution:")
    for idx, name in enumerate(ACTION_NAMES):
        pct = 100.0 * float(prediction_counts[idx]) / float(denom)
        print(f"  {idx} {name}: {int(prediction_counts[idx])} ({pct:.1f}%)")

    if prediction_counts[0] > denom * 0.35:
        print("WARNING: model predicts STOP very often.")
    if prediction_counts[5] < denom * 0.005:
        print("WARNING: model almost never predicts PLACE_BOMB.")
    if prediction_counts.max() > denom * 0.60:
        print("WARNING: prediction distribution collapsed toward one action.")

    return {
        "accuracy": accuracy,
        "top2_accuracy": top2_accuracy,
        "confusion": confusion,
        "prediction_counts": prediction_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a tiny Bomberland imitation policy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
