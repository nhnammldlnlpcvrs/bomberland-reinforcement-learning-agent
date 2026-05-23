"""Analyze behavioral bias in a trained imitation policy."""

import argparse

import numpy as np

from ml.models.simple_cnn_policy import TORCH_AVAILABLE, load_checkpoint
from ml.train_imitation import ACTION_NAMES


if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
else:  # pragma: no cover - environment dependent
    torch = None
    DataLoader = None
    TensorDataset = None


def _pct(count, total):
    return 0.0 if total <= 0 else 100.0 * float(count) / float(total)


def _load_dataset(path, max_samples=None, seed=42):
    data = np.load(path, allow_pickle=False)
    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    if max_samples is not None and len(actions) > max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(np.arange(len(actions)), size=int(max_samples), replace=False))
        observations = observations[indices]
        actions = actions[indices]
    return observations, actions


def _confusion_matrix(actual, predicted):
    matrix = np.zeros((len(ACTION_NAMES), len(ACTION_NAMES)), dtype=np.int64)
    for truth, pred in zip(actual, predicted):
        if 0 <= truth < len(ACTION_NAMES) and 0 <= pred < len(ACTION_NAMES):
            matrix[int(truth), int(pred)] += 1
    return matrix


def _print_confusion(matrix):
    print("actual\\pred " + " ".join(f"{idx:>7}" for idx in range(len(ACTION_NAMES))))
    for idx, row in enumerate(matrix):
        print(f"{idx:>4} {ACTION_NAMES[idx]:<10} " + " ".join(f"{int(value):>7}" for value in row))


def analyze_policy_bias(args):
    if not TORCH_AVAILABLE:
        print("PyTorch not installed, skipping policy bias analysis.")
        return None

    observations, actions = _load_dataset(args.dataset, max_samples=args.max_samples, seed=args.seed)
    model, _checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(observations), torch.from_numpy(actions)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    predictions = []
    probabilities = []
    with torch.no_grad():
        for features, _labels in loader:
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs.argmax(dim=1).cpu().numpy())
            probabilities.append(probs.cpu().numpy())

    predicted = np.concatenate(predictions, axis=0)
    probs = np.concatenate(probabilities, axis=0)
    total = len(predicted)
    pred_counts = np.bincount(predicted, minlength=len(ACTION_NAMES))
    actual_counts = np.bincount(actions, minlength=len(ACTION_NAMES))
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)), axis=1)
    max_entropy = float(np.log(len(ACTION_NAMES)))
    normalized_entropy = entropy / max_entropy
    matrix = _confusion_matrix(actions, predicted)

    print("=== Policy Bias Analysis ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"dataset: {args.dataset}")
    print(f"samples: {total}")
    print(f"average entropy: {float(np.mean(entropy)):.4f}")
    print(f"normalized entropy: {float(np.mean(normalized_entropy)):.4f}")

    print("\n=== Actual Action Distribution ===")
    for idx, name in enumerate(ACTION_NAMES):
        print(f"{idx} {name}: {int(actual_counts[idx])} ({_pct(actual_counts[idx], total):.1f}%)")

    print("\n=== Prediction Distribution ===")
    for idx, name in enumerate(ACTION_NAMES):
        print(f"{idx} {name}: {int(pred_counts[idx])} ({_pct(pred_counts[idx], total):.1f}%)")

    movement_total = int(pred_counts[1:5].sum())
    vertical = int(pred_counts[3] + pred_counts[4])
    horizontal = int(pred_counts[1] + pred_counts[2])
    print("\n=== Movement Diversity ===")
    print(f"movement predictions: {movement_total} ({_pct(movement_total, total):.1f}%)")
    print(f"horizontal movement: {horizontal} ({_pct(horizontal, max(1, movement_total)):.1f}% of moves)")
    print(f"vertical movement: {vertical} ({_pct(vertical, max(1, movement_total)):.1f}% of moves)")

    print("\n=== Confusion Matrix ===")
    _print_confusion(matrix)
    stop_row = matrix[0]
    bomb_row = matrix[5]
    print("\n=== STOP / PLACE_BOMB Confusion ===")
    print(f"STOP correctly predicted: {int(stop_row[0])}/{int(stop_row.sum())}")
    print(f"PLACE_BOMB correctly predicted: {int(bomb_row[5])}/{int(bomb_row.sum())}")

    print("\n=== Warnings ===")
    warnings = []
    if _pct(pred_counts[0], total) > 35.0:
        warnings.append("passive policy: STOP predicted too often")
    if _pct(pred_counts[5], total) < 1.0:
        warnings.append("PLACE_BOMB nearly absent from predictions")
    if _pct(pred_counts[5], total) > 20.0:
        warnings.append("PLACE_BOMB predicted too often; possible bomb-spam bias")
    if float(np.mean(normalized_entropy)) < 0.55:
        warnings.append("low entropy: policy may be overconfident or collapsed")
    if _pct(pred_counts.max(), total) > 60.0:
        warnings.append("output distribution highly skewed")
    if movement_total and (horizontal / movement_total > 0.85 or vertical / movement_total > 0.85):
        warnings.append("directional collapse in movement predictions")
    if not warnings:
        print("none")
    else:
        for warning in warnings:
            print(f"WARNING: {warning}")

    return {
        "prediction_counts": pred_counts,
        "actual_counts": actual_counts,
        "entropy": float(np.mean(entropy)),
        "normalized_entropy": float(np.mean(normalized_entropy)),
        "confusion": matrix,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze imitation policy behavioral bias.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze_policy_bias(args)


if __name__ == "__main__":
    main()
