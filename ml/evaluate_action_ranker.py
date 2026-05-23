"""Evaluate a neural prior over safe heuristic actions."""

import argparse
import numpy as np

from ml.models.simple_cnn_policy import TORCH_AVAILABLE, load_checkpoint
from ml.train_action_ranker import load_ranking_dataset, ranking_metrics
from ml.train_imitation import ACTION_NAMES


if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
else:  # pragma: no cover - depends on environment
    torch = None
    DataLoader = None
    TensorDataset = None


def evaluate_action_ranker(args):
    if not TORCH_AVAILABLE:
        print("PyTorch not installed, skipping action ranker evaluation.")
        return None

    observations, targets, masks, _metadata = load_ranking_dataset(
        args.dataset,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    model, checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(observations),
            torch.from_numpy(targets),
            torch.from_numpy(masks),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    logits_all = []
    targets_all = []
    masks_all = []
    with torch.no_grad():
        for obs, batch_targets, batch_masks in loader:
            logits_all.append(model(obs).cpu())
            targets_all.append(batch_targets.cpu())
            masks_all.append(batch_masks.cpu())

    logits = torch.cat(logits_all)
    target_tensor = torch.cat(targets_all)
    mask_tensor = torch.cat(masks_all)
    metrics = ranking_metrics(logits, target_tensor, mask_tensor)
    masked_logits = logits.masked_fill(~mask_tensor, -1e9)
    preds = masked_logits.argmax(dim=1).numpy()
    target_np = target_tensor.numpy()
    mask_np = mask_tensor.numpy()

    pred_counts = np.bincount(preds, minlength=len(ACTION_NAMES))
    target_counts = np.bincount(target_np, minlength=len(ACTION_NAMES))
    safe_counts = mask_np.sum(axis=0).astype(np.int64)
    total = len(preds)

    print("=== Action Ranker Evaluation ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"dataset: {args.dataset}")
    print(f"samples: {total}")
    print(f"ranking_accuracy: {metrics['ranking_accuracy']:.4f}")
    print(f"top2_safe_agreement: {metrics['top2_safe_agreement']:.4f}")
    print(f"entropy_normalized: {metrics['entropy_normalized']:.4f}")
    print(f"checkpoint best_metrics: {checkpoint.get('best_metrics')}")

    print("\nTarget vs prediction distribution:")
    for idx, name in enumerate(ACTION_NAMES):
        target_pct = 100.0 * target_counts[idx] / max(1, total)
        pred_pct = 100.0 * pred_counts[idx] / max(1, total)
        safe_pct = 100.0 * safe_counts[idx] / max(1, total)
        print(
            f"  {idx} {name}: target={target_pct:.1f}% "
            f"pred={pred_pct:.1f}% safe_available={safe_pct:.1f}%"
        )

    safe_bomb_available = safe_counts[5] / max(1, total)
    bomb_preference = pred_counts[5] / max(1, safe_counts[5]) if safe_counts[5] else 0.0
    movement_preds = pred_counts[1:5].sum()
    movement_diversity = np.count_nonzero(pred_counts[1:5] > 0) / 4.0
    print("\nBehavior metrics:")
    print(f"safe_bomb_available: {safe_bomb_available * 100.0:.1f}%")
    print(f"bomb_preference_when_safe: {bomb_preference * 100.0:.1f}%")
    print(f"movement_prediction_diversity: {movement_diversity * 100.0:.1f}%")
    print(f"movement_predictions: {100.0 * movement_preds / max(1, total):.1f}%")

    warnings = []
    stop_pct = 100.0 * pred_counts[0] / max(1, total)
    bomb_pct = 100.0 * pred_counts[5] / max(1, total)
    if stop_pct > 35.0:
        warnings.append("STOP-heavy neural prior")
    if safe_counts[5] and bomb_preference > 0.35:
        warnings.append("bomb-heavy neural prior")
    if safe_counts[5] and bomb_preference < 0.02:
        warnings.append("neural prior almost never ranks safe bombs first")
    if movement_preds and pred_counts[1:5].max() / movement_preds > 0.70:
        warnings.append("directional movement collapse")

    print("\nWarnings:")
    if warnings:
        for warning in warnings:
            print(f"WARNING: {warning}")
    else:
        print("none")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a safe-action neural prior ranker.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate_action_ranker(args)


if __name__ == "__main__":
    main()
