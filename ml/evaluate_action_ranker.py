"""Evaluate a neural prior over safe heuristic actions."""

import argparse
import numpy as np

from ml.models.simple_cnn_policy import TORCH_AVAILABLE, load_checkpoint
from ml.train_action_ranker import (
    behavior_score_from_counts,
    load_ranking_dataset,
    ranking_metrics,
)
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
    behavior = behavior_score_from_counts(
        metrics,
        target_bomb_pred_min=args.target_bomb_pred_min,
        target_bomb_pred_max=args.target_bomb_pred_max,
        target_stop_pred_max=args.target_stop_pred_max,
    )
    metrics.update(behavior)
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
    print(f"behavior_score: {metrics['behavior_score']:.4f}")
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
    direction_total = max(1, int(movement_preds))
    print("\nBehavior metrics:")
    print(f"safe_bomb_available: {safe_bomb_available * 100.0:.1f}%")
    print(f"bomb_preference_when_safe: {bomb_preference * 100.0:.1f}%")
    print(f"movement_prediction_diversity: {movement_diversity * 100.0:.1f}%")
    print(f"movement_predictions: {100.0 * movement_preds / max(1, total):.1f}%")
    print("directional prediction distribution:")
    for idx in range(1, 5):
        print(f"  {ACTION_NAMES[idx]}: {100.0 * pred_counts[idx] / direction_total:.1f}% of moves")
    print(f"max_movement_direction: {metrics['max_direction_pct']:.1f}%")

    # --- Bomb-specific diagnostics ---
    print("\nBomb diagnostics:")

    bomb_candidate_rate = float(safe_counts[5] / max(1, total))
    bomb_target_rate = float(target_counts[5] / max(1, total))
    print(f"bomb_candidate_rate: {bomb_candidate_rate * 100.0:.1f}%")
    print(f"bomb_target_rate: {bomb_target_rate * 100.0:.1f}%")

    bomb_candidate_mask = mask_np[:, 5]
    bomb_candidate_indices = np.where(bomb_candidate_mask)[0]
    target_not_bomb_mask = target_np != 5

    if len(bomb_candidate_indices) > 0:
        bomb_top1_when_candidate = float(
            (preds[bomb_candidate_indices] == 5).sum() / len(bomb_candidate_indices)
        )
        print(f"bomb_top1_when_candidate: {bomb_top1_when_candidate * 100.0:.1f}%")
    else:
        bomb_top1_when_candidate = 0.0
        print("bomb_top1_when_candidate: N/A (no bomb candidates)")

    bomb_candidate_not_target = bomb_candidate_mask & target_not_bomb_mask
    n_bomb_candidate_not_target = int(bomb_candidate_not_target.sum())
    if n_bomb_candidate_not_target > 0:
        bomb_top1_when_not_target = float(
            (preds[bomb_candidate_not_target] == 5).sum() / n_bomb_candidate_not_target
        )
        print(f"bomb_top1_when_target_not_bomb: {bomb_top1_when_not_target * 100.0:.1f}%")
    else:
        bomb_top1_when_not_target = 0.0
        print("bomb_top1_when_target_not_bomb: N/A")

    # Raw BOMB top1 (unmasked)
    raw_preds = logits.argmax(dim=1).numpy()
    raw_bomb_top1 = float((raw_preds == 5).sum() / total)
    print(f"raw_bomb_top1 (unmasked): {raw_bomb_top1 * 100.0:.1f}%")

    # Mean BOMB logit gap vs max non-BOMB
    logits_np = logits.numpy()
    bomb_logits = logits_np[:, 5]
    non_bomb_logits = np.where(mask_np[:, :5], logits_np[:, :5], -1e9)
    max_non_bomb = non_bomb_logits.max(axis=1)
    bomb_gaps = bomb_logits - max_non_bomb
    mean_gap = float(bomb_gaps.mean())
    print(f"mean_bomb_logit_gap_vs_max_non_bomb: {mean_gap:+.4f}")

    # BOMB logit gap when target is bomb vs not
    bomb_target_mask = target_np == 5
    if bomb_target_mask.any():
        gap_when_target_bomb = float(bomb_gaps[bomb_target_mask].mean())
        print(f"bomb_logit_gap_when_target_is_bomb: {gap_when_target_bomb:+.4f}")
    if target_not_bomb_mask.any():
        gap_when_not_bomb = float(bomb_gaps[target_not_bomb_mask].mean())
        print(f"bomb_logit_gap_when_target_not_bomb: {gap_when_not_bomb:+.4f}")

    # --- Warnings ---
    warnings = []
    if raw_bomb_top1 > 0.90:
        warnings.append(f"RAW_BOMB_COLLAPSE: raw BOMB top1={raw_bomb_top1*100:.1f}% — unconstrained logit likely")
    stop_pct = 100.0 * pred_counts[0] / max(1, total)
    bomb_pct = 100.0 * pred_counts[5] / max(1, total)
    if stop_pct > 35.0:
        warnings.append("STOP-heavy neural prior")
    if safe_counts[5] and bomb_preference > 0.35:
        warnings.append("bomb-heavy neural prior")
    if safe_counts[5] and bomb_preference < 0.02:
        warnings.append("neural prior almost never ranks safe bombs first")
    if movement_preds and pred_counts[1:5].max() / movement_preds > 0.50:
        warnings.append("directional movement collapse")
    warnings.extend(metrics.get("warnings", []))

    unique_warnings = []
    for warning in warnings:
        if warning not in unique_warnings:
            unique_warnings.append(warning)

    print("\nWarnings:")
    if unique_warnings:
        for warning in unique_warnings:
            print(f"WARNING: {warning}")
    else:
        print("none")

    print("\nDeployability:")
    if (
        metrics["top2_safe_agreement"] >= 0.70
        and metrics["bomb_pred_pct"] >= 3.0
        and metrics["bomb_pred_pct"] <= 10.0
        and metrics["stop_pred_pct"] <= 35.0
        and metrics["max_direction_pct"] <= 55.0
        and metrics["entropy_normalized"] > 0.50
    ):
        print("research-pass: suitable for further offline hybrid-prior testing, not production deployment")
    else:
        print("research-only: behavior constraints are not all satisfied")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a safe-action neural prior ranker.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_bomb_pred_min", type=float, default=0.03)
    parser.add_argument("--target_bomb_pred_max", type=float, default=0.10)
    parser.add_argument("--target_stop_pred_max", type=float, default=0.30)
    args = parser.parse_args()
    evaluate_action_ranker(args)


if __name__ == "__main__":
    main()
