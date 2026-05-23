"""Sweep tiny imitation-training hyperparameters and rank behavior quality."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from ml.models.simple_cnn_policy import TORCH_AVAILABLE, build_model
from ml.train_imitation import ACTION_NAMES


if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
else:  # pragma: no cover - depends on local environment
    torch = None
    DataLoader = None
    TensorDataset = None


DEFAULT_CLASS_WEIGHT_POWERS = [0.0, 0.3, 0.6, 0.8, 1.0]
DEFAULT_BOMB_BOOSTS = [1.0, 1.5, 2.0]
DEFAULT_STOP_PENALTIES = [1.0, 0.8, 0.6]


def _dataset_label(path):
    name = Path(path).stem
    if "wins_only" in name:
        return "wins_only"
    if "balanced" in name:
        return "balanced"
    return "original"


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


def _split_indices(total, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)
    val_size = max(1, int(total * val_ratio))
    return indices[val_size:], indices[:val_size]


def _class_weights(actions, class_weight_power, bomb_boost_weight, stop_penalty_weight):
    counts = np.bincount(actions, minlength=len(ACTION_NAMES)).astype(np.float32)
    base = counts.sum() / (len(ACTION_NAMES) * np.maximum(counts, 1.0))
    weights = np.power(base, float(class_weight_power))
    weights[5] *= float(bomb_boost_weight)
    weights[0] *= float(stop_penalty_weight)
    return weights.astype(np.float32)


def _evaluate_model(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    top2_correct = 0
    pred_counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    entropy_values = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            top2 = probs.topk(k=min(2, probs.shape[1]), dim=1).indices

            total += int(labels.shape[0])
            correct += int((preds == labels).sum().item())
            top2_correct += int((top2 == labels.unsqueeze(1)).any(dim=1).sum().item())
            pred_counts += np.bincount(preds.cpu().numpy(), minlength=len(ACTION_NAMES))
            entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-9)), dim=1)
            entropy_values.extend(entropy.cpu().numpy().tolist())

    max_entropy = float(np.log(len(ACTION_NAMES)))
    entropy_mean = float(np.mean(entropy_values)) if entropy_values else 0.0
    entropy_norm = entropy_mean / max_entropy if max_entropy else 0.0
    return {
        "val_accuracy": correct / max(1, total),
        "top2_accuracy": top2_correct / max(1, total),
        "entropy": entropy_mean,
        "entropy_normalized": entropy_norm,
        "prediction_counts": pred_counts,
        "total": total,
    }


def _selection_score(metrics):
    total = max(1, int(metrics["total"]))
    pred = metrics["prediction_counts"]
    pred_pct = 100.0 * pred.astype(np.float64) / float(total)
    stop_pct = float(pred_pct[0])
    bomb_pct = float(pred_pct[5])
    penalties = 0.0
    warnings = []

    if stop_pct > 35.0:
        value = min(0.40, 0.15 + (stop_pct - 35.0) / 80.0)
        penalties += value
        warnings.append("STOP>35")
    if bomb_pct < 2.0:
        value = min(0.45, 0.20 + (2.0 - bomb_pct) / 10.0)
        penalties += value
        warnings.append("BOMB<2")
    if bomb_pct > 12.0:
        value = min(0.45, 0.15 + (bomb_pct - 12.0) / 70.0)
        penalties += value
        warnings.append("BOMB>12")

    for action in range(1, 5):
        if pred_pct[action] > 45.0:
            value = min(0.35, 0.10 + (pred_pct[action] - 45.0) / 80.0)
            penalties += value
            warnings.append(f"{ACTION_NAMES[action]}>45")

    score = float(metrics["top2_accuracy"]) + 0.2 * float(metrics["entropy_normalized"]) - penalties
    return score, warnings


def _train_one(observations, actions, args, class_weight_power, bomb_boost_weight,
               stop_penalty_weight, seed):
    torch.manual_seed(int(seed))
    train_idx, val_idx = _split_indices(len(actions), seed=seed)
    device = torch.device("cpu")
    x_train = torch.from_numpy(observations[train_idx])
    y_train = torch.from_numpy(actions[train_idx])
    x_val = torch.from_numpy(observations[val_idx])
    y_val = torch.from_numpy(actions[val_idx])

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
    )

    weights = _class_weights(actions, class_weight_power, bomb_boost_weight, stop_penalty_weight)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    model = build_model(input_channels=12, num_actions=len(ACTION_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for _epoch in range(args.epochs):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

    return _evaluate_model(model, val_loader, device)


def _result_row(run_id, dataset_path, class_weight_power, bomb_boost_weight,
                stop_penalty_weight, metrics):
    score, warnings = _selection_score(metrics)
    total = max(1, int(metrics["total"]))
    pred = metrics["prediction_counts"]
    pred_pct = 100.0 * pred.astype(np.float64) / float(total)
    movement = pred_pct[1:5]
    return {
        "run_id": run_id,
        "dataset": _dataset_label(dataset_path),
        "dataset_path": str(dataset_path),
        "class_weight_power": float(class_weight_power),
        "bomb_boost_weight": float(bomb_boost_weight),
        "stop_penalty_weight": float(stop_penalty_weight),
        "val_accuracy": float(metrics["val_accuracy"]),
        "top2_accuracy": float(metrics["top2_accuracy"]),
        "entropy_normalized": float(metrics["entropy_normalized"]),
        "stop_pred_pct": float(pred_pct[0]),
        "bomb_pred_pct": float(pred_pct[5]),
        "movement_diversity_pct": float(np.count_nonzero(movement > 1.0) / 4.0 * 100.0),
        "max_movement_direction_pct": float(np.max(movement)) if len(movement) else 0.0,
        "selection_score": float(score),
        "warnings": warnings,
    }


def _format_pct(value):
    return f"{value * 100.0:.1f}%"


def _write_report(path, results, best_config):
    lines = [
        "# Imitation Hyperparameter Sweep Report",
        "",
        "This report ranks tiny imitation policies by behavior quality, not accuracy alone.",
        "",
        "## All Runs",
        "",
        "| Run | Dataset | CWP | Bomb Boost | STOP Weight | Val Acc | Top-2 | Entropy | STOP Pred | Bomb Pred | Move Div | Max Move Dir | Score | Warnings |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        warning_text = ", ".join(item["warnings"]) or "none"
        lines.append(
            "| {run_id} | {dataset} | {class_weight_power:.1f} | {bomb_boost_weight:.1f} | "
            "{stop_penalty_weight:.1f} | {val_accuracy:.1%} | {top2_accuracy:.1%} | "
            "{entropy_normalized:.3f} | {stop_pred_pct:.1f}% | {bomb_pred_pct:.1f}% | "
            "{movement_diversity_pct:.0f}% | {max_movement_direction_pct:.1f}% | "
            "{selection_score:.3f} | {warning_text} |".format(
                warning_text=warning_text,
                **item,
            )
        )

    lines.extend([
        "",
        "## Top 5 Recommended Configs",
        "",
        "| Rank | Run | Dataset | CWP | Bomb Boost | STOP Weight | Top-2 | Entropy | STOP Pred | Bomb Pred | Score | Why |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for rank, item in enumerate(results[:5], start=1):
        why = "balanced" if not item["warnings"] else "tradeoff: " + ", ".join(item["warnings"])
        lines.append(
            f"| {rank} | {item['run_id']} | {item['dataset']} | "
            f"{item['class_weight_power']:.1f} | {item['bomb_boost_weight']:.1f} | "
            f"{item['stop_penalty_weight']:.1f} | {_format_pct(item['top2_accuracy'])} | "
            f"{item['entropy_normalized']:.3f} | {item['stop_pred_pct']:.1f}% | "
            f"{item['bomb_pred_pct']:.1f}% | {item['selection_score']:.3f} | {why} |"
        )

    lines.extend([
        "",
        "## Selected Config",
        "",
        f"Best run: `{best_config.get('run_id')}` on `{best_config.get('dataset')}`.",
        "",
        "The selection score rewards top-2 accuracy and entropy, then penalizes STOP-heavy, no-bomb, bomb-heavy, and single-direction movement collapse. This keeps the chosen policy closer to a usable action-ranker for a future safety-filtered hybrid system.",
    ])
    if best_config.get("warnings"):
        lines.extend([
            "",
            "No warning-free policy was found in this sweep. Treat the selected config as the least-bad research candidate, not a deployable policy.",
        ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(args):
    if not TORCH_AVAILABLE:
        message = {
            "error": "PyTorch not installed",
            "best_config": None,
        }
        Path(args.best_config).write_text(json.dumps(message, indent=2), encoding="utf-8")
        Path(args.output_report).write_text(
            "# Imitation Hyperparameter Sweep Report\n\nPyTorch not installed; sweep skipped.\n",
            encoding="utf-8",
        )
        print("PyTorch not installed, skipping sweep.")
        return []

    configs = []
    for dataset in args.datasets:
        dataset_configs = list(product(
            [dataset],
            args.class_weight_powers,
            args.bomb_boost_weights,
            args.stop_penalty_weights,
        ))
        if args.limit_per_dataset is not None:
            dataset_configs = dataset_configs[:args.limit_per_dataset]
        configs.extend(dataset_configs)
    if args.limit is not None:
        configs = configs[:args.limit]

    results = []
    for run_id, (dataset_path, cwp, bomb_boost, stop_weight) in enumerate(configs, start=1):
        observations, actions = _load_dataset(dataset_path, max_samples=args.max_samples, seed=args.seed)
        if len(actions) < 10:
            print(f"Skipping {dataset_path}: not enough samples")
            continue
        print(
            f"run {run_id}/{len(configs)} dataset={_dataset_label(dataset_path)} "
            f"cwp={cwp} bomb={bomb_boost} stop={stop_weight}"
        )
        metrics = _train_one(
            observations,
            actions,
            args,
            class_weight_power=cwp,
            bomb_boost_weight=bomb_boost,
            stop_penalty_weight=stop_weight,
            seed=args.seed + run_id,
        )
        results.append(_result_row(run_id, dataset_path, cwp, bomb_boost, stop_weight, metrics))

    results.sort(key=lambda item: item["selection_score"], reverse=True)
    best_config = results[0] if results else {"error": "no completed runs"}
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.best_config).parent.mkdir(parents=True, exist_ok=True)
    _write_report(args.output_report, results, best_config)
    Path(args.best_config).write_text(json.dumps(best_config, indent=2), encoding="utf-8")

    print(f"wrote report: {args.output_report}")
    print(f"wrote best config: {args.best_config}")
    if results:
        print(
            "best: "
            f"dataset={best_config['dataset']} "
            f"cwp={best_config['class_weight_power']} "
            f"bomb={best_config['bomb_boost_weight']} "
            f"stop={best_config['stop_penalty_weight']} "
            f"score={best_config['selection_score']:.3f}"
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Sweep tiny imitation policy training hyperparameters.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_report", default="ml/imitation_sweep_report.md")
    parser.add_argument("--best_config", default="ml/best_imitation_config.json")
    parser.add_argument("--class_weight_powers", nargs="+", type=float, default=DEFAULT_CLASS_WEIGHT_POWERS)
    parser.add_argument("--bomb_boost_weights", nargs="+", type=float, default=DEFAULT_BOMB_BOOSTS)
    parser.add_argument("--stop_penalty_weights", nargs="+", type=float, default=DEFAULT_STOP_PENALTIES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit_per_dataset", type=int, default=None)
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
