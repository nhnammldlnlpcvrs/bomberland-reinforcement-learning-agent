from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ml.train_bomb_value_model import BombValueNet


def _load_model(path: str, device: str):
    checkpoint = torch.load(path, map_location=device)
    model = BombValueNet(checkpoint.get("in_channels", 19), checkpoint.get("scalar_dim", 5))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _predict(model, obs, scalars, device: str, batch_size: int):
    probs = []
    with torch.no_grad():
        for start in range(0, len(obs), batch_size):
            batch_obs = torch.as_tensor(obs[start:start + batch_size], dtype=torch.float32, device=device)
            batch_scalars = torch.as_tensor(scalars[start:start + batch_size], dtype=torch.float32, device=device)
            probs.append(torch.sigmoid(model(batch_obs, batch_scalars)).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)


def _summarize(name, probs, labels, sources, boxes, thresholds):
    rows = []
    y = labels.astype(bool)
    for threshold in thresholds:
        selected = probs >= threshold
        positives = y
        negatives = ~y
        row = {
            "name": name,
            "threshold": float(threshold),
            "samples": int(len(probs)),
            "selected_count": int(selected.sum()),
            "selected_fraction": float(selected.mean()) if len(selected) else 0.0,
            "precision": float((selected & positives).sum() / max(1, selected.sum())),
            "recall": float((selected & positives).sum() / max(1, positives.sum())),
            "false_positive_rate": float((selected & negatives).sum() / max(1, negatives.sum())),
            "selected_boxes_mean": float(np.mean(boxes[selected])) if selected.any() and boxes is not None else 0.0,
        }
        for source_id, source_name in ((3, "rollout_hard"), (4, "counterfactual"), (5, "legacy_hard")):
            hard = sources == source_id
            if hard.any():
                row[f"{source_name}_pass_through_rate"] = float(selected[hard].mean())
                row[f"{source_name}_selected"] = int(selected[hard].sum())
                row[f"{source_name}_total"] = int(hard.sum())
        rows.append(row)
    return rows


def _dataset_view(path: str, max_samples: int | None, seed: int):
    data = np.load(path)
    obs = data["observations"].astype(np.float32)
    scalars = data["scalar_features"].astype(np.float32)
    labels = data["labels"].astype(np.float32) if "labels" in data.files else np.zeros(len(obs), dtype=np.float32)
    sources = data["sample_type"].astype(np.int8) if "sample_type" in data.files else data["source"].astype(np.int8) if "source" in data.files else np.zeros(len(obs), dtype=np.int8)
    if "boxes_destroyed_after_bomb" in data.files:
        boxes = data["boxes_destroyed_after_bomb"].astype(np.float32)
    elif "nearby_box_count" in data.files:
        boxes = data["nearby_box_count"].astype(np.float32)
    else:
        boxes = np.zeros(len(obs), dtype=np.float32)
    if max_samples and len(obs) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.arange(len(obs)), size=max_samples, replace=False)
        obs, scalars, labels, sources, boxes = obs[idx], scalars[idx], labels[idx], sources[idx], boxes[idx]
    return obs, scalars, labels, sources, boxes


def evaluate(args):
    model, checkpoint = _load_model(args.model, args.device)
    thresholds = args.thresholds or [float(checkpoint.get("threshold", 0.8))]
    all_rows = []
    for item in args.datasets:
        name, path = item.split("=", 1) if "=" in item else (Path(item).stem, item)
        obs, scalars, labels, sources, boxes = _dataset_view(path, args.max_samples, args.seed)
        probs = _predict(model, obs, scalars, args.device, args.batch_size)
        all_rows.extend(_summarize(name, probs, labels, sources, boxes, thresholds))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"model": args.model, "thresholds": thresholds, "results": all_rows}
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Selector-only evaluation for offline bomb value model.")
    parser.add_argument("--model", default="ml/checkpoints/rl_agent_pure/bomb_value_model_v2.pt")
    parser.add_argument("--datasets", nargs="+", required=True, help="Entries like name=path.npz")
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.6, 0.7, 0.8])
    parser.add_argument("--output", default="logs/bomb_selector_eval.json")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
