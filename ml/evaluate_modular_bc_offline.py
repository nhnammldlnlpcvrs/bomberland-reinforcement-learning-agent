from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.constants import PLACE_BOMB
from agent.rl_agent_recurrent.modular_model import ModularBomberCnnLstm
from ml.train_modular_recurrent_bc import (
    TYPE_ESCAPE,
    TYPE_NORMAL,
    TYPE_SAFE_BOMB,
    TYPE_UNSAFE,
    SequenceDataset,
    load_dataset,
)


def load_model(path, in_channels, device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    model = ModularBomberCnnLstm(
        in_channels=int(config.get("in_channels", in_channels)),
        embedding_dim=int(config.get("embedding_dim", 128)),
        hidden_size=int(config.get("hidden_size", 128)),
        num_lstm_layers=int(config.get("num_lstm_layers", 1)),
        dropout=float(config.get("dropout", 0.0)),
        layer_norm=bool(config.get("layer_norm", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


def first_valid_score(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    first = mask.float().argmax(dim=1)
    return logits[torch.arange(logits.shape[0], device=logits.device), first]


@torch.no_grad()
def evaluate(model, loader, threshold, device, value_threshold=None):
    move_conf = np.zeros((5, 5), dtype=np.int64)
    escape_conf = np.zeros((5, 5), dtype=np.int64)
    combined_conf = np.zeros((6, 6), dtype=np.int64)
    bomb_tp = bomb_fp = bomb_tn = bomb_fn = 0
    unsafe_fp = unsafe_total = 0
    pred_counts = np.zeros(6, dtype=np.int64)
    bomb_scores = []
    bomb_labels = []
    bomb_unsafe_flags = []
    for obs, actions, mask, seq_type in loader:
        obs = obs.to(device=device, dtype=torch.float32)
        out, _ = model(obs)
        move_pred = out["movement_logits"].argmax(dim=-1).cpu().numpy()
        escape_pred = out["escape_logits"].argmax(dim=-1).cpu().numpy()
        bomb_prob = torch.sigmoid(out["bomb_logit"]).cpu().numpy()
        value_prob = torch.sigmoid(out.get("bomb_value_logit", out["bomb_logit"])).cpu().numpy()
        actions_np = actions.numpy()
        mask_np = mask.numpy().astype(bool)
        types_np = seq_type.numpy()
        for b in range(actions_np.shape[0]):
            valid = mask_np[b]
            typ = int(types_np[b])
            combined = move_pred[b].copy()
            if typ == TYPE_ESCAPE:
                combined = escape_pred[b].copy()
            bomb_pred = bomb_prob[b] >= threshold
            if value_threshold is not None:
                bomb_pred = bomb_pred & (value_prob[b] >= value_threshold)
            if typ in {TYPE_SAFE_BOMB, TYPE_UNSAFE}:
                combined[bomb_pred] = PLACE_BOMB
                target_bomb = typ == TYPE_SAFE_BOMB
                bomb_scores.extend(bomb_prob[b][valid].tolist())
                bomb_labels.extend(([1] if target_bomb else [0]) * int(np.sum(valid)))
                bomb_unsafe_flags.extend(([0] if target_bomb else [1]) * int(np.sum(valid)))
                if target_bomb:
                    bomb_tp += int(np.sum(bomb_pred[valid]))
                    bomb_fn += int(np.sum(~bomb_pred[valid]))
                else:
                    bomb_fp += int(np.sum(bomb_pred[valid]))
                    bomb_tn += int(np.sum(~bomb_pred[valid]))
                    unsafe_fp += int(np.sum(bomb_pred[valid]))
                    unsafe_total += int(np.sum(valid))
            for t, p in zip(actions_np[b][valid], combined[valid]):
                combined_conf[int(t), int(p)] += 1
                pred_counts[int(p)] += 1
            if typ == TYPE_NORMAL:
                for t, p in zip(actions_np[b][valid], move_pred[b][valid]):
                    if int(t) < PLACE_BOMB:
                        move_conf[int(t), int(p)] += 1
            if typ == TYPE_ESCAPE:
                for t, p in zip(actions_np[b][valid], escape_pred[b][valid]):
                    if int(t) < PLACE_BOMB:
                        escape_conf[int(t), int(p)] += 1
    precision = bomb_tp / max(1, bomb_tp + bomb_fp)
    recall = bomb_tp / max(1, bomb_tp + bomb_fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    total = int(combined_conf.sum())
    max_single = float(pred_counts.max() / max(1, pred_counts.sum()))
    return {
        "threshold": float(threshold),
        "safety_threshold": float(threshold),
        "value_threshold": None if value_threshold is None else float(value_threshold),
        "combined_accuracy": float(np.trace(combined_conf) / max(1, total)),
        "movement_accuracy": float(np.trace(move_conf) / max(1, move_conf.sum())),
        "post_bomb_escape_accuracy": float(np.trace(escape_conf) / max(1, escape_conf.sum())),
        "bomb_precision": float(precision),
        "bomb_recall": float(recall),
        "bomb_f1": float(f1),
        "unsafe_false_bomb_rate": float(unsafe_fp / max(1, unsafe_total)),
        "max_single_action_frac": max_single,
        "predicted_action_distribution": {str(i): int(v) for i, v in enumerate(pred_counts)},
        "combined_confusion": combined_conf.tolist(),
    }


def pr_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if labels.size == 0 or int(labels.sum()) == 0:
        return 0.0
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / max(1, int(y.sum()))
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


@torch.no_grad()
def collect_bomb_scores(model, loader, device):
    scores = []
    labels = []
    unsafe = []
    for obs, _actions, mask, seq_type in loader:
        obs = obs.to(device=device, dtype=torch.float32)
        out, _ = model(obs)
        bomb_prob = torch.sigmoid(out["bomb_logit"]).cpu().numpy()
        mask_np = mask.numpy().astype(bool)
        types_np = seq_type.numpy()
        for b in range(mask_np.shape[0]):
            typ = int(types_np[b])
            if typ not in {TYPE_SAFE_BOMB, TYPE_UNSAFE}:
                continue
            valid = mask_np[b]
            target = 1 if typ == TYPE_SAFE_BOMB else 0
            scores.extend(bomb_prob[b][valid].tolist())
            labels.extend([target] * int(np.sum(valid)))
            unsafe.extend([1 - target] * int(np.sum(valid)))
    return np.asarray(labels), np.asarray(scores), np.asarray(unsafe)


@torch.no_grad()
def evaluate_value_dataset(model, path, thresholds, batch_size, device):
    if not path:
        return None
    data_path = Path(path)
    if not data_path.exists():
        return None
    data = np.load(data_path, allow_pickle=True)
    obs = torch.from_numpy(data["observations"].astype(np.float32))
    mask = torch.from_numpy(data["valid_mask"].astype(bool))
    labels = data["label_value_now"].astype(np.float32)
    source = data["source_type"].astype(np.int16)
    source_names = data.get("source_type_names", None)
    source_name_map = {}
    if source_names is not None:
        source_name_map = {
            str(name): idx for idx, name in enumerate(source_names.tolist())
        }
    onpolicy_zero_idx = source_name_map.get("onpolicy_zero_value_negative", 3)
    onpolicy_death_idx = source_name_map.get("onpolicy_death_negative", 4)
    rows = []
    scores = []
    for start in range(0, len(labels), batch_size):
        batch_obs = obs[start:start + batch_size].to(device=device, dtype=torch.float32)
        batch_mask = mask[start:start + batch_size].to(device=device, dtype=torch.bool)
        out, _ = model(batch_obs)
        scores.append(torch.sigmoid(first_valid_score(out.get("bomb_value_logit", out["bomb_logit"]), batch_mask)).cpu().numpy())
    scores = np.concatenate(scores) if scores else np.zeros(0)
    for threshold in thresholds:
        pred = scores >= threshold
        pos = labels > 0.5
        safe_zero = source == 1
        unsafe = source == 2
        onpolicy_zero = source == onpolicy_zero_idx
        onpolicy_death = source == onpolicy_death_idx
        tp = int(np.sum(pred & pos))
        fp = int(np.sum(pred & ~pos))
        fn = int(np.sum(~pred & pos))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        rows.append({
            "value_threshold": float(threshold),
            "value_precision": float(precision),
            "value_recall": float(recall),
            "value_f1": float(2 * precision * recall / max(1e-8, precision + recall)),
            "safe_zero_value_fpr": float(np.sum(pred & safe_zero) / max(1, np.sum(safe_zero))),
            "unsafe_fpr": float(np.sum(pred & unsafe) / max(1, np.sum(unsafe))),
            "onpolicy_zero_value_fpr": float(np.sum(pred & onpolicy_zero) / max(1, np.sum(onpolicy_zero))),
            "death_after_bomb_fpr": float(np.sum(pred & onpolicy_death) / max(1, np.sum(onpolicy_death))),
            "predicted_value_frequency": float(np.mean(pred)) if len(pred) else 0.0,
        })
    return {
        "dataset": str(data_path),
        "rows": rows,
        "score_mean_positive": float(scores[labels > 0.5].mean()) if np.any(labels > 0.5) else 0.0,
        "score_mean_safe_zero": float(scores[source == 1].mean()) if np.any(source == 1) else 0.0,
        "score_mean_unsafe": float(scores[source == 2].mean()) if np.any(source == 2) else 0.0,
        "score_mean_onpolicy_zero": float(scores[source == onpolicy_zero_idx].mean()) if np.any(source == onpolicy_zero_idx) else 0.0,
        "score_mean_death_after_bomb": float(scores[source == onpolicy_death_idx].mean()) if np.any(source == onpolicy_death_idx) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Offline threshold sweep for modular recurrent BC.")
    parser.add_argument("--dataset", default="ml/datasets/recurrent_bc_balanced_sequences.npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.8")
    parser.add_argument("--safety_thresholds", default="")
    parser.add_argument("--value_thresholds", default="")
    parser.add_argument("--value_dataset", default="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=9970)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    ds, _train_idx, val_idx, _names = load_dataset(args.dataset, args.val_fraction, args.seed)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, int(ds.obs.shape[2]), device)
    loader = DataLoader(Subset(ds, val_idx.tolist()), batch_size=args.batch_size, shuffle=False)
    safety_thresholds = [float(t) for t in (args.safety_thresholds or args.thresholds).split(",")]
    value_thresholds = [float(t) for t in args.value_thresholds.split(",") if t.strip()]
    if value_thresholds:
        rows = [
            evaluate(model, loader, safety_threshold, device, value_threshold=value_threshold)
            for safety_threshold in safety_thresholds
            for value_threshold in value_thresholds
        ]
    else:
        rows = [evaluate(model, loader, float(t), device) for t in args.thresholds.split(",")]
    labels, scores, unsafe = collect_bomb_scores(model, loader, device)
    value_report = evaluate_value_dataset(model, args.value_dataset, value_thresholds, args.batch_size, device)
    report = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "bomb_pr_auc": pr_auc(labels, scores),
        "bomb_score_mean_positive": float(scores[labels == 1].mean()) if np.any(labels == 1) else 0.0,
        "bomb_score_mean_unsafe": float(scores[unsafe == 1].mean()) if np.any(unsafe == 1) else 0.0,
        "value_report": value_report,
        "rows": rows,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
