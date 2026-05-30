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


@torch.no_grad()
def evaluate(model, loader, threshold, device):
    move_conf = np.zeros((5, 5), dtype=np.int64)
    escape_conf = np.zeros((5, 5), dtype=np.int64)
    combined_conf = np.zeros((6, 6), dtype=np.int64)
    bomb_tp = bomb_fp = bomb_tn = bomb_fn = 0
    unsafe_fp = unsafe_total = 0
    pred_counts = np.zeros(6, dtype=np.int64)
    for obs, actions, mask, seq_type in loader:
        obs = obs.to(device=device, dtype=torch.float32)
        out, _ = model(obs)
        move_pred = out["movement_logits"].argmax(dim=-1).cpu().numpy()
        escape_pred = out["escape_logits"].argmax(dim=-1).cpu().numpy()
        bomb_prob = torch.sigmoid(out["bomb_logit"]).cpu().numpy()
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
            if typ in {TYPE_SAFE_BOMB, TYPE_UNSAFE}:
                combined[bomb_pred] = PLACE_BOMB
                target_bomb = typ == TYPE_SAFE_BOMB
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


def main():
    parser = argparse.ArgumentParser(description="Offline threshold sweep for modular recurrent BC.")
    parser.add_argument("--dataset", default="ml/datasets/recurrent_bc_balanced_sequences.npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.8")
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
    rows = [evaluate(model, loader, float(t), device) for t in args.thresholds.split(",")]
    report = {"checkpoint": args.checkpoint, "dataset": args.dataset, "rows": rows}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
