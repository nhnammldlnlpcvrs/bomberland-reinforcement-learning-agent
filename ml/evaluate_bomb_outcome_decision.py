from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_recurrent.bomb_outcome_model import BombOutcomeCnnLstm
from ml.train_bomb_outcome_model import BombOutcomeDataset, first_valid_score


def load_model(path: str, in_channels: int, device: torch.device) -> BombOutcomeCnnLstm:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    model = BombOutcomeCnnLstm(
        in_channels=int(config.get("in_channels", in_channels)),
        embedding_dim=int(config.get("embedding_dim", 128)),
        hidden_size=int(config.get("hidden_size", 128)),
        num_lstm_layers=int(config.get("num_lstm_layers", 1)),
        dropout=float(config.get("dropout", 0.0)),
        layer_norm=bool(config.get("layer_norm", False)),
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    pred_boxes = []
    pred_death = []
    pred_zero = []
    pred_survival = []
    true_boxes = []
    source = []
    for obs, mask, boxes, _death, _zero, _survived, _escape, _reachable, batch_source in loader:
        obs = obs.to(device=device, dtype=torch.float32)[:, :1]
        mask = mask.to(device=device, dtype=torch.bool)[:, :1]
        out, _ = model(obs)
        pred_boxes.append(first_valid_score(out["box_value"], mask).cpu().numpy())
        pred_death.append(torch.sigmoid(first_valid_score(out["death_risk_logit"], mask)).cpu().numpy())
        pred_zero.append(torch.sigmoid(first_valid_score(out["zero_value_logit"], mask)).cpu().numpy())
        pred_survival.append(torch.sigmoid(first_valid_score(out["escape_success_logit"], mask)).cpu().numpy())
        true_boxes.append(boxes.numpy())
        source.append(batch_source.numpy())
    return {
        "pred_boxes": np.concatenate(pred_boxes) if pred_boxes else np.zeros(0),
        "pred_death": np.concatenate(pred_death) if pred_death else np.zeros(0),
        "pred_zero": np.concatenate(pred_zero) if pred_zero else np.zeros(0),
        "pred_survival": np.concatenate(pred_survival) if pred_survival else np.zeros(0),
        "true_boxes": np.concatenate(true_boxes) if true_boxes else np.zeros(0),
        "source": np.concatenate(source) if source else np.zeros(0, dtype=np.int16),
    }


def parse_floats(text: str):
    return [float(value) for value in text.split(",") if value.strip()]


def main():
    parser = argparse.ArgumentParser(description="Sweep multi-outcome bomb decision scores offline.")
    parser.add_argument("--dataset", default="ml/datasets/bomb_outcome_dataset.npz")
    parser.add_argument("--checkpoint", default="ml/checkpoints/rl_agent_recurrent/bomb_outcome_model.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--death_weights", default="2,3,5")
    parser.add_argument("--zero_weights", default="1,2,3")
    parser.add_argument("--box_weights", default="1,2")
    parser.add_argument("--survival_weights", default="1")
    parser.add_argument("--score_thresholds", default="0,0.25,0.5,1.0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="logs/bomb_outcome_decision_sweep.json")
    args = parser.parse_args()

    ds = BombOutcomeDataset(args.dataset)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, int(ds.obs.shape[2]), device)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    pred = collect_predictions(model, loader, device)
    source = pred["source"]
    useful = source == 0
    zero = source == 1
    death = source == 2
    rows = []
    for death_weight, zero_weight, box_weight, survival_weight, threshold in product(
        parse_floats(args.death_weights),
        parse_floats(args.zero_weights),
        parse_floats(args.box_weights),
        parse_floats(args.survival_weights),
        parse_floats(args.score_thresholds),
    ):
        score = (
            box_weight * pred["pred_boxes"]
            + survival_weight * pred["pred_survival"]
            - death_weight * pred["pred_death"]
            - zero_weight * pred["pred_zero"]
        )
        selected = score >= threshold
        tp = int(np.sum(selected & useful))
        fp = int(np.sum(selected & ~useful))
        fn = int(np.sum(~selected & useful))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        row = {
            "box_weight": float(box_weight),
            "survival_weight": float(survival_weight),
            "death_weight": float(death_weight),
            "zero_weight": float(zero_weight),
            "score_threshold": float(threshold),
            "selected_bomb_count": int(np.sum(selected)),
            "useful_precision": float(precision),
            "useful_recall": float(recall),
            "zero_value_fpr": float(np.sum(selected & zero) / max(1, np.sum(zero))),
            "death_after_bomb_fpr": float(np.sum(selected & death) / max(1, np.sum(death))),
            "expected_boxes_per_selected": float(np.mean(pred["true_boxes"][selected])) if np.any(selected) else 0.0,
            "predicted_boxes_per_selected": float(np.mean(pred["pred_boxes"][selected])) if np.any(selected) else 0.0,
            "passes_gate": bool(
                precision >= 0.70
                and recall >= 0.20
                and (np.sum(selected & zero) / max(1, np.sum(zero))) <= 0.10
                and (np.sum(selected & death) / max(1, np.sum(death))) <= 0.10
                and int(np.sum(selected)) > 0
            ),
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["passes_gate"],
            row["useful_precision"],
            row["useful_recall"],
            -row["zero_value_fpr"],
            -row["death_after_bomb_fpr"],
        ),
        reverse=True,
    )
    report = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "sample_count": int(len(source)),
        "useful_count": int(np.sum(useful)),
        "zero_value_count": int(np.sum(zero)),
        "death_count": int(np.sum(death)),
        "gate_passed": bool(any(row["passes_gate"] for row in rows)),
        "best_rows": rows[:20],
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
