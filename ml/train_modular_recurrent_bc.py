from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.constants import PLACE_BOMB
from agent.rl_agent_recurrent.modular_model import ModularBomberCnnLstm


TYPE_NORMAL = 0
TYPE_SAFE_BOMB = 1
TYPE_ESCAPE = 2
TYPE_UNSAFE = 3


class SequenceDataset(Dataset):
    def __init__(self, obs, actions, mask, sequence_type):
        self.obs = obs
        self.actions = actions
        self.mask = mask
        self.sequence_type = sequence_type

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.mask[idx]),
            int(self.sequence_type[idx]),
        )


def split_by_sequence(dataset, val_fraction, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(dataset.actions))
    split = int(len(idx) * (1.0 - val_fraction))
    return idx[:split], idx[split:]


def load_dataset(path, val_fraction, seed):
    data = np.load(path, allow_pickle=True)
    obs = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    mask = data["valid_mask"].astype(bool)
    sequence_type = data["sequence_type"].astype(np.int16)
    names = [str(v) for v in data["sequence_type_names"].tolist()]
    ds = SequenceDataset(obs, actions, mask, sequence_type)
    train_idx, val_idx = split_by_sequence(ds, val_fraction, seed)
    return ds, train_idx, val_idx, names


def make_model(args, in_channels):
    return ModularBomberCnnLstm(
        in_channels=in_channels,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout,
        layer_norm=args.layer_norm,
    )


def load_checkpoint(model, path, device, strict=False):
    if not path:
        return
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=strict)


def set_trainable(model, stage, freeze_encoder):
    for p in model.parameters():
        p.requires_grad = False
    if stage == "movement":
        modules = [model.movement_head]
        if not freeze_encoder:
            modules += [model.cnn, model.lstm, model.layer_norm]
    elif stage == "bomb":
        modules = [model.bomb_head]
        if not freeze_encoder:
            modules += [model.cnn, model.lstm, model.layer_norm]
    elif stage == "escape":
        modules = [model.escape_head]
        if not freeze_encoder:
            modules += [model.cnn, model.lstm, model.layer_norm]
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    for module in modules:
        for p in module.parameters():
            p.requires_grad = True


def masked_ce(logits, actions, mask):
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_actions = actions.reshape(-1)
    flat_mask = mask.reshape(-1).float()
    loss = F.cross_entropy(flat_logits, flat_actions, reduction="none")
    return (loss * flat_mask).sum() / torch.clamp(flat_mask.sum(), min=1.0)


def masked_bce(logits, targets, mask, pos_weight=None, focal_gamma=0.0):
    flat_logits = logits.reshape(-1)
    flat_targets = targets.reshape(-1).float()
    flat_mask = mask.reshape(-1).float()
    loss = F.binary_cross_entropy_with_logits(flat_logits, flat_targets, pos_weight=pos_weight, reduction="none")
    if focal_gamma > 0:
        probs = torch.sigmoid(flat_logits)
        pt = torch.where(flat_targets > 0.5, probs, 1.0 - probs)
        loss = loss * torch.pow(1.0 - pt, focal_gamma)
    return (loss * flat_mask).sum() / torch.clamp(flat_mask.sum(), min=1.0)


def filter_indices(ds, indices, stage):
    types = ds.sequence_type[indices]
    if stage == "movement":
        keep = types == TYPE_NORMAL
    elif stage == "bomb":
        keep = np.isin(types, [TYPE_SAFE_BOMB, TYPE_UNSAFE])
    elif stage == "escape":
        keep = types == TYPE_ESCAPE
    else:
        raise ValueError(stage)
    return indices[keep]


def train_one_epoch(model, loader, args, device):
    model.train()
    losses = []
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    pos_weight = torch.as_tensor(args.bomb_pos_weight, device=device) if args.bomb_pos_weight > 0 else None
    for obs, actions, mask, seq_type in loader:
        obs = obs.to(device=device, dtype=torch.float32)
        actions = actions.to(device=device, dtype=torch.long)
        mask = mask.to(device=device, dtype=torch.bool)
        seq_type = seq_type.to(device=device, dtype=torch.long)
        out, _ = model(obs)
        if args.stage == "movement":
            non_bomb = mask & (actions != PLACE_BOMB)
            loss = masked_ce(out["movement_logits"], torch.clamp(actions, max=4), non_bomb)
        elif args.stage == "bomb":
            targets = (seq_type[:, None].expand_as(actions) == TYPE_SAFE_BOMB).float()
            loss = masked_bce(
                out["bomb_logit"],
                targets,
                mask,
                pos_weight=pos_weight,
                focal_gamma=args.focal_gamma if args.focal_bomb else 0.0,
            )
        elif args.stage == "escape":
            non_bomb = mask & (actions != PLACE_BOMB)
            loss = masked_ce(out["escape_logits"], torch.clamp(actions, max=4), non_bomb)
        else:
            raise ValueError(args.stage)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def eval_stage(model, loader, device, stage):
    model.eval()
    movement_conf = np.zeros((5, 5), dtype=np.int64)
    escape_conf = np.zeros((5, 5), dtype=np.int64)
    bomb_tp = bomb_fp = bomb_tn = bomb_fn = 0
    unsafe_fp = unsafe_total = 0
    pred_counts = np.zeros(6, dtype=np.int64)
    for obs, actions, mask, seq_type in loader:
        obs = obs.to(device=device, dtype=torch.float32)
        actions_t = actions.to(device=device, dtype=torch.long)
        mask_t = mask.to(device=device, dtype=torch.bool)
        out, _ = model(obs)
        move_pred = out["movement_logits"].argmax(dim=-1).cpu().numpy()
        escape_pred = out["escape_logits"].argmax(dim=-1).cpu().numpy()
        bomb_prob = torch.sigmoid(out["bomb_logit"]).cpu().numpy()
        actions_np = actions.numpy()
        mask_np = mask.numpy().astype(bool)
        types_np = seq_type.numpy()
        for b in range(actions_np.shape[0]):
            valid = mask_np[b]
            if types_np[b] == TYPE_NORMAL:
                for t, p in zip(actions_np[b][valid], move_pred[b][valid]):
                    if int(t) < PLACE_BOMB:
                        movement_conf[int(t), int(p)] += 1
            elif types_np[b] == TYPE_ESCAPE:
                for t, p in zip(actions_np[b][valid], escape_pred[b][valid]):
                    if int(t) < PLACE_BOMB:
                        escape_conf[int(t), int(p)] += 1
            elif types_np[b] in {TYPE_SAFE_BOMB, TYPE_UNSAFE}:
                pred_bomb = bomb_prob[b][valid] >= 0.5
                target_bomb = types_np[b] == TYPE_SAFE_BOMB
                if types_np[b] == TYPE_UNSAFE:
                    unsafe_fp += int(np.sum(pred_bomb))
                    unsafe_total += int(np.sum(valid))
                if target_bomb:
                    bomb_tp += int(np.sum(pred_bomb))
                    bomb_fn += int(np.sum(~pred_bomb))
                else:
                    bomb_fp += int(np.sum(pred_bomb))
                    bomb_tn += int(np.sum(~pred_bomb))
            combined = move_pred[b].copy()
            combined[types_np[b] == TYPE_ESCAPE if False else valid] = move_pred[b][valid]
            for value in move_pred[b][valid]:
                pred_counts[int(value)] += 1
    movement_total = int(movement_conf.sum())
    escape_total = int(escape_conf.sum())
    precision = bomb_tp / max(1, bomb_tp + bomb_fp)
    recall = bomb_tp / max(1, bomb_tp + bomb_fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "stage": stage,
        "movement_accuracy": float(np.trace(movement_conf) / max(1, movement_total)),
        "escape_accuracy": float(np.trace(escape_conf) / max(1, escape_total)),
        "bomb_precision": float(precision),
        "bomb_recall": float(recall),
        "bomb_f1": float(f1),
        "unsafe_false_bomb_rate": float(unsafe_fp / max(1, unsafe_total)),
        "movement_confusion": movement_conf.tolist(),
        "escape_confusion": escape_conf.tolist(),
        "predicted_movement_distribution": {str(i): int(v) for i, v in enumerate(pred_counts[:5])},
    }


def save_checkpoint(path, model, args, metrics, type_names):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "in_channels": int(model.in_channels),
            "embedding_dim": int(model.embedding_dim),
            "hidden_size": int(model.hidden_size),
            "num_lstm_layers": int(model.num_lstm_layers),
            "dropout": float(model.dropout),
            "layer_norm": bool(model.layer_norm_enabled),
        },
        "stage": args.stage,
        "metrics": metrics,
        "sequence_type_names": type_names,
    }
    torch.save(checkpoint, output)
    output.with_suffix(".json").write_text(json.dumps(checkpoint, indent=2, default=str), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train modular recurrent BC heads.")
    parser.add_argument("--dataset", default="ml/datasets/recurrent_bc_balanced_sequences.npz")
    parser.add_argument("--stage", choices=["movement", "bomb", "escape"], required=True)
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_lstm_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--layer_norm", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--focal_bomb", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--bomb_pos_weight", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=9970)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ds, train_idx, val_idx, type_names = load_dataset(args.dataset, args.val_fraction, args.seed)
    device = torch.device(args.device)
    model = make_model(args, int(ds.obs.shape[2])).to(device)
    load_checkpoint(model, args.init_checkpoint, device)
    set_trainable(model, args.stage, args.freeze_encoder)
    stage_train_idx = filter_indices(ds, train_idx, args.stage)
    stage_val_idx = filter_indices(ds, val_idx, args.stage)
    train_loader = DataLoader(Subset(ds, stage_train_idx.tolist()), batch_size=args.batch_size, shuffle=True)
    full_val_loader = DataLoader(Subset(ds, val_idx.tolist()), batch_size=args.batch_size, shuffle=False)
    stage_val_loader = DataLoader(Subset(ds, stage_val_idx.tolist()), batch_size=args.batch_size, shuffle=False)
    history = []
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, args, device)
        stage_metrics = eval_stage(model, stage_val_loader, device, args.stage)
        full_metrics = eval_stage(model, full_val_loader, device, args.stage)
        row = {"epoch": epoch + 1, "loss": loss, "stage_val": stage_metrics, "full_val": full_metrics}
        history.append(row)
        print(json.dumps(row))
    metrics = {
        "stage": args.stage,
        "history": history,
        "final_stage_val": eval_stage(model, stage_val_loader, device, args.stage),
        "final_full_val": eval_stage(model, full_val_loader, device, args.stage),
        "train_sequences": int(len(stage_train_idx)),
        "val_sequences": int(len(stage_val_idx)),
    }
    save_checkpoint(args.output, model, args, metrics, type_names)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
