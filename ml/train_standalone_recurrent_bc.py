from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.constants import NUM_ACTIONS, PLACE_BOMB
from agent.rl_agent_recurrent.standalone_model import StandaloneBomberCnnLstm


DEFAULT_TYPE_NAMES = ["unknown"]


class ChunkDataset(Dataset):
    def __init__(self, obs, actions, mask, source_episodes, sequence_types=None):
        self.obs = obs
        self.actions = actions
        self.mask = mask
        self.source_episodes = source_episodes
        self.sequence_types = sequence_types if sequence_types is not None else np.zeros(len(actions), dtype=np.int16)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.mask[idx]),
            int(self.source_episodes[idx]),
            int(self.sequence_types[idx]),
        )


def _canonical_mode(mode):
    return "movement" if mode == "movement_only" else mode


def _mode_mask(actions, valid_mask, mode):
    mode = _canonical_mode(mode)
    mask = valid_mask.astype(np.float32).copy()
    if mode == "movement":
        mask[actions == PLACE_BOMB] = 0.0
    elif mode in {"all", "bomb_light"}:
        pass
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return mask


def make_chunks(data, seq_len, burn_in, mode, overfit_subset=0, stride=0):
    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    valid = data["valid_mask"].astype(bool)
    sequence_types = data["sequence_type"].astype(np.int16) if "sequence_type" in data.files else np.zeros(len(actions), dtype=np.int16)
    episode_indices = np.arange(len(actions))
    if overfit_subset and overfit_subset > 0:
        episode_indices = episode_indices[:min(overfit_subset, len(episode_indices))]
    obs_chunks = []
    action_chunks = []
    mask_chunks = []
    source_episodes = []
    chunk_types = []
    stride = int(stride) if stride and stride > 0 else max(1, seq_len - burn_in)
    for ep_idx in episode_indices:
        length = int(valid[ep_idx].sum())
        if length <= 0:
            continue
        for start in range(0, length, stride):
            end = min(length, start + seq_len)
            actual = end - start
            obs = np.zeros((seq_len, *observations.shape[2:]), dtype=np.float32)
            act = np.zeros((seq_len,), dtype=np.int64)
            mask = np.zeros((seq_len,), dtype=np.float32)
            obs[:actual] = observations[ep_idx, start:end]
            act[:actual] = actions[ep_idx, start:end]
            base_mask = _mode_mask(actions[ep_idx, start:end], valid[ep_idx, start:end], mode)
            if burn_in > 0:
                base_mask[:min(burn_in, actual)] = 0.0
            mask[:actual] = base_mask
            if mask.sum() <= 0:
                continue
            obs_chunks.append(obs)
            action_chunks.append(act)
            mask_chunks.append(mask)
            source_episodes.append(ep_idx)
            chunk_types.append(int(sequence_types[ep_idx]))
    if not obs_chunks:
        raise ValueError("No chunks created")
    return (
        np.asarray(obs_chunks, dtype=np.float32),
        np.asarray(action_chunks, dtype=np.int64),
        np.asarray(mask_chunks, dtype=np.float32),
        np.asarray(source_episodes, dtype=np.int32),
        np.asarray(chunk_types, dtype=np.int16),
    )


def split_chunks(source_episodes, val_fraction, seed, episode_split):
    rng = np.random.default_rng(seed)
    if episode_split:
        eps = rng.permutation(np.unique(source_episodes))
        split = int(len(eps) * (1.0 - val_fraction))
        train_eps = set(int(v) for v in eps[:split])
        train_idx = np.asarray([i for i, ep in enumerate(source_episodes) if int(ep) in train_eps], dtype=np.int64)
        val_idx = np.asarray([i for i, ep in enumerate(source_episodes) if int(ep) not in train_eps], dtype=np.int64)
    else:
        idx = rng.permutation(len(source_episodes))
        split = int(len(idx) * (1.0 - val_fraction))
        train_idx = idx[:split]
        val_idx = idx[split:]
    if len(val_idx) == 0:
        val_idx = train_idx.copy()
    return train_idx, val_idx


def action_distribution(actions, mask):
    values = actions[mask.astype(bool)]
    return {str(i): int(v) for i, v in enumerate(np.bincount(values.reshape(-1), minlength=NUM_ACTIONS))}


def sequence_type_distribution(types, type_names):
    counts = Counter(int(v) for v in types)
    return {
        str(type_names[idx] if idx < len(type_names) else idx): int(counts.get(idx, 0))
        for idx in sorted(set(range(len(type_names))) | set(counts.keys()))
    }


def class_weights_for(actions, mask, max_class_weight, bomb_weight_multiplier=1.0):
    values = actions[mask.astype(bool)]
    counts = np.bincount(values.reshape(-1), minlength=NUM_ACTIONS).astype(np.float32)
    mean = counts[counts > 0].mean() if np.any(counts > 0) else 1.0
    weights = np.sqrt(mean / np.maximum(counts, 1.0))
    weights[PLACE_BOMB] *= float(bomb_weight_multiplier)
    return np.clip(weights, 0.5, max_class_weight).astype(np.float32)


def masked_ce_loss(logits, actions, mask, class_weights=None):
    batch, seq_len, num_actions = logits.shape
    flat_logits = logits.reshape(batch * seq_len, num_actions)
    flat_actions = actions.reshape(batch * seq_len)
    flat_mask = mask.reshape(batch * seq_len)
    ce = F.cross_entropy(flat_logits, flat_actions, weight=class_weights, reduction="none")
    return (ce * flat_mask).sum() / torch.clamp(flat_mask.sum(), min=1.0)


def masked_bce_loss(logits, targets, mask, pos_weight=None, focal_gamma=0.0):
    flat_logits = logits.reshape(-1)
    flat_targets = targets.reshape(-1).float()
    flat_mask = mask.reshape(-1)
    bce = F.binary_cross_entropy_with_logits(
        flat_logits,
        flat_targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    if focal_gamma and focal_gamma > 0:
        probs = torch.sigmoid(flat_logits)
        pt = torch.where(flat_targets > 0.5, probs, 1.0 - probs)
        bce = bce * torch.pow(1.0 - pt, focal_gamma)
    return (bce * flat_mask).sum() / torch.clamp(flat_mask.sum(), min=1.0)


def multi_head_loss(logits, aux, actions, mask, action_weights, args, bomb_pos_weight):
    loss = torch.zeros((), device=logits.device)
    if args.action_coef > 0:
        loss = loss + args.action_coef * masked_ce_loss(logits, actions, mask, action_weights)
    if args.multi_head and aux:
        non_bomb_mask = mask * (actions != PLACE_BOMB).float()
        if args.movement_coef > 0:
            movement_actions = torch.clamp(actions, max=PLACE_BOMB - 1)
            loss = loss + args.movement_coef * masked_ce_loss(
                aux["movement_logits"],
                movement_actions,
                non_bomb_mask,
                None,
            )
        if args.bomb_coef > 0:
            bomb_targets = (actions == PLACE_BOMB).float()
            loss = loss + args.bomb_coef * masked_bce_loss(
                aux["bomb_logit"],
                bomb_targets,
                mask,
                pos_weight=bomb_pos_weight,
                focal_gamma=args.bomb_focal_gamma if args.focal_bomb_bce else 0.0,
            )
    return loss


def combine_action_logits(action_logits, aux, args):
    if not args.multi_head or not args.combine_heads or not aux:
        return action_logits
    head_logits = torch.cat([aux["movement_logits"], aux["bomb_logit"].unsqueeze(-1)], dim=-1)
    return (1.0 - args.head_combine_alpha) * action_logits + args.head_combine_alpha * head_logits


def chunk_sample_weights(actions, mask, bomb_batch_frac, escape_batch_frac=0.0):
    weights = np.ones((len(actions),), dtype=np.float32)
    valid = mask.astype(bool)
    has_bomb = np.any((actions == PLACE_BOMB) & valid, axis=1)
    if bomb_batch_frac and bomb_batch_frac > 0 and np.any(has_bomb):
        base_bomb_frac = float(has_bomb.mean())
        target = min(0.95, max(base_bomb_frac, float(bomb_batch_frac)))
        multiplier = (target / max(1e-6, base_bomb_frac)) / max(1e-6, (1.0 - target) / max(1e-6, 1.0 - base_bomb_frac))
        weights[has_bomb] *= max(1.0, multiplier)
    # Escape-specific sequence labels are not present in recurrent_bc_online_robust.npz;
    # keep the argument for compatible experiments and degrade to bomb-context weighting.
    if escape_batch_frac and escape_batch_frac > 0:
        weights[has_bomb] *= 1.0 + float(escape_batch_frac)
    return weights


def parse_sequence_type_weights(spec, type_names):
    if not spec:
        return {}
    result = {}
    name_to_id = {str(name): idx for idx, name in enumerate(type_names)}
    for item in spec.split(","):
        if not item.strip():
            continue
        name, value = item.split("=", 1)
        key = name.strip()
        if key not in name_to_id:
            raise ValueError(f"Unknown sequence type weight: {key}. Known: {sorted(name_to_id)}")
        result[name_to_id[key]] = float(value)
    return result


def sequence_type_sample_weights(types, target_weights):
    weights = np.ones((len(types),), dtype=np.float32)
    if not target_weights:
        return weights
    counts = Counter(int(v) for v in types)
    total = max(1, len(types))
    for type_id, target_frac in target_weights.items():
        current_frac = counts.get(int(type_id), 0) / total
        if current_frac <= 0:
            continue
        weights[types == int(type_id)] *= max(0.01, float(target_frac) / current_frac)
    return weights


@torch.no_grad()
def evaluate_model(model, loader, device, args=None):
    model.eval()
    confusion = np.zeros((NUM_ACTIONS, NUM_ACTIONS), dtype=np.int64)
    pred_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    losses = []
    for obs, actions, mask, _eps, _types in loader:
        obs = obs.to(device=device, dtype=torch.float32)
        actions = actions.to(device=device, dtype=torch.long)
        mask = mask.to(device=device, dtype=torch.float32)
        if args is not None and args.multi_head:
            logits, _, aux = model(obs, return_aux=True)
            logits = combine_action_logits(logits, aux, args)
        else:
            logits, _ = model(obs)
        losses.append(float(masked_ce_loss(logits, actions, mask).item()))
        pred = logits.argmax(dim=-1).detach().cpu().numpy()
        true = actions.detach().cpu().numpy()
        valid = mask.detach().cpu().numpy().astype(bool)
        for t, p in zip(true[valid], pred[valid]):
            confusion[int(t), int(p)] += 1
        pred_counts += np.bincount(pred[valid], minlength=NUM_ACTIONS)
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    per_action_recall = {
        str(i): float(confusion[i, i] / max(1, confusion[i].sum()))
        for i in range(NUM_ACTIONS)
    }
    per_action_precision = {
        str(i): float(confusion[i, i] / max(1, confusion[:, i].sum()))
        for i in range(NUM_ACTIONS)
    }
    pred_bomb = int(confusion[:, PLACE_BOMB].sum())
    true_bomb = int(confusion[PLACE_BOMB].sum())
    movement_true = int(confusion[:PLACE_BOMB].sum())
    movement_correct = int(np.trace(confusion[:PLACE_BOMB, :PLACE_BOMB]))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(correct / max(1, total)),
        "movement_accuracy": float(movement_correct / max(1, movement_true)),
        "per_action_accuracy": per_action_recall,
        "per_action_recall": per_action_recall,
        "per_action_precision": per_action_precision,
        "place_bomb_precision": float(confusion[PLACE_BOMB, PLACE_BOMB] / max(1, pred_bomb)),
        "place_bomb_recall": float(confusion[PLACE_BOMB, PLACE_BOMB] / max(1, true_bomb)),
        "predicted_place_bomb_frequency": float(pred_counts[PLACE_BOMB] / max(1, pred_counts.sum())),
        "predicted_action_distribution": {str(i): int(v) for i, v in enumerate(pred_counts)},
        "confusion_matrix": confusion.tolist(),
    }


def make_scheduler(name, optimizer, epochs):
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    raise ValueError(f"Unsupported scheduler: {name}")


def metric_score(metrics, args):
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    total_pred = max(1, int(confusion.sum()))
    max_single_action_frac = float(confusion.sum(axis=0).max() / total_pred)
    precision = float(metrics["place_bomb_precision"])
    recall = float(metrics["place_bomb_recall"])
    f1 = float(2.0 * precision * recall / max(1e-8, precision + recall))
    collapse_penalty = 0.0
    if metrics["predicted_place_bomb_frequency"] < args.min_pred_bomb_freq:
        collapse_penalty += args.zero_bomb_penalty
    if max_single_action_frac > args.max_single_action_frac:
        collapse_penalty += args.single_action_penalty * (max_single_action_frac - args.max_single_action_frac)
    if metrics["movement_accuracy"] < args.min_movement_acc:
        collapse_penalty += args.low_movement_penalty * (args.min_movement_acc - metrics["movement_accuracy"])
    composite = (
        metrics["movement_accuracy"]
        + 0.5 * f1
        + 0.2 * precision
        + 0.2 * recall
        - collapse_penalty
    )
    return {
        "val_acc": float(metrics["accuracy"]),
        "composite": float(composite),
        "bomb_f1": float(f1),
        "bomb_precision": precision,
        "bomb_recall": recall,
        "max_single_action_frac": max_single_action_frac,
        "collapse_penalty": float(collapse_penalty),
    }


def checkpoint_allowed(metrics, args):
    if args.selection_metric != "composite":
        return True
    if metrics["place_bomb_recall"] < args.min_bomb_recall:
        return False
    if metrics["place_bomb_precision"] < args.min_bomb_precision:
        return False
    if metrics["predicted_place_bomb_frequency"] < args.min_pred_bomb_freq:
        return False
    if metrics["movement_accuracy"] < args.min_movement_acc:
        return False
    return True


def evaluate_type_slices(model, dataset, type_names, device, args):
    results = {}
    types = np.asarray(dataset.sequence_types)
    for type_id in sorted(np.unique(types).tolist()):
        name = str(type_names[type_id] if type_id < len(type_names) else type_id)
        indices = np.where(types == type_id)[0].tolist()
        if not indices:
            continue
        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False)
        results[name] = evaluate_model(model, loader, device, args)
    return results


def build_loaders(args, data, mode):
    obs, actions, mask, source_eps, chunk_types = make_chunks(
        data,
        seq_len=args.seq_len,
        burn_in=args.burn_in,
        mode=mode,
        overfit_subset=args.overfit_subset,
        stride=args.stride,
    )
    train_idx, val_idx = split_chunks(source_eps, args.val_fraction, args.seed, args.episode_split)
    train_ds = ChunkDataset(
        obs[train_idx], actions[train_idx], mask[train_idx], source_eps[train_idx], chunk_types[train_idx]
    )
    val_ds = ChunkDataset(
        obs[val_idx], actions[val_idx], mask[val_idx], source_eps[val_idx], chunk_types[val_idx]
    )
    type_names = [str(v) for v in data["sequence_type_names"].tolist()] if "sequence_type_names" in data.files else DEFAULT_TYPE_NAMES
    sampler = None
    shuffle = True
    if args.bomb_batch_frac > 0 or args.escape_batch_frac > 0 or args.sequence_type_weights:
        sample_weights = chunk_sample_weights(
            actions[train_idx],
            mask[train_idx],
            args.bomb_batch_frac,
            args.escape_batch_frac,
        )
        sample_weights *= sequence_type_sample_weights(
            chunk_types[train_idx],
            parse_sequence_type_weights(args.sequence_type_weights, type_names),
        )
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    return {
        "obs": obs,
        "actions": actions,
        "mask": mask,
        "source_eps": source_eps,
        "chunk_types": chunk_types,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "train_loader": DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler),
        "metric_train_loader": DataLoader(train_ds, batch_size=args.batch_size, shuffle=False),
        "val_loader": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False),
        "type_names": type_names,
    }


def train_epochs(model, loaders, args, device, mode, epochs, learning_rate, stage_name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    scheduler = make_scheduler(args.scheduler, optimizer, epochs)
    weights = None
    if args.class_balance:
        bomb_multiplier = args.bomb_light_multiplier if _canonical_mode(mode) == "bomb_light" else args.bomb_weight_multiplier
        weights = torch.as_tensor(
            class_weights_for(
                loaders["actions"][loaders["train_idx"]],
                loaders["mask"][loaders["train_idx"]],
                args.max_class_weight,
                bomb_multiplier,
            ),
            device=device,
        )
    bomb_pos_weight = None
    if args.multi_head and args.bomb_pos_weight > 0:
        bomb_pos_weight = torch.as_tensor(float(args.bomb_pos_weight), device=device)
    elif args.multi_head and args.class_balance:
        train_actions = loaders["actions"][loaders["train_idx"]]
        train_mask = loaders["mask"][loaders["train_idx"]].astype(bool)
        valid_actions = train_actions[train_mask]
        positives = max(1, int(np.sum(valid_actions == PLACE_BOMB)))
        negatives = max(1, int(np.sum(valid_actions != PLACE_BOMB)))
        bomb_pos_weight = torch.as_tensor(min(args.max_bomb_pos_weight, negatives / positives), device=device)
    history = []
    best = {
        "val_acc": {"score": -1.0, "state": None, "metrics": None},
        "composite": {"score": -1.0, "state": None, "metrics": None},
        "bomb_f1": {"score": -1.0, "state": None, "metrics": None},
    }
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch_obs, batch_actions, batch_mask, _eps, _types in loaders["train_loader"]:
            batch_obs = batch_obs.to(device=device, dtype=torch.float32)
            batch_actions = batch_actions.to(device=device, dtype=torch.long)
            batch_mask = batch_mask.to(device=device, dtype=torch.float32)
            if args.multi_head:
                logits, _, aux = model(batch_obs, return_aux=True)
            else:
                logits, _ = model(batch_obs)
                aux = {}
            action_logits_for_loss = combine_action_logits(logits, aux, args)
            loss = multi_head_loss(action_logits_for_loss, aux, batch_actions, batch_mask, weights, args, bomb_pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
        train_metrics = evaluate_model(model, loaders["metric_train_loader"], device, args)
        val_metrics = evaluate_model(model, loaders["val_loader"], device, args)
        scores = metric_score(val_metrics, args)
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_metrics["accuracy"])
            else:
                scheduler.step()
        row = {
            "stage": stage_name,
            "mode": mode,
            "epoch": epoch + 1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(np.mean(losses)),
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_movement_accuracy": train_metrics["movement_accuracy"],
            "val_movement_accuracy": val_metrics["movement_accuracy"],
            "train_bomb_precision": train_metrics["place_bomb_precision"],
            "train_bomb_recall": train_metrics["place_bomb_recall"],
            "val_bomb_precision": val_metrics["place_bomb_precision"],
            "val_bomb_recall": val_metrics["place_bomb_recall"],
            "val_bomb_f1": scores["bomb_f1"],
            "val_composite_score": scores["composite"],
            "val_max_single_action_frac": scores["max_single_action_frac"],
            "val_collapse_penalty": scores["collapse_penalty"],
            "train_predicted_action_distribution": train_metrics["predicted_action_distribution"],
            "val_predicted_action_distribution": val_metrics["predicted_action_distribution"],
        }
        history.append(row)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if scores["val_acc"] > best["val_acc"]["score"]:
            best["val_acc"] = {"score": scores["val_acc"], "state": state, "metrics": val_metrics}
        if scores["bomb_f1"] > best["bomb_f1"]["score"]:
            best["bomb_f1"] = {"score": scores["bomb_f1"], "state": state, "metrics": val_metrics}
        if checkpoint_allowed(val_metrics, args) and scores["composite"] > best["composite"]["score"]:
            best["composite"] = {"score": scores["composite"], "state": state, "metrics": val_metrics}
        print(json.dumps(row))
    return history, best


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = np.load(args.dataset, allow_pickle=True)
    first_loaders = build_loaders(args, data, args.mode)
    obs = first_loaders["obs"]
    device = torch.device(args.device)
    model = StandaloneBomberCnnLstm(
        in_channels=int(obs.shape[2]),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout,
        layer_norm=args.layer_norm,
        multi_head=args.multi_head,
    ).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
    history = []
    if args.schedule == "two_stage":
        stages = [
            ("movement", "movement", args.stage_a_epochs, args.stage_a_lr),
            ("bomb_light", "bomb_light", args.stage_b_epochs, args.stage_b_lr),
            ("full", "all", args.stage_c_epochs, args.stage_c_lr),
        ]
    else:
        stages = [("single", args.mode, args.epochs, args.learning_rate)]
    best_states = {
        "val_acc": {"score": -1.0, "state": None, "metrics": None},
        "composite": {"score": -1.0, "state": None, "metrics": None},
        "bomb_f1": {"score": -1.0, "state": None, "metrics": None},
    }
    stage_reports = []
    final_loaders = first_loaders
    for stage_name, mode, epochs, lr in stages:
        if epochs <= 0:
            continue
        loaders = build_loaders(args, data, mode)
        final_loaders = loaders
        stage_history, stage_best = train_epochs(
            model, loaders, args, device, mode, epochs, lr, stage_name
        )
        history.extend(stage_history)
        stage_reports.append(
            {
                "stage": stage_name,
                "mode": mode,
                "epochs": int(epochs),
                "learning_rate": float(lr),
                "chunks": int(len(loaders["actions"])),
                "train_chunks": int(len(loaders["train_idx"])),
                "val_chunks": int(len(loaders["val_idx"])),
                "train_episodes": int(len(np.unique(loaders["source_eps"][loaders["train_idx"]]))),
                "val_episodes": int(len(np.unique(loaders["source_eps"][loaders["val_idx"]]))),
                "train_action_distribution": action_distribution(
                    loaders["actions"][loaders["train_idx"]], loaders["mask"][loaders["train_idx"]]
                ),
                "val_action_distribution": action_distribution(
                    loaders["actions"][loaders["val_idx"]], loaders["mask"][loaders["val_idx"]]
                ),
                "train_sequence_type_distribution": sequence_type_distribution(
                    loaders["chunk_types"][loaders["train_idx"]], loaders["type_names"]
                ),
                "val_sequence_type_distribution": sequence_type_distribution(
                    loaders["chunk_types"][loaders["val_idx"]], loaders["type_names"]
                ),
            }
        )
        for key in best_states:
            if stage_best[key]["score"] > best_states[key]["score"]:
                best_states[key] = stage_best[key]
    selected_key = args.selection_metric
    if args.restore_best and best_states[selected_key]["state"] is not None:
        model.load_state_dict(best_states[selected_key]["state"])
    final_train = evaluate_model(model, final_loaders["metric_train_loader"], device, args)
    final_val = evaluate_model(model, final_loaders["val_loader"], device, args)
    final_train_slices = evaluate_type_slices(model, final_loaders["train_ds"], final_loaders["type_names"], device, args)
    final_val_slices = evaluate_type_slices(model, final_loaders["val_ds"], final_loaders["type_names"], device, args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "in_channels": int(obs.shape[2]),
            "embedding_dim": int(args.embedding_dim),
            "hidden_size": int(args.hidden_size),
            "num_lstm_layers": int(args.num_lstm_layers),
            "dropout": float(args.dropout),
            "layer_norm": bool(args.layer_norm),
            "multi_head": bool(args.multi_head),
            "combine_heads": bool(args.combine_heads),
            "head_combine_alpha": float(args.head_combine_alpha),
            "num_actions": NUM_ACTIONS,
            "seq_len": int(args.seq_len),
            "burn_in": int(args.burn_in),
        },
        "dataset": args.dataset,
    }
    torch.save(checkpoint, output)
    summary = {
        "dataset": args.dataset,
        "output": str(output),
        "chunks": int(len(final_loaders["actions"])),
        "train_chunks": int(len(final_loaders["train_idx"])),
        "val_chunks": int(len(final_loaders["val_idx"])),
        "train_episodes": int(len(np.unique(final_loaders["source_eps"][final_loaders["train_idx"]]))),
        "val_episodes": int(len(np.unique(final_loaders["source_eps"][final_loaders["val_idx"]]))),
        "seq_len": int(args.seq_len),
        "stride": int(args.stride),
        "burn_in": int(args.burn_in),
        "overfit_subset": int(args.overfit_subset),
        "mode": args.mode,
        "schedule": args.schedule,
        "selection_metric": args.selection_metric,
        "best_scores": {
            key: {
                "score": float(value["score"]),
                "metrics": value["metrics"],
            }
            for key, value in best_states.items()
        },
        "stage_reports": stage_reports,
        "class_balance": bool(args.class_balance),
        "train_action_distribution": action_distribution(
            final_loaders["actions"][final_loaders["train_idx"]], final_loaders["mask"][final_loaders["train_idx"]]
        ),
        "val_action_distribution": action_distribution(
            final_loaders["actions"][final_loaders["val_idx"]], final_loaders["mask"][final_loaders["val_idx"]]
        ),
        "train_sequence_type_distribution": sequence_type_distribution(
            final_loaders["chunk_types"][final_loaders["train_idx"]], final_loaders["type_names"]
        ),
        "val_sequence_type_distribution": sequence_type_distribution(
            final_loaders["chunk_types"][final_loaders["val_idx"]], final_loaders["type_names"]
        ),
        "final_train": final_train,
        "final_val": final_val,
        "final_train_slices": final_train_slices,
        "final_val_slices": final_val_slices,
        "history": history,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for key, value in best_states.items():
        if value["state"] is None:
            continue
        best_path = output.with_name(f"{output.stem}_best_by_{key}{output.suffix}")
        best_checkpoint = dict(checkpoint)
        best_checkpoint["model_state_dict"] = value["state"]
        best_checkpoint["selected_by"] = key
        best_checkpoint["selected_score"] = float(value["score"])
        torch.save(best_checkpoint, best_path)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Standalone PyTorch CNN-LSTM BC for Bomberland.")
    parser.add_argument("--dataset", default="ml/datasets/recurrent_bc_online_robust.npz")
    parser.add_argument("--output", default="ml/checkpoints/rl_agent_recurrent/standalone_bc_full.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--burn_in", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_lstm_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--layer_norm", action="store_true")
    parser.add_argument("--multi_head", action="store_true")
    parser.add_argument("--combine_heads", action="store_true")
    parser.add_argument("--head_combine_alpha", type=float, default=0.5)
    parser.add_argument("--action_coef", type=float, default=1.0)
    parser.add_argument("--movement_coef", type=float, default=1.0)
    parser.add_argument("--bomb_coef", type=float, default=1.0)
    parser.add_argument("--bomb_pos_weight", type=float, default=0.0)
    parser.add_argument("--max_bomb_pos_weight", type=float, default=20.0)
    parser.add_argument("--focal_bomb_bce", action="store_true")
    parser.add_argument("--bomb_focal_gamma", type=float, default=2.0)
    parser.add_argument("--class_balance", action="store_true")
    parser.add_argument("--bomb_weight_multiplier", type=float, default=1.0)
    parser.add_argument("--bomb_light_multiplier", type=float, default=0.5)
    parser.add_argument("--max_class_weight", type=float, default=10.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="none")
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--restore_best", action="store_true")
    parser.add_argument("--selection_metric", choices=["val_acc", "composite", "bomb_f1"], default="val_acc")
    parser.add_argument("--min_bomb_recall", type=float, default=0.0)
    parser.add_argument("--min_bomb_precision", type=float, default=0.0)
    parser.add_argument("--min_pred_bomb_freq", type=float, default=0.0)
    parser.add_argument("--min_movement_acc", type=float, default=0.0)
    parser.add_argument("--max_single_action_frac", type=float, default=0.8)
    parser.add_argument("--zero_bomb_penalty", type=float, default=0.2)
    parser.add_argument("--single_action_penalty", type=float, default=1.0)
    parser.add_argument("--low_movement_penalty", type=float, default=1.0)
    parser.add_argument("--bomb_batch_frac", type=float, default=0.0)
    parser.add_argument("--escape_batch_frac", type=float, default=0.0)
    parser.add_argument("--sequence_type_weights", default="")
    parser.add_argument("--overfit_subset", type=int, default=0)
    parser.add_argument("--episode_split", action="store_true", default=True)
    parser.add_argument("--random_chunk_split", action="store_true")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--mode", choices=["all", "movement", "movement_only", "bomb_light"], default="all")
    parser.add_argument("--schedule", choices=["single", "two_stage"], default="single")
    parser.add_argument("--stage_a_epochs", type=int, default=30)
    parser.add_argument("--stage_b_epochs", type=int, default=15)
    parser.add_argument("--stage_c_epochs", type=int, default=8)
    parser.add_argument("--stage_a_lr", type=float, default=3e-4)
    parser.add_argument("--stage_b_lr", type=float, default=2e-4)
    parser.add_argument("--stage_c_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=9950)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.lr is not None:
        args.learning_rate = args.lr
    if args.grad_clip is None:
        args.grad_clip = args.max_grad_norm
    if args.random_chunk_split:
        args.episode_split = False
    train(args)


if __name__ == "__main__":
    main()
