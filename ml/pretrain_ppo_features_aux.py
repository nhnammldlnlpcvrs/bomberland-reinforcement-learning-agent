from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.aux_model import BomberAuxModel
from agent.rl_agent_pure.model import BomberFeaturesExtractor


def _standardize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(values.mean())
    std = float(values.std() + 1e-6)
    return ((values - mean) / std).astype(np.float32), mean, std


def _load_dataset(path: str, seed: int, val_fraction: float):
    data = np.load(path)
    obs = data["observations"].astype(np.float32)
    blast_distance_raw = data["blast_corridor_distance"].astype(np.float32)
    blast_distance_raw = np.where(blast_distance_raw < 0, 14.0, blast_distance_raw)
    boxes, boxes_mean, boxes_std = _standardize(data["boxes_destroyed_future"].astype(np.float32))
    reachable, reach_mean, reach_std = _standardize(data["reachable_area_delta"].astype(np.float32))
    safe_tiles, safe_mean, safe_std = _standardize(data["safe_tiles_after_bomb_count"].astype(np.float32))
    blast_distance, blast_mean, blast_std = _standardize(blast_distance_raw)
    returns = data["discounted_returns"].astype(np.float32)
    returns_norm, ret_mean, ret_std = _standardize(returns)

    if "train_split" in data:
        train_idx = np.flatnonzero(data["train_split"].astype(np.int8) > 0)
        val_idx = np.flatnonzero(data["train_split"].astype(np.int8) <= 0)
    else:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(obs))
        rng.shuffle(idx)
        split = int(len(idx) * (1.0 - val_fraction))
        train_idx, val_idx = idx[:split], idx[split:]

    tensors = [
        torch.as_tensor(obs, dtype=torch.float32),
        torch.as_tensor(data["death_within_7"].astype(np.float32), dtype=torch.float32),
        torch.as_tensor(data["escaped_blast"].astype(np.float32), dtype=torch.float32),
        torch.as_tensor(data["has_escape_path_now"].astype(np.float32), dtype=torch.float32),
        torch.as_tensor(data["has_escape_after_bomb"].astype(np.float32), dtype=torch.float32),
        torch.as_tensor(data["trapped_if_bomb"].astype(np.float32), dtype=torch.float32),
        torch.as_tensor(data["in_future_blast"].astype(np.float32), dtype=torch.float32),
        torch.as_tensor(boxes, dtype=torch.float32),
        torch.as_tensor(reachable, dtype=torch.float32),
        torch.as_tensor(safe_tiles, dtype=torch.float32),
        torch.as_tensor(blast_distance, dtype=torch.float32),
        torch.as_tensor(returns_norm, dtype=torch.float32),
        torch.as_tensor(returns, dtype=torch.float32),
    ]
    normalization = {
        "boxes_mean": boxes_mean,
        "boxes_std": boxes_std,
        "reachable_mean": reach_mean,
        "reachable_std": reach_std,
        "safe_tiles_mean": safe_mean,
        "safe_tiles_std": safe_std,
        "blast_distance_mean": blast_mean,
        "blast_distance_std": blast_std,
        "return_mean": ret_mean,
        "return_std": ret_std,
    }
    return tensors, train_idx, val_idx, normalization


def _make_loader(tensors, indices, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(*(tensor[indices] for tensor in tensors))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _aux_loss(out: dict[str, torch.Tensor], batch: list[torch.Tensor], args) -> torch.Tensor:
    (
        _obs,
        death,
        escaped,
        escape_available,
        bomb_escape_available,
        trapped,
        future_blast,
        boxes,
        reachable,
        safe_tiles,
        blast_distance,
        returns_norm,
        _returns_raw,
    ) = batch
    return (
        args.death_weight * F.binary_cross_entropy_with_logits(out["death_logit"], death)
        + args.escape_weight * F.binary_cross_entropy_with_logits(out["escape_logit"], escaped)
        + args.escape_available_weight * F.binary_cross_entropy_with_logits(out["escape_available_logit"], escape_available)
        + args.bomb_escape_available_weight * F.binary_cross_entropy_with_logits(out["bomb_escape_available_logit"], bomb_escape_available)
        + args.trapped_weight * F.binary_cross_entropy_with_logits(out["trapped_if_bomb_logit"], trapped)
        + args.future_blast_weight * F.binary_cross_entropy_with_logits(out["future_blast_logit"], future_blast)
        + args.box_weight * F.smooth_l1_loss(out["box_value"], boxes)
        + args.reachable_weight * F.smooth_l1_loss(out["reachable_delta"], reachable)
        + args.safe_tiles_weight * F.smooth_l1_loss(out["safe_tiles_after_bomb"], safe_tiles)
        + args.blast_distance_weight * F.smooth_l1_loss(out["blast_corridor_distance"], blast_distance)
        + args.return_weight * F.smooth_l1_loss(out["return"], returns_norm)
    )


def _binary_summary(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits)
    pred = probs >= 0.5
    target = labels >= 0.5
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    fn = int((~pred & target).sum())
    tn = int((~pred & ~target).sum())
    return {
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
        "positive_rate": float(target.float().mean()),
    }


def _evaluate_aux(model: BomberAuxModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    buckets = {key: [] for key in ("death", "escaped", "escape_available", "bomb_escape_available", "trapped", "future_blast")}
    labels = {key: [] for key in buckets}
    reg_abs = {key: [] for key in ("box", "reachable", "safe_tiles", "blast_distance", "return")}
    with torch.no_grad():
        for batch in loader:
            batch = [item.to(device) for item in batch]
            out = model(batch[0])
            pairs = {
                "death": (out["death_logit"], batch[1]),
                "escaped": (out["escape_logit"], batch[2]),
                "escape_available": (out["escape_available_logit"], batch[3]),
                "bomb_escape_available": (out["bomb_escape_available_logit"], batch[4]),
                "trapped": (out["trapped_if_bomb_logit"], batch[5]),
                "future_blast": (out["future_blast_logit"], batch[6]),
            }
            for key, (logit, target) in pairs.items():
                buckets[key].append(logit.cpu())
                labels[key].append(target.cpu())
            reg_abs["box"].append(torch.abs(out["box_value"] - batch[7]).cpu())
            reg_abs["reachable"].append(torch.abs(out["reachable_delta"] - batch[8]).cpu())
            reg_abs["safe_tiles"].append(torch.abs(out["safe_tiles_after_bomb"] - batch[9]).cpu())
            reg_abs["blast_distance"].append(torch.abs(out["blast_corridor_distance"] - batch[10]).cpu())
            reg_abs["return"].append(torch.abs(out["return"] - batch[11]).cpu())
    metrics = {key: _binary_summary(torch.cat(buckets[key]), torch.cat(labels[key])) for key in buckets}
    for key, values in reg_abs.items():
        metrics[f"{key}_mae_norm"] = float(torch.cat(values).mean())
    return metrics


def _load_ppo(path: str, device: str) -> PPO:
    policy_kwargs = {
        "features_extractor_class": BomberFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    return PPO.load(path, device=device, custom_objects={"policy_kwargs": policy_kwargs})


def _policy_logits(model: PPO, obs: torch.Tensor) -> torch.Tensor:
    model.policy.eval()
    with torch.no_grad():
        dist = model.policy.get_distribution(obs)
        return dist.distribution.logits.detach().cpu()


def _logit_drift(model: PPO, reference: torch.Tensor, obs: torch.Tensor) -> dict:
    after = _policy_logits(model, obs)
    delta = torch.abs(after - reference)
    return {
        "max_logit_delta": float(delta.max()),
        "mean_logit_delta": float(delta.mean()),
    }


def _freeze_for_value_only(model: PPO) -> list[torch.nn.Parameter]:
    trainable = []
    for name, param in model.policy.named_parameters():
        allow = ("value_net" in name) or ("mlp_extractor.value_net" in name)
        param.requires_grad = allow
        if allow:
            trainable.append(param)
    return trainable


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    tensors, train_idx, val_idx, normalization = _load_dataset(args.dataset, args.seed, args.val_fraction)
    train_loader = _make_loader(tensors, train_idx, args.batch_size, shuffle=True)
    val_loader = _make_loader(tensors, val_idx, args.batch_size, shuffle=False)

    aux = BomberAuxModel(features_dim=args.features_dim).to(device)
    aux_optimizer = torch.optim.Adam(aux.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    aux_history = []
    for epoch in range(args.epochs):
        aux.train()
        running = 0.0
        count = 0
        for batch in train_loader:
            batch = [item.to(device) for item in batch]
            out = aux(batch[0])
            loss = _aux_loss(out, batch, args)
            aux_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(aux.parameters(), 1.0)
            aux_optimizer.step()
            running += float(loss.item()) * len(batch[0])
            count += len(batch[0])
        metrics = _evaluate_aux(aux, val_loader, device)
        metrics["epoch"] = epoch + 1
        metrics["aux_train_loss"] = running / max(1, count)
        aux_history.append(metrics)
        print(json.dumps({"aux": metrics}))

    aux_output = Path(args.aux_output)
    aux_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": aux.state_dict(),
            "features_dim": args.features_dim,
            "normalization": normalization,
            "metrics": aux_history[-1] if aux_history else {},
            "history": aux_history,
        },
        aux_output,
    )

    ppo_metrics = {"enabled": bool(args.update_ppo_value)}
    if args.update_ppo_value:
        ppo = _load_ppo(args.base_policy, args.device)
        val_obs = tensors[0][val_idx[: min(args.drift_samples, len(val_idx))]].to(device)
        before_logits = _policy_logits(ppo, val_obs)
        trainable = _freeze_for_value_only(ppo)
        if not trainable:
            ppo_metrics.update({"saved": False, "reason": "no_value_parameters_found"})
        else:
            optimizer = torch.optim.Adam(trainable, lr=args.value_learning_rate, weight_decay=args.weight_decay)
            value_history = []
            for epoch in range(args.value_epochs):
                running = 0.0
                count = 0
                ppo.policy.train()
                for batch in train_loader:
                    obs = batch[0].to(device)
                    target = (batch[12] if not args.standardize_value_targets else batch[11]).to(device)
                    values = ppo.policy.predict_values(obs).flatten()
                    loss = F.smooth_l1_loss(values, target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 0.5)
                    optimizer.step()
                    running += float(loss.item()) * len(obs)
                    count += len(obs)
                with torch.no_grad():
                    val_losses = []
                    for batch in val_loader:
                        obs = batch[0].to(device)
                        target = (batch[12] if not args.standardize_value_targets else batch[11]).to(device)
                        values = ppo.policy.predict_values(obs).flatten()
                        val_losses.append(F.smooth_l1_loss(values, target).detach().cpu())
                    value_history.append(
                        {
                            "epoch": epoch + 1,
                            "value_train_loss": running / max(1, count),
                            "value_val_loss": float(torch.stack(val_losses).mean()),
                        }
                    )
                    print(json.dumps({"ppo_value": value_history[-1]}))
            drift = _logit_drift(ppo, before_logits, val_obs)
            ppo_metrics.update({"value_history": value_history, **drift})
            if drift["max_logit_delta"] <= args.max_logit_delta:
                output = Path(args.ppo_output)
                output.parent.mkdir(parents=True, exist_ok=True)
                ppo.save(str(output))
                ppo_metrics.update({"saved": True, "path": str(output)})
            else:
                ppo_metrics.update({"saved": False, "reason": "actor_logit_drift_exceeded"})

    report = {
        "dataset": str(args.dataset),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "aux_output": str(aux_output),
        "aux_metrics": aux_history[-1] if aux_history else {},
        "ppo_value": ppo_metrics,
    }
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Pretrain auxiliary representation and optional PPO critic without actor updates.")
    parser.add_argument("--dataset", default="ml/datasets/aux_pretrain_dataset_v3.npz")
    parser.add_argument("--base_policy", default="ml/checkpoints/rl_agent_pure/ppo_bomb_selector_best.zip")
    parser.add_argument("--aux_output", default="ml/checkpoints/rl_agent_pure/aux_pretrained_features.pt")
    parser.add_argument("--ppo_output", default="ml/checkpoints/rl_agent_pure/ppo_aux_pretrained_critic.zip")
    parser.add_argument("--report_output", default="logs/aux_pretrain_report_v3.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--value_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--features_dim", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--value_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7800)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--update_ppo_value", action="store_true")
    parser.add_argument("--standardize_value_targets", action="store_true")
    parser.add_argument("--drift_samples", type=int, default=512)
    parser.add_argument("--max_logit_delta", type=float, default=1e-6)
    parser.add_argument("--death_weight", type=float, default=1.0)
    parser.add_argument("--escape_weight", type=float, default=0.5)
    parser.add_argument("--escape_available_weight", type=float, default=1.5)
    parser.add_argument("--bomb_escape_available_weight", type=float, default=1.5)
    parser.add_argument("--trapped_weight", type=float, default=1.5)
    parser.add_argument("--future_blast_weight", type=float, default=1.0)
    parser.add_argument("--box_weight", type=float, default=0.5)
    parser.add_argument("--reachable_weight", type=float, default=0.25)
    parser.add_argument("--safe_tiles_weight", type=float, default=0.25)
    parser.add_argument("--blast_distance_weight", type=float, default=0.25)
    parser.add_argument("--return_weight", type=float, default=0.25)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
