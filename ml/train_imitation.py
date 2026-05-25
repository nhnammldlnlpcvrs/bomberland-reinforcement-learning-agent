"""Imitation pretraining for hybrid PPO policy.

Trains PPOPolicy to predict teacher actions with safe action masking.
Tracks per-category accuracy, bomb precision/recall, and source-agent breakdown.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.hybrid_ppo.ppo_policy import PPOPolicy, NUM_CHANNELS, NUM_ACTIONS

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ACTION_NAMES = ["STOP", "LEFT", "RIGHT", "UP", "DOWN", "BOMB"]
BOMB_IDX = 5
STOP_IDX = 0


class ImitationDataset:
    """Loads .npz dataset and serves (obs, mask, action, meta) tuples."""

    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.observations = torch.from_numpy(data["observations"].copy()).float()
        self.masks = torch.from_numpy(data["safe_action_masks"].copy()).bool()
        self.actions = torch.from_numpy(data["actions"].copy()).long()
        self.categories = np.array(data["categories"].copy())
        self.source_agents = np.array(data["source_agents"].copy())
        self.bomb_candidates = torch.from_numpy(data["bomb_candidates"].copy()).bool()
        self.metadata = json.loads(str(data["metadata_json"]))
        self._n = len(self.actions)

    def __len__(self):
        return self._n

    def get_split(self, indices):
        return _ImitationSubset(self, indices)

    def split_train_val(self, val_frac=0.2, seed=42):
        rng = np.random.default_rng(seed)
        n_val = max(1, int(self._n * val_frac))
        perm = rng.permutation(self._n)
        val_idx = set(perm[:n_val].tolist())
        train_idx = [i for i in range(self._n) if i not in val_idx]
        return self.get_split(train_idx), self.get_split(list(val_idx))


class _ImitationSubset:
    def __init__(self, parent, indices):
        self.parent = parent
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            self.parent.observations[i],
            self.parent.masks[i],
            self.parent.actions[i],
            self.parent.categories[i],
            self.parent.source_agents[i],
            self.parent.bomb_candidates[i],
        )


def _masked_logits(logits, mask):
    result = logits.clone()
    result[~mask] = -1e9
    return result


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    top2_correct = 0
    per_action_correct = {a: 0 for a in range(6)}
    per_action_total = {a: 0 for a in range(6)}
    per_cat_correct = {}
    per_cat_total = {}
    per_src_correct = {}
    per_src_total = {}
    bomb_tp, bomb_fp, bomb_fn = 0, 0, 0

    for obs, mask, action, cat, src, _bc in loader:
        obs = obs.to(device)
        mask = mask.to(device)
        action = action.to(device)
        bs = obs.size(0)

        logits, _ = model(obs)
        masked = _masked_logits(logits, mask)
        pred = torch.argmax(masked, dim=-1)
        correct_mask = pred == action

        correct += int(correct_mask.sum().item())
        total += bs

        _, top2 = torch.topk(masked, k=2, dim=-1)
        top2_correct += int((top2 == action.unsqueeze(-1)).any(dim=-1).sum().item())

        for a in range(6):
            a_mask = action == a
            per_action_total[a] += int(a_mask.sum().item())
            per_action_correct[a] += int((correct_mask & a_mask).sum().item())

        for j in range(bs):
            c = cat[j]
            s = src[j]
            per_cat_total[c] = per_cat_total.get(c, 0) + 1
            per_src_total[s] = per_src_total.get(s, 0) + 1
            if correct_mask[j]:
                per_cat_correct[c] = per_cat_correct.get(c, 0) + 1
                per_src_correct[s] = per_src_correct.get(s, 0) + 1

        pred_bomb = pred == BOMB_IDX
        true_bomb = action == BOMB_IDX
        bomb_tp += int((pred_bomb & true_bomb).sum().item())
        bomb_fp += int((pred_bomb & ~true_bomb).sum().item())
        bomb_fn += int((~pred_bomb & true_bomb).sum().item())

    metrics = {
        "accuracy": correct / max(1, total),
        "top2_accuracy": top2_correct / max(1, total),
        "total": total,
        "bomb_precision": bomb_tp / max(1, bomb_tp + bomb_fp),
        "bomb_recall": bomb_tp / max(1, bomb_tp + bomb_fn),
        "bomb_f1": 0.0,
    }
    denom = bomb_tp + bomb_fp + bomb_fn
    if denom > 0:
        p = metrics["bomb_precision"]
        r = metrics["bomb_recall"]
        metrics["bomb_f1"] = 2 * p * r / max(1e-9, p + r)

    metrics["per_action"] = {}
    for a in range(6):
        ta = per_action_total[a]
        metrics["per_action"][ACTION_NAMES[a]] = (
            per_action_correct[a] / max(1, ta), ta
        )

    metrics["per_category"] = {}
    for c in sorted(per_cat_total.keys()):
        metrics["per_category"][c] = (
            per_cat_correct.get(c, 0) / max(1, per_cat_total[c]),
            per_cat_total[c],
        )

    metrics["per_source"] = {}
    for s in sorted(per_src_total.keys()):
        metrics["per_source"][s] = (
            per_src_correct.get(s, 0) / max(1, per_src_total[s]),
            per_src_total[s],
        )

    return metrics


def _print_metrics(metrics, title="METRICS"):
    print(f"\n{title}:")
    print(f"  Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Top-2 Acc:    {metrics['top2_accuracy']:.4f} ({metrics['top2_accuracy']*100:.1f}%)")
    print(f"  Val samples:  {metrics['total']}")

    print(f"\n  Per-action:")
    for name, (acc, count) in metrics["per_action"].items():
        print(f"    {name:8s}: acc={acc:.3f}  n={count}")

    print(f"\n  Bomb:")
    print(f"    Precision: {metrics['bomb_precision']:.3f}")
    print(f"    Recall:    {metrics['bomb_recall']:.3f}")
    print(f"    F1:        {metrics['bomb_f1']:.3f}")

    print(f"\n  Per-category:")
    for cat, (acc, count) in sorted(metrics["per_category"].items()):
        print(f"    {cat:25s}: acc={acc:.3f}  n={count}")

    print(f"\n  Per source-agent:")
    for src, (acc, count) in sorted(metrics["per_source"].items()):
        print(f"    {src:35s}: acc={acc:.3f}  n={count}")


def _benchmark_latency(model):
    model.eval()
    sample_obs = torch.randn(1, NUM_CHANNELS, 13, 13)
    sample_mask = torch.ones(1, 6, dtype=torch.bool)
    for _ in range(10):
        with torch.no_grad():
            model.get_action_logits(sample_obs, sample_mask)

    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.get_action_logits(sample_obs, sample_mask)
        times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1000  # ms
    print(f"\n  Latency (CPU):")
    print(f"    Mean: {arr.mean():.3f} ms")
    print(f"    P95:  {np.percentile(arr, 95):.3f} ms")
    print(f"    P99:  {np.percentile(arr, 99):.3f} ms")
    print(f"    Max:  {arr.max():.3f} ms")
    return arr.mean()


def _save_checkpoint(model, optimizer, epoch, metrics, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": {k: v for k, v in metrics.items() if k != "per_action"},
    }, path)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading {args.dataset}...")
    full = ImitationDataset(args.dataset)
    print(f"  Samples: {len(full)}")
    meta_summary = {k: full.metadata[k] for k in
                    ["action_names", "num_channels", "total_matches",
                     "raw_counts", "kept_counts"] if k in full.metadata}
    print(f"  Metadata: {json.dumps(meta_summary, indent=2)}")

    train_set, val_set = full.split_train_val(val_frac=args.val_split, seed=args.seed)
    print(f"  Train: {len(train_set)}, Val: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2,
                            shuffle=False)

    model = PPOPolicy(input_channels=NUM_CHANNELS, num_actions=NUM_ACTIONS)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for obs, mask, action, _cat, _src, _bc in train_loader:
            obs = obs.to(device)
            mask = mask.to(device)
            action = action.to(device)

            logits, _ = model(obs)
            masked_logits = _masked_logits(logits, mask)
            loss = F.cross_entropy(masked_logits, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * obs.size(0)
            pred = torch.argmax(masked_logits, dim=-1)
            train_correct += int((pred == action).sum().item())
            train_total += obs.size(0)

        train_acc = train_correct / max(1, train_total)
        avg_loss = train_loss / max(1, train_total)
        val_metrics = evaluate(model, val_loader, device)

        # Scheduler on val error (1 - accuracy)
        scheduler.step(1.0 - val_metrics["accuracy"])

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_acc={val_metrics['accuracy']:.3f}  "
              f"top2={val_metrics['top2_accuracy']:.3f}  "
              f"bomb_r={val_metrics['bomb_recall']:.3f}  "
              f"bomb_p={val_metrics['bomb_precision']:.3f}  "
              f"stop_acc={val_metrics['per_action']['STOP'][0]:.3f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            _save_checkpoint(model, optimizer, epoch, val_metrics, args.output)
        else:
            patience_counter += 1

        if patience_counter >= args.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # ---- Final evaluation ----
    best = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    final_metrics = evaluate(model, val_loader, device)

    print(f"\n{'='*60}")
    print(f"FINAL METRICS  (best epoch {best['epoch']})")
    print(f"{'='*60}")
    _print_metrics(final_metrics)

    mean_latency = _benchmark_latency(model)

    # ---- Verdict ----
    print(f"\n{'='*60}")
    print("ACCEPTANCE CHECK")
    print(f"{'='*60}")
    acc_ok = final_metrics["accuracy"] >= 0.55
    top2_ok = final_metrics["top2_accuracy"] >= 0.80
    bomb_ok = final_metrics["bomb_recall"] > 0.01
    latency_ok = mean_latency < 1.0
    checks = [
        ("Overall acc >= 55%", acc_ok, f"{final_metrics['accuracy']*100:.1f}%"),
        ("Top-2 acc >= 80%", top2_ok, f"{final_metrics['top2_accuracy']*100:.1f}%"),
        ("Bomb recall > 0", bomb_ok, f"{final_metrics['bomb_recall']:.4f}"),
        ("Latency < 1ms", latency_ok, f"{mean_latency:.3f} ms"),
    ]
    all_ok = True
    for label, ok, val in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: {val}")
        if not ok:
            all_ok = False

    if all_ok:
        print(f"\n  VERDICT: imitation model is good enough for PPO fine-tune")
    else:
        print(f"\n  VERDICT: imitation model needs improvement before PPO fine-tune")

    return final_metrics, all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Imitation pretraining for hybrid PPO policy.")
    parser.add_argument("--dataset", required=True,
                        help="Path to .npz imitation dataset")
    parser.add_argument("--output", default="ml/checkpoints/hybrid_ppo/imitation_cnn.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr_patience", type=int, default=4)
    parser.add_argument("--early_stop", type=int, default=8)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
