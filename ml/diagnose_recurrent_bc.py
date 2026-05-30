from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from sb3_contrib import RecurrentPPO
except ImportError as exc:  # pragma: no cover
    raise SystemExit("sb3-contrib is required. Install requirements.txt first.") from exc

from agent.rl_agent_pure.constants import NUM_ACTIONS, PLACE_BOMB
from agent.rl_agent_pure.model import BomberFeaturesExtractor
from ml.pretrain_recurrent_bc import _make_chunks, _metrics


def _action_distribution(actions):
    values = np.asarray(actions, dtype=np.int64).reshape(-1)
    return {str(i): int(v) for i, v in enumerate(np.bincount(values, minlength=NUM_ACTIONS))}


def _transition_distribution(actions, mask):
    counts = {}
    for ep_actions, ep_mask in zip(actions, mask):
        valid = ep_actions[ep_mask.astype(bool)]
        for prev, nxt in zip(valid[:-1], valid[1:]):
            key = f"{int(prev)}->{int(nxt)}"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:30])


def _split_indices(n_episodes, val_fraction, seed):
    rng = np.random.default_rng(seed)
    eps = rng.permutation(np.arange(n_episodes))
    split = int(n_episodes * (1.0 - val_fraction))
    return eps[:split], eps[split:]


def diagnose(args):
    data = np.load(args.dataset, allow_pickle=True)
    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    valid_mask = data["valid_mask"].astype(np.float32)
    episode_starts = data["episode_starts"].astype(np.float32) if "episode_starts" in data else np.zeros_like(valid_mask)
    episode_starts[:, 0] = 1.0
    lengths = data["lengths"].astype(np.int64) if "lengths" in data else valid_mask.sum(axis=1).astype(np.int64)
    is_bomb = data["is_bomb_action"].astype(bool) if "is_bomb_action" in data else actions == PLACE_BOMB
    is_escape = data["is_post_bomb_escape"].astype(bool) if "is_post_bomb_escape" in data else np.zeros_like(is_bomb)
    weights = np.ones_like(valid_mask, dtype=np.float32)
    chunks = _make_chunks(
        observations,
        actions,
        valid_mask,
        episode_starts,
        weights,
        args.mode,
        args.seq_len,
        args.burn_in,
        args.overfit_subset,
    )
    chunks_obs, chunks_actions, chunks_mask, chunks_weights, chunks_starts, chunk_episodes = chunks
    train_eps, val_eps = _split_indices(len(actions), args.val_fraction, args.seed)
    train_set = set(int(v) for v in train_eps)
    train_chunks = np.asarray([idx for idx, ep in enumerate(chunk_episodes) if int(ep) in train_set], dtype=np.int64)
    val_chunks = np.asarray([idx for idx, ep in enumerate(chunk_episodes) if int(ep) not in train_set], dtype=np.int64)
    if len(val_chunks) == 0:
        val_chunks = train_chunks.copy()

    valid_actions = actions[valid_mask.astype(bool)]
    train_action_mask = np.isin(np.arange(len(actions)), train_eps)
    train_actions = actions[train_action_mask][valid_mask[train_action_mask].astype(bool)]
    val_actions = actions[~train_action_mask][valid_mask[~train_action_mask].astype(bool)]
    report = {
        "dataset": args.dataset,
        "episodes": int(len(actions)),
        "steps": int(valid_mask.sum()),
        "lengths": {
            "min": int(lengths.min()) if len(lengths) else 0,
            "p50": float(np.percentile(lengths, 50)) if len(lengths) else 0.0,
            "p90": float(np.percentile(lengths, 90)) if len(lengths) else 0.0,
            "max": int(lengths.max()) if len(lengths) else 0,
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
        },
        "train_episodes": int(len(train_eps)),
        "val_episodes": int(len(val_eps)),
        "action_distribution": _action_distribution(valid_actions),
        "train_action_distribution": _action_distribution(train_actions),
        "val_action_distribution": _action_distribution(val_actions),
        "place_bomb_fraction": float((valid_actions == PLACE_BOMB).mean()) if len(valid_actions) else 0.0,
        "post_bomb_escape_fraction": float(is_escape[valid_mask.astype(bool)].mean()) if valid_mask.any() else 0.0,
        "top_transitions": _transition_distribution(actions, valid_mask),
        "episode_start_count": int(episode_starts[valid_mask.astype(bool)].sum()),
        "chunks": int(len(chunks_actions)),
        "train_chunks": int(len(train_chunks)),
        "val_chunks": int(len(val_chunks)),
        "seq_len": int(args.seq_len),
        "burn_in": int(args.burn_in),
        "mode": args.mode,
    }
    if args.model and Path(args.model).exists():
        model = RecurrentPPO.load(
            args.model,
            device=args.device,
            custom_objects={
                "policy_kwargs": {
                    "features_extractor_class": BomberFeaturesExtractor,
                    "features_extractor_kwargs": {"features_dim": 256},
                    "normalize_images": False,
                },
            },
        )
        report["model"] = args.model
        report["train_metrics"] = _metrics(
            model.policy,
            chunks_obs[train_chunks],
            chunks_actions[train_chunks],
            chunks_mask[train_chunks],
            chunks_weights[train_chunks],
            episode_starts=chunks_starts[train_chunks],
            batch_size=args.batch_size,
            device=model.policy.device,
        )
        report["val_metrics"] = _metrics(
            model.policy,
            chunks_obs[val_chunks],
            chunks_actions[val_chunks],
            chunks_mask[val_chunks],
            chunks_weights[val_chunks],
            episode_starts=chunks_starts[val_chunks],
            batch_size=args.batch_size,
            device=model.policy.device,
        )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Diagnose recurrent BC dataset/model fit.")
    parser.add_argument("--dataset", default="ml/datasets/recurrent_bc_online_robust.npz")
    parser.add_argument("--model", default=None)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--burn_in", type=int, default=16)
    parser.add_argument("--mode", choices=["all", "movement", "bomb_light"], default="all")
    parser.add_argument("--overfit_subset", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=9900)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    diagnose(args)


if __name__ == "__main__":
    main()
