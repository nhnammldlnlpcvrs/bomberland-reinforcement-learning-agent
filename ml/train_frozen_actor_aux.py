from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.train_auxiliary_heads import train


def main():
    parser = argparse.ArgumentParser(description="Frozen-actor auxiliary experiment wrapper.")
    parser.add_argument("--dataset", default="ml/datasets/curriculum_rollouts.npz")
    parser.add_argument("--baseline_policy", default="ml/checkpoints/rl_agent_pure/ppo_bomb_selector_best.zip")
    parser.add_argument("--output", default="ml/checkpoints/rl_agent_pure/aux_curriculum_model.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--features_dim", type=int, default=256)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=6200)
    parser.add_argument("--death_weight", type=float, default=1.0)
    parser.add_argument("--escape_weight", type=float, default=1.0)
    parser.add_argument("--box_weight", type=float, default=0.5)
    parser.add_argument("--reachable_weight", type=float, default=0.25)
    parser.add_argument("--return_weight", type=float, default=0.25)
    args = parser.parse_args()
    print(
        "Training standalone auxiliary model only. "
        f"Baseline actor is referenced for provenance and is not modified: {args.baseline_policy}"
    )
    train(args)


if __name__ == "__main__":
    main()
