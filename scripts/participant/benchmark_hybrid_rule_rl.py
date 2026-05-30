from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.evaluate_rl_pure import evaluate


@contextmanager
def rl_tiebreaker_env(enabled: bool, checkpoint: str | None):
    old_enable = os.environ.get("HYBRID_RULE_RL_ENABLE")
    old_checkpoint = os.environ.get("HYBRID_RULE_RL_CHECKPOINT")
    os.environ["HYBRID_RULE_RL_ENABLE"] = "true" if enabled else "false"
    if checkpoint:
        os.environ["HYBRID_RULE_RL_CHECKPOINT"] = checkpoint
    try:
        yield
    finally:
        if old_enable is None:
            os.environ.pop("HYBRID_RULE_RL_ENABLE", None)
        else:
            os.environ["HYBRID_RULE_RL_ENABLE"] = old_enable
        if old_checkpoint is None:
            os.environ.pop("HYBRID_RULE_RL_CHECKPOINT", None)
        else:
            os.environ["HYBRID_RULE_RL_CHECKPOINT"] = old_checkpoint


def run_candidate(label, path, opponents, episodes, max_steps, seed, enable_rl, checkpoint):
    print(f"\n=== {label} ===")
    print(f"agent_path: {path}")
    print(f"rl_enabled: {enable_rl}")
    with rl_tiebreaker_env(enable_rl, checkpoint):
        for opponent in opponents:
            result = evaluate(path, opponent, episodes, max_steps, seed)
            print(f"\nOpponent: {opponent}")
            keys = (
                "win_rate",
                "draw_rate",
                "death_rate",
                "place_bomb_frequency",
                "bomb_suicide_rate",
                "death_within_7_steps_after_bomb",
                "average_boxes_destroyed_per_bomb",
                "invalid_action_count",
                "crash_count",
                "timeout_count",
            )
            for key in keys:
                print(f"{key}: {result.get(key)}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark experimental hybrid_rule_rl without promoting production.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--opponents", nargs="+", default=["random", "simple", "online_robust"])
    parser.add_argument("--rl_checkpoint", default="ml/checkpoints/rl_agent_pure/ppo_validation_gated_distill_best.zip")
    parser.add_argument("--skip_enabled", action="store_true")
    args = parser.parse_args()

    run_candidate(
        "online_robust baseline",
        "agent/hybrid_agent_online_robust",
        args.opponents,
        args.episodes,
        args.max_steps,
        args.seed,
        enable_rl=False,
        checkpoint=None,
    )
    run_candidate(
        "hybrid_rule_rl RL disabled",
        "agent/hybrid_rule_rl",
        args.opponents,
        args.episodes,
        args.max_steps,
        args.seed,
        enable_rl=False,
        checkpoint=None,
    )
    if not args.skip_enabled:
        run_candidate(
            "hybrid_rule_rl RL enabled",
            "agent/hybrid_rule_rl",
            args.opponents,
            args.episodes,
            args.max_steps,
            args.seed,
            enable_rl=True,
            checkpoint=args.rl_checkpoint,
        )


if __name__ == "__main__":
    main()
