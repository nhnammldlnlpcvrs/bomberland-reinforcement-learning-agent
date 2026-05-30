from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.evaluate_rl_pure import evaluate


def main():
    parser = argparse.ArgumentParser(description="Benchmark pure RL candidate without promoting production.")
    parser.add_argument("--candidate", default="agent/rl_agent_pure")
    parser.add_argument("--opponents", nargs="+", default=["random", "simple", "online_robust", "hybrid_agent_rl"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print("=== Pure RL Benchmark ===")
    print(f"Candidate: {args.candidate}")
    for opponent in args.opponents:
        result = evaluate(args.candidate, opponent, args.episodes, args.max_steps, args.seed)
        print(f"\nOpponent: {opponent}")
        for key, value in result.items():
            if key != "opponent":
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
