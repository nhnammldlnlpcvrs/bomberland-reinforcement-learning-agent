from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.evaluate_rl_pure import evaluate as evaluate_one


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a strong Bomberland research checkpoint.")
    parser.add_argument("--agent_path", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--opponents", nargs="+", default=["random", "simple"])
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--output", default="logs/bomberland_strong_eval.json")
    args = parser.parse_args()
    results = []
    for idx, opponent in enumerate(args.opponents):
        try:
            row = evaluate_one(
                args.agent_path,
                opponent,
                args.episodes,
                args.max_steps,
                args.seed + idx * 1000,
                frame_stack=args.frame_stack,
            )
        except Exception as exc:
            row = {
                "opponent": opponent,
                "episodes": args.episodes,
                "win_rate": 0.0,
                "draw_rate": 0.0,
                "death_rate": 1.0,
                "average_reward": 0.0,
                "average_survival_step": 0.0,
                "place_bomb_frequency": 0.0,
                "bomb_suicide_rate": 1.0,
                "death_within_7_steps_after_bomb": 0,
                "average_boxes_destroyed_per_bomb": 0.0,
                "useful_bomb_count": 0,
                "useless_bomb_count": 0,
                "invalid_action_count": 0,
                "crash_count": args.episodes,
                "timeout_count": 0,
                "action_counts": {str(i): 0 for i in range(6)},
                "error": repr(exc),
            }
        results.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
