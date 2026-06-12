from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.evaluate_rl_pure import evaluate


KEYS = (
    "win_rate",
    "draw_rate",
    "death_rate",
    "average_reward",
    "average_survival_step",
    "place_bomb_frequency",
    "bomb_suicide_rate",
    "average_boxes_destroyed_per_bomb",
    "invalid_action_count",
    "crash_count",
    "timeout_count",
)


def _print_table(rows):
    headers = ("opponent", *KEYS)
    table = [headers]
    for row in rows:
        line = [str(row.get("opponent", ""))]
        for key in KEYS:
            value = row.get(key, 0)
            line.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        table.append(tuple(line))
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    for idx, row in enumerate(table):
        print("  ".join(row[i].rjust(widths[i]) for i in range(len(headers))))
        if idx == 0:
            print("  ".join("-" * width for width in widths))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark rl_strong without promoting production.")
    parser.add_argument("--candidate", default="agent/rl_strong")
    parser.add_argument("--opponents", nargs="+", default=["random", "simple", "tactical", "online_robust", "hybrid_agent_rl"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--output", default="logs/rl_strong_benchmark.json")
    args = parser.parse_args()

    rows = []
    for idx, opponent in enumerate(args.opponents):
        row = evaluate(
            args.candidate,
            opponent,
            args.episodes,
            args.max_steps,
            args.seed + idx * 1000,
            frame_stack=args.frame_stack,
        )
        row["opponent"] = opponent
        rows.append(row)

    _print_table(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
