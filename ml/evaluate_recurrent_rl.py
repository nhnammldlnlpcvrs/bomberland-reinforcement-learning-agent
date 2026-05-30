from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.evaluate_rl_pure import evaluate

RECURRENT_AGENT_FILES = (
    "agent.py",
    "constants.py",
    "utils.py",
    "model.py",
    "policy.py",
    "encoder.py",
    "action_mask.py",
    "metadata.json",
)


def prepare_recurrent_agent(agent_path: str) -> str:
    path = Path(agent_path)
    if path.suffix.lower() != ".zip":
        return agent_path
    if not path.exists():
        raise FileNotFoundError(f"Recurrent checkpoint not found: {agent_path}")
    source_dir = ROOT / "agent" / "rl_agent_recurrent"
    eval_dir = ROOT / "ml" / "checkpoints" / "rl_agent_recurrent" / "eval_agents" / path.stem
    eval_dir.mkdir(parents=True, exist_ok=True)
    for filename in RECURRENT_AGENT_FILES:
        shutil.copy2(source_dir / filename, eval_dir / filename)
    shutil.copy2(path, eval_dir / "policy.zip")
    return str(eval_dir)


def main():
    parser = argparse.ArgumentParser(description="Evaluate research-only recurrent Bomberland agent.")
    parser.add_argument("--agent_path", required=True)
    parser.add_argument("--opponents", nargs="+", default=["random", "simple"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=9700)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    packaged = prepare_recurrent_agent(args.agent_path)
    results = [
        evaluate(packaged, opponent, args.episodes, args.max_steps, args.seed + idx * 1000, frame_stack=1)
        for idx, opponent in enumerate(args.opponents)
    ]
    text = json.dumps(results, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
