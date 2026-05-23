"""Replay-to-dataset skeleton for future imitation learning."""

import argparse
import json
from pathlib import Path


def iter_replay_files(log_dir):
    """Yield replay JSON files from a log directory."""
    root = Path(log_dir)
    if not root.exists():
        return
    yield from sorted(root.rglob("*.json"))


def load_replay(path):
    """Load one replay JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_obs_action_pairs(replay):
    """
    Extract observation/action pairs from one replay.

    TODO:
    - reconstruct obs from history frames
    - pick the target agent index
    - collect teacher actions
    - filter unsafe or low-quality decisions
    """
    raise NotImplementedError("dataset extraction is not implemented yet")


def build_dataset(log_dir, output):
    """
    Build a dataset from replay JSON files.

    TODO:
    - stream replay files
    - encode observations
    - write a compact dataset format
    """
    raise NotImplementedError("dataset building is not implemented yet")


def main():
    parser = argparse.ArgumentParser(description="Build future ML datasets from Bomberland replays.")
    parser.add_argument("--log_dir", default="logs/json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_dataset(args.log_dir, args.output)


if __name__ == "__main__":
    main()

