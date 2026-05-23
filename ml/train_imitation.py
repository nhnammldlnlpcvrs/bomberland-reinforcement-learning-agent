"""Imitation learning training placeholder."""

import argparse


def train_imitation(dataset, output):
    """
    Train a future policy from heuristic labels.

    TODO:
    - load dataset
    - create a tiny policy model
    - train supervised action logits
    - save checkpoint and metadata
    """
    raise NotImplementedError("imitation training is not implemented yet")


def main():
    parser = argparse.ArgumentParser(description="Train a future imitation policy.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train_imitation(args.dataset, args.output)


if __name__ == "__main__":
    main()

