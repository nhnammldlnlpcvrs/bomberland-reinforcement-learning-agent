"""Learned policy evaluation placeholder."""

import argparse


def evaluate_policy(model, agent_path, num_matches):
    """
    Evaluate a future learned policy with a safety filter.

    TODO:
    - load model
    - wrap policy with action mask and heuristic fallback
    - run local matches and replay analysis
    """
    raise NotImplementedError("policy evaluation is not implemented yet")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a future learned policy.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--agent_path", default="agent/hybrid_agent")
    parser.add_argument("--num_matches", type=int, default=100)
    args = parser.parse_args()
    evaluate_policy(args.model, args.agent_path, args.num_matches)


if __name__ == "__main__":
    main()

