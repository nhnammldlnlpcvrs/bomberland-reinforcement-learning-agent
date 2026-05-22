import argparse
import re
import statistics
import subprocess
import sys


WIN_RATE_RE = re.compile(r"Win Rate:\s*([0-9.]+)%")
DRAW_RATE_RE = re.compile(r"Draw Rate:\s*([0-9.]+)%")
AVG_RANK_RE = re.compile(r"Average Rank:\s*([0-9.]+)")
TRUESKILL_RE = re.compile(
    r"Estimated TrueSkill:\s*Score\s*=\s*([0-9.]+)\s*"
    r"\(mu=([0-9.]+),\s*sigma=([0-9.]+)\)"
)


def _parse_metric(pattern, output, name):
    match = pattern.search(output)
    if match is None:
        raise ValueError(f"Could not parse {name} from estimate_rankings output")
    if len(match.groups()) == 1:
        return float(match.group(1))
    return tuple(float(group) for group in match.groups())


def parse_estimate_output(output):
    score, mu, sigma = _parse_metric(TRUESKILL_RE, output, "TrueSkill")
    return {
        "score": score,
        "win_rate": _parse_metric(WIN_RATE_RE, output, "win rate"),
        "draw_rate": _parse_metric(DRAW_RATE_RE, output, "draw rate"),
        "avg_rank": _parse_metric(AVG_RANK_RE, output, "average rank"),
        "mu": mu,
        "sigma": sigma,
    }


def run_estimate(agent_path, num_matches, max_steps):
    cmd = [
        sys.executable,
        "-m",
        "scripts.participant.estimate_rankings",
        "--agent_path",
        agent_path,
        "--num_matches",
        str(num_matches),
        "--max_steps",
        str(max_steps),
    ]
    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"estimate_rankings failed with exit code {completed.returncode}\n"
            f"{output}"
        )
    return parse_estimate_output(output)


def _format_table(rows):
    headers = ["run_id", "score", "win_rate", "draw_rate", "avg_rank", "mu", "sigma"]
    table = [headers]
    for row in rows:
        table.append([
            str(row["run_id"]),
            f"{row['score']:.2f}",
            f"{row['win_rate']:.1f}",
            f"{row['draw_rate']:.1f}",
            f"{row['avg_rank']:.2f}",
            f"{row['mu']:.2f}",
            f"{row['sigma']:.2f}",
        ])

    widths = [max(len(line[i]) for line in table) for i in range(len(headers))]
    formatted = []
    for idx, line in enumerate(table):
        formatted.append(
            "  ".join(value.rjust(widths[i]) for i, value in enumerate(line))
        )
        if idx == 0:
            formatted.append(
                "  ".join("-" * widths[i] for i in range(len(headers)))
            )
    return "\n".join(formatted)


def _summary(values):
    if len(values) == 1:
        std = 0.0
    else:
        std = statistics.stdev(values)
    return {
        "mean": statistics.mean(values),
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def print_summary(rows):
    print("\n=== Stability Summary ===")
    for metric in ("score", "draw_rate", "avg_rank"):
        stats = _summary([row[metric] for row in rows])
        print(
            f"{metric}: "
            f"mean={stats['mean']:.2f}, "
            f"std={stats['std']:.2f}, "
            f"min={stats['min']:.2f}, "
            f"max={stats['max']:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run estimate_rankings multiple times and summarize variance."
    )
    parser.add_argument("--agent_path", required=True, help="Path to agent folder or file.")
    parser.add_argument("--num_matches", type=int, default=300)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=500)
    args = parser.parse_args()

    rows = []
    for run_id in range(1, args.runs + 1):
        print(f"\n=== Stability Run {run_id}/{args.runs} ===")
        result = run_estimate(args.agent_path, args.num_matches, args.max_steps)
        result["run_id"] = run_id
        rows.append(result)
        print(
            f"Run {run_id}: "
            f"score={result['score']:.2f}, "
            f"win_rate={result['win_rate']:.1f}, "
            f"draw_rate={result['draw_rate']:.1f}, "
            f"avg_rank={result['avg_rank']:.2f}, "
            f"mu={result['mu']:.2f}, "
            f"sigma={result['sigma']:.2f}"
        )

    print("\n=== Stability Runs ===")
    print(_format_table(rows))
    print_summary(rows)


if __name__ == "__main__":
    main()
