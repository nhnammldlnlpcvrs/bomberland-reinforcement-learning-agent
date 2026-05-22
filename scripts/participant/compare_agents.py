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
    return tuple(float(value) for value in match.groups())


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
            f"estimate_rankings failed for {agent_path} "
            f"with exit code {completed.returncode}\n{output}"
        )
    return parse_estimate_output(output)


def run_agent(agent_label, agent_path, num_matches, runs, max_steps):
    rows = []
    for run_id in range(1, runs + 1):
        print(f"\n=== {agent_label} Run {run_id}/{runs} ===")
        result = run_estimate(agent_path, num_matches, max_steps)
        result["agent"] = agent_label
        result["run_id"] = run_id
        rows.append(result)
        print(
            f"{agent_label} run {run_id}: "
            f"score={result['score']:.2f}, "
            f"win_rate={result['win_rate']:.1f}, "
            f"draw_rate={result['draw_rate']:.1f}, "
            f"avg_rank={result['avg_rank']:.2f}, "
            f"mu={result['mu']:.2f}, "
            f"sigma={result['sigma']:.2f}"
        )
    return rows


def _std(values):
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def summarize(rows):
    scores = [row["score"] for row in rows]
    draws = [row["draw_rate"] for row in rows]
    ranks = [row["avg_rank"] for row in rows]
    wins = [row["win_rate"] for row in rows]
    return {
        "mean_score": statistics.mean(scores),
        "std_score": _std(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "mean_draw_rate": statistics.mean(draws),
        "std_draw_rate": _std(draws),
        "mean_avg_rank": statistics.mean(ranks),
        "std_avg_rank": _std(ranks),
        "mean_win_rate": statistics.mean(wins),
    }


def _format_runs(rows):
    headers = [
        "agent",
        "run_id",
        "score",
        "win_rate",
        "draw_rate",
        "avg_rank",
        "mu",
        "sigma",
    ]
    table = [headers]
    for row in rows:
        table.append([
            row["agent"],
            str(row["run_id"]),
            f"{row['score']:.2f}",
            f"{row['win_rate']:.1f}",
            f"{row['draw_rate']:.1f}",
            f"{row['avg_rank']:.2f}",
            f"{row['mu']:.2f}",
            f"{row['sigma']:.2f}",
        ])

    widths = [max(len(line[i]) for line in table) for i in range(len(headers))]
    lines = []
    for idx, line in enumerate(table):
        lines.append("  ".join(value.rjust(widths[i]) for i, value in enumerate(line)))
        if idx == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    return "\n".join(lines)


def print_summary(label, summary):
    print(f"\n=== Summary {label} ===")
    print(f"mean_score={summary['mean_score']:.2f}")
    print(f"std_score={summary['std_score']:.2f}")
    print(f"min_score={summary['min_score']:.2f}")
    print(f"max_score={summary['max_score']:.2f}")
    print(f"mean_draw_rate={summary['mean_draw_rate']:.2f}")
    print(f"std_draw_rate={summary['std_draw_rate']:.2f}")
    print(f"mean_avg_rank={summary['mean_avg_rank']:.2f}")
    print(f"std_avg_rank={summary['std_avg_rank']:.2f}")
    print(f"mean_win_rate={summary['mean_win_rate']:.2f}")


def verdict(summary_a, summary_b):
    reject_reasons = []
    if summary_b["min_score"] < summary_a["min_score"] - 2.0:
        reject_reasons.append("min_score_b < min_score_a - 2.0")
    if summary_b["mean_avg_rank"] > summary_a["mean_avg_rank"] + 0.08:
        reject_reasons.append("mean_avg_rank_b > mean_avg_rank_a + 0.08")
    if summary_b["std_score"] > summary_a["std_score"] + 1.0:
        reject_reasons.append("std_score_b > std_score_a + 1.0")

    strong_accept = summary_b["mean_score"] >= summary_a["mean_score"] + 1.0
    anti_draw_accept = (
        summary_b["mean_score"] >= summary_a["mean_score"] - 0.5
        and summary_b["mean_draw_rate"] <= summary_a["mean_draw_rate"] - 5.0
        and summary_b["mean_avg_rank"] <= summary_a["mean_avg_rank"] + 0.03
    )

    if reject_reasons:
        return False, reject_reasons
    if strong_accept:
        return True, ["mean_score_b >= mean_score_a + 1.0"]
    if anti_draw_accept:
        return True, [
            "mean_score_b >= mean_score_a - 0.5",
            "mean_draw_rate_b <= mean_draw_rate_a - 5.0",
            "mean_avg_rank_b <= mean_avg_rank_a + 0.03",
        ]
    return False, ["no acceptance condition met"]


def main():
    parser = argparse.ArgumentParser(
        description="Compare two agents across repeated estimate_rankings runs."
    )
    parser.add_argument("--agent_a", required=True)
    parser.add_argument("--agent_b", required=True)
    parser.add_argument("--num_matches", type=int, default=300)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=500)
    args = parser.parse_args()

    try:
        rows_a = run_agent("A", args.agent_a, args.num_matches, args.runs, args.max_steps)
        rows_b = run_agent("B", args.agent_b, args.num_matches, args.runs, args.max_steps)
        summary_a = summarize(rows_a)
        summary_b = summarize(rows_b)

        print("\n=== Runs ===")
        print(_format_runs(rows_a + rows_b))
        print_summary("A", summary_a)
        print_summary("B", summary_b)

        accepted, reasons = verdict(summary_a, summary_b)
        print("\n=== Verdict ===")
        if accepted:
            print("ACCEPT")
        else:
            print("REJECT")
        for reason in reasons:
            print(f"- {reason}")
        return 0 if accepted else 1
    except Exception as exc:
        print("\n=== Error ===")
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
