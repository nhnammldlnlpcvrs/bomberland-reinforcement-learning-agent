from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition.evaluation.runtime_guard import load_agent_instance
from engine.game import BomberEnv
from scripts.participant import benchmark_rl_strong_pool as pool


AGENTS = {
    "hybrid_model_optimized": "agent/hybrid_agent_model_optimized",
    "online_robust": "agent/hybrid_agent_online_robust",
    "tactical_rule": "TacticalRuleAgent",
    "genius_rule": "GeniusRuleAgent",
    "smarter_rule": "SmarterRuleAgent",
}


def _set_model_env(enabled: bool, checkpoint: str | None, max_latency_ms: float) -> None:
    os.environ["HYBRID_MODEL_ENABLE"] = "true" if enabled else "false"
    os.environ["HYBRID_MODEL_MAX_LATENCY_MS"] = str(float(max_latency_ms))
    os.environ["HYBRID_MODEL_ONLY_SAFE_ACTIONS"] = "true"
    if checkpoint:
        os.environ["HYBRID_MODEL_CHECKPOINT"] = str(checkpoint)
    else:
        os.environ.pop("HYBRID_MODEL_CHECKPOINT", None)


def _load_path_agent(path: str, agent_id: int):
    pool._clear_submission_import_state()
    agent = load_agent_instance(str(ROOT / path / "agent.py"), agent_id)
    pool._clear_submission_import_state()
    return agent


def disabled_parity_check(seeds: list[int], max_steps: int) -> dict:
    """Compare disabled candidate actions against online_robust on same states."""
    _set_model_env(False, None, 5.0)
    total_calls = 0
    mismatches = []

    for seed in seeds:
        online_agents = [
            _load_path_agent("agent/hybrid_agent_online_robust", idx)
            for idx in range(4)
        ]
        candidate_agents = [
            _load_path_agent("agent/hybrid_agent_model_optimized", idx)
            for idx in range(4)
        ]
        env = BomberEnv(max_steps=max_steps, seed=seed)
        obs = {**env.reset(seed=seed), "step": 0}
        done = False
        step = 0

        while not done and step < max_steps:
            actions = []
            for idx in range(4):
                if not bool(obs["players"][idx, 2]):
                    actions.append(0)
                    continue
                online_action = int(online_agents[idx].act(obs))
                candidate_action = int(candidate_agents[idx].act(obs))
                total_calls += 1
                if online_action != candidate_action:
                    mismatches.append({
                        "seed": int(seed),
                        "step": int(step),
                        "agent_id": int(idx),
                        "online_action": int(online_action),
                        "candidate_action": int(candidate_action),
                    })
                actions.append(online_action)
            obs, terminated, truncated = env.step(actions)
            step += 1
            obs = {**obs, "step": step}
            done = terminated or truncated

    return {
        "calls": total_calls,
        "mismatches": len(mismatches),
        "mismatch_rate": (len(mismatches) / total_calls) if total_calls else 0.0,
        "examples": mismatches[:20],
    }


def run_pool(label: str, enabled: bool, args) -> dict:
    _set_model_env(enabled, args.checkpoint, args.max_latency_ms)
    agent_names = list(AGENTS)
    episodes = []
    for episode_idx in range(args.episodes):
        base_seed = args.seeds[episode_idx % len(args.seeds)]
        seed = base_seed + episode_idx * 9973
        episodes.append(
            pool.run_episode(agent_names, AGENTS, seed, args.max_steps, args.timeout_s)
        )
        if (episode_idx + 1) % args.progress_every == 0:
            print(f"{label}: completed {episode_idx + 1}/{args.episodes} episodes")

    stats = pool._aggregate(episodes, agent_names)
    summary = pool._summary_dict(stats)
    return {
        "label": label,
        "enabled": enabled,
        "summary": summary,
        "episodes": episodes,
        "latency": _latency_summary(episodes, agent_names),
    }


def collect_candidate_diagnostics(args) -> dict:
    _set_model_env(True, args.checkpoint, args.max_latency_ms)
    merged_counters: dict[str, int] = {}
    reject_reasons: dict[str, int] = {}
    interventions = []
    seeds = args.parity_seeds[:3]

    for seed in seeds:
        agents = [
            _load_path_agent("agent/hybrid_agent_model_optimized", idx)
            for idx in range(4)
        ]
        env = BomberEnv(max_steps=min(args.max_steps, 160), seed=seed)
        obs = {**env.reset(seed=seed), "step": 0}
        done = False
        step = 0
        while not done and step < min(args.max_steps, 160):
            actions = []
            for idx, agent in enumerate(agents):
                if not bool(obs["players"][idx, 2]):
                    actions.append(0)
                    continue
                try:
                    action = int(agent.act(obs))
                except Exception:
                    action = 0
                actions.append(action if 0 <= action <= 5 else 0)
            obs, terminated, truncated = env.step(actions)
            step += 1
            obs = {**obs, "step": step}
            done = terminated or truncated

        for agent in agents:
            for key, value in getattr(agent, "counters", {}).items():
                merged_counters[key] = merged_counters.get(key, 0) + int(value)
            for key, value in getattr(agent, "reject_reason_counts", {}).items():
                reject_reasons[key] = reject_reasons.get(key, 0) + int(value)
            for item in getattr(agent, "intervention_log", []):
                if len(interventions) < 25:
                    interventions.append(item)

    return {
        "seeds": seeds,
        "counters": merged_counters,
        "reject_reason_counts": reject_reasons,
        "intervention_examples": interventions,
    }


def _latency_summary(episodes: list[dict], agent_names: list[str]) -> dict[str, dict]:
    rows = {name: [] for name in agent_names}
    for episode in episodes:
        for runtime in episode["runtime"]:
            rows[runtime["name"]].append(float(runtime["latency_ms"]))

    output = {}
    for name, values in rows.items():
        if not values:
            output[name] = {"mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
            continue
        ordered = sorted(values)
        p95_idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        output[name] = {
            "mean_ms": statistics.fmean(values),
            "p95_ms": ordered[p95_idx],
            "max_ms": max(values),
        }
    return output


def _promotion_verdict(enabled_summary: dict[str, dict], latency: dict[str, dict]) -> tuple[str, list[str]]:
    candidate = enabled_summary["hybrid_model_optimized"]
    production = enabled_summary["online_robust"]
    reasons = []

    if candidate["timeout_count"] > production["timeout_count"]:
        reasons.append("candidate has more timeouts than online_robust")
    if candidate["error_count"] > production["error_count"]:
        reasons.append("candidate has more runtime errors than online_robust")
    if candidate["invalid_action_count"] > production["invalid_action_count"]:
        reasons.append("candidate has more invalid actions than online_robust")
    if candidate["self_bomb_deaths"] > production["self_bomb_deaths"]:
        reasons.append("candidate increases self-bomb deaths")
    if latency["hybrid_model_optimized"]["p95_ms"] > 20.0:
        reasons.append("candidate p95 latency exceeds 20ms preferred gate")
    if latency["hybrid_model_optimized"]["max_ms"] > 100.0:
        reasons.append("candidate max latency exceeds 100ms action budget")
    if candidate["loss_rate"] >= production["loss_rate"]:
        reasons.append("candidate loss rate is not lower than online_robust")
    if candidate["average_rank"] > production["average_rank"] + 0.03:
        reasons.append("candidate average rank regresses by more than 0.03")

    clearly_better = (
        candidate["win_rate"] > production["win_rate"]
        and candidate["average_rank"] < production["average_rank"]
        and candidate["loss_rate"] < production["loss_rate"]
        and candidate["self_bomb_deaths"] <= production["self_bomb_deaths"]
        and candidate["timeout_count"] <= production["timeout_count"]
        and candidate["error_count"] <= production["error_count"]
    )

    if clearly_better and not reasons:
        return "PROMOTION_CANDIDATE", [
            "candidate clearly beats online_robust on win rate and average rank without safety/runtime regression",
        ]

    if not reasons:
        reasons.append("candidate does not clearly beat online_robust")
    return "REJECT_PROMOTION", reasons


def _summary_table(summary: dict[str, dict], latency: dict[str, dict]) -> list[str]:
    lines = [
        "| Agent | Matches | Win | Draw | Loss | Win Rate | Loss Rate | Avg Rank | Avg Survival | Self-Bomb Deaths | Timeouts | Errors | Invalid | Avg Act ms | P95 Act ms | Max Act ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in sorted(summary.items(), key=lambda item: item[1]["average_rank"]):
        lat = latency[name]
        lines.append(
            f"| `{name}` | {row['matches']} | {row['wins']} | {row['draws']} | {row['losses']} | "
            f"{row['win_rate']:.3f} | {row['loss_rate']:.3f} | {row['average_rank']:.3f} | "
            f"{row['average_survival_step']:.1f} | {row['self_bomb_deaths']} | "
            f"{row['timeout_count']} | {row['error_count']} | {row['invalid_action_count']} | "
            f"{lat['mean_ms']:.3f} | {lat['p95_ms']:.3f} | {lat['max_ms']:.3f} |"
        )
    return lines


def _markdown_report(args, parity: dict, disabled: dict, enabled: dict,
                     diagnostics: dict, verdict: str, reasons: list[str]) -> str:
    lines = [
        "# Hybrid Model Optimized Benchmark Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "Candidate: `agent/hybrid_agent_model_optimized/`",
        "",
        "Protected production files were not modified:",
        "",
        "- `submission/agent.py`",
        "- `agent/hybrid_agent_online_robust/`",
        "",
        "The model is disabled by default with `HYBRID_MODEL_ENABLE=false`.",
        "",
        "## Configuration",
        "",
        f"- Episodes per pool run: `{args.episodes}`",
        f"- Seeds: `{', '.join(str(seed) for seed in args.seeds)}`",
        f"- Max steps: `{args.max_steps}`",
        f"- Timeout threshold: `{args.timeout_s:.3f}s`",
        f"- Model checkpoint: `{args.checkpoint or 'default optional checkpoint'}`",
        f"- Model max latency gate: `{args.max_latency_ms:.3f}ms`",
        "",
        "## Disabled Parity Check",
        "",
        f"- Calls compared: `{parity['calls']}`",
        f"- Mismatches: `{parity['mismatches']}`",
        f"- Mismatch rate: `{parity['mismatch_rate']:.6f}`",
        "",
    ]
    if parity["examples"]:
        lines.extend(["Representative mismatches:", ""])
        for item in parity["examples"][:10]:
            lines.append(
                f"- seed `{item['seed']}`, step `{item['step']}`, agent `{item['agent_id']}`: "
                f"online `{item['online_action']}` vs candidate `{item['candidate_action']}`"
            )
        lines.append("")

    lines.extend([
        "## Disabled Pool Results",
        "",
        "This checks that the copied candidate behaves like production when the model flag is off.",
        "",
    ])
    lines.extend(_summary_table(disabled["summary"], disabled["latency"]))
    lines.extend([
        "",
        "## Enabled Pool Results",
        "",
    ])
    lines.extend(_summary_table(enabled["summary"], enabled["latency"]))
    lines.extend([
        "",
        "## Model Intervention Diagnostics",
        "",
        f"- Diagnostic seeds: `{', '.join(str(seed) for seed in diagnostics['seeds'])}`",
        "",
        "Counters:",
        "",
    ])
    for key, value in sorted(diagnostics["counters"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "Reject reason counts:", ""])
    for key, value in sorted(diagnostics["reject_reason_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")
    if diagnostics["intervention_examples"]:
        lines.extend(["", "Intervention examples:", ""])
        for item in diagnostics["intervention_examples"][:10]:
            lines.append(
                f"- step `{item['step']}`: rule `{item['rule_action']}` -> model "
                f"`{item['model_action']}`, candidates `{item['candidates']}`, "
                f"margin `{item['confidence_margin']:.3f}`"
            )
    lines.extend([
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
    ])
    for reason in reasons:
        lines.append(f"- {reason}")

    if verdict != "PROMOTION_CANDIDATE":
        lines.extend([
            "",
            "Do not promote `hybrid_agent_model_optimized` yet. Production remains `online_robust`.",
        ])
    else:
        lines.extend([
            "",
            "`hybrid_agent_model_optimized` is a promotion candidate only after a final independent 5x300 seed-block validation.",
        ])

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark hybrid_agent_model_optimized against online_robust.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 3026, 4026])
    parser.add_argument("--parity_seeds", nargs="+", type=int, default=[7100, 7200, 7300])
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--timeout_s", type=float, default=0.1)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--max_latency_ms", type=float, default=5.0)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--json_output", default="logs/hybrid_model_optimized_benchmark.json")
    parser.add_argument("--report", default="docs/HYBRID_MODEL_OPTIMIZED_REPORT.md")
    parser.add_argument("--skip_enabled", action="store_true")
    args = parser.parse_args()

    parity = disabled_parity_check(args.parity_seeds, args.max_steps)
    disabled = run_pool("disabled", False, args)
    if args.skip_enabled:
        enabled = disabled
        diagnostics = {"seeds": [], "counters": {}, "reject_reason_counts": {}, "intervention_examples": []}
        verdict, reasons = "REJECT_PROMOTION", ["enabled benchmark was skipped"]
    else:
        enabled = run_pool("enabled", True, args)
        diagnostics = collect_candidate_diagnostics(args)
        verdict, reasons = _promotion_verdict(enabled["summary"], enabled["latency"])

    payload = {
        "config": vars(args),
        "parity": parity,
        "disabled": disabled,
        "enabled": enabled,
        "diagnostics": diagnostics,
        "verdict": verdict,
        "reasons": reasons,
    }

    json_output = ROOT / args.json_output
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _markdown_report(args, parity, disabled, enabled, diagnostics, verdict, reasons),
        encoding="utf-8",
    )

    print(f"Disabled parity mismatches: {parity['mismatches']} / {parity['calls']}")
    print(f"Verdict: {verdict}")
    for reason in reasons:
        print(f"- {reason}")
    print(f"Wrote {json_output}")
    print(f"Wrote {report_path}")
    return 0 if verdict == "PROMOTION_CANDIDATE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
