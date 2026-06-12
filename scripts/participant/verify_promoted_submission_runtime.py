from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition.evaluation.runtime_guard import load_agent_instance
from engine.game import BomberEnv


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify promoted submission runtime.")
    parser.add_argument("--agent", default="submission/agent.py")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=60608)
    parser.add_argument("--output", default="logs/promoted_submission_runtime_check.json")
    args = parser.parse_args()

    agent_path = ROOT / args.agent
    agents = [load_agent_instance(str(agent_path), idx) for idx in range(4)]
    env = BomberEnv(max_steps=args.max_steps, seed=args.seed)
    latencies = []
    errors = 0

    for episode in range(args.episodes):
        obs = {**env.reset(seed=args.seed + episode), "step": 0}
        done = False
        step = 0
        while not done and step < args.max_steps:
            actions = []
            for idx, agent in enumerate(agents):
                if not bool(obs["players"][idx, 2]):
                    actions.append(0)
                    continue
                started = time.perf_counter()
                try:
                    action = int(agent.act(obs))
                except Exception:
                    action = 0
                    errors += 1
                latencies.append((time.perf_counter() - started) * 1000.0)
                actions.append(action if 0 <= action <= 5 else 0)
            obs, terminated, truncated = env.step(actions)
            step += 1
            obs = {**obs, "step": step}
            done = terminated or truncated

    ordered = sorted(latencies)
    p95_idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))) if ordered else 0
    counters = {}
    reject_reasons = {}
    for agent in agents:
        for key, value in getattr(agent, "counters", {}).items():
            counters[key] = counters.get(key, 0) + int(value)
        for key, value in getattr(agent, "reject_reason_counts", {}).items():
            reject_reasons[key] = reject_reasons.get(key, 0) + int(value)

    payload = {
        "agent": str(agent_path),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "act_calls": len(latencies),
        "errors": errors,
        "average_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "p95_latency_ms": ordered[p95_idx] if ordered else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "model_counters": counters,
        "reject_reason_counts": reject_reasons,
    }

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    ok = (
        errors == 0
        and payload["max_latency_ms"] < 100.0
        and int(counters.get("model_loaded", 0)) > 0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
