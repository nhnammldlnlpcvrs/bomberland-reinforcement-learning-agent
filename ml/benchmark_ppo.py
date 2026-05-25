"""Final benchmark: imitation vs PPO-finetuned vs online_robust.

Runs matches against baselines and head-to-head comparisons.
Tracks: win/draw/loss, STOP/BOMB rates, own-bomb deaths, runtime.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.game import BomberEnv

# --- Registry of known agents ---
BASELINE_FILES = {
    "RandomAgent": ("agent/random_agent.py", "RandomAgent"),
    "SimpleRuleAgent": ("agent/simple_rule_agent.py", "SimpleRuleAgent"),
    "SmarterRuleAgent": ("agent/smarter_rule_agent.py", "SmarterRuleAgent"),
    "TacticalRuleAgent": ("agent/tactical_rule_agent.py", "TacticalRuleAgent"),
    "GeniusRuleAgent": ("agent/genius_rule_agent.py", "GeniusRuleAgent"),
    "BoxFarmerAgent": ("agent/box_farmer_agent.py", "BoxFarmerAgent"),
}

DIR_AGENTS = {
    "online_robust": "agent/hybrid_agent_online_robust",
    "hybrid_ppo": "agent/hybrid_ppo",
}


def load_agent(path_str, agent_id):
    """Load an agent instance from a path or known name."""
    import_str = None
    class_name = "Agent"

    # Check if it's a known baseline
    if path_str in BASELINE_FILES:
        file_path, class_name = BASELINE_FILES[path_str]
        import_str = path_str  # unique module name
        abs_path = ROOT / file_path
    elif path_str in DIR_AGENTS:
        abs_path = ROOT / DIR_AGENTS[path_str] / "agent.py"
        import_str = path_str.replace("/", "_").replace("\\", "_")
    else:
        abs_path = Path(path_str)
        if not abs_path.is_absolute():
            abs_path = ROOT / abs_path
        if abs_path.is_dir():
            abs_path = abs_path / "agent.py"
        import_str = abs_path.stem

    spec = importlib.util.spec_from_file_location(import_str, str(abs_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    agent_cls = getattr(mod, class_name)
    return agent_cls(agent_id)


def run_match(agent_paths, seed, max_steps=500):
    """Run one match, return detailed stats."""
    agents = []
    for i, ap in enumerate(agent_paths):
        try:
            a = load_agent(ap, i)
            agents.append(a)
        except Exception as e:
            print(f"  FAILED agent {i} from {ap}: {e}")
            return None

    env = BomberEnv(max_steps=max_steps, seed=seed)
    obs = env.reset(seed=seed)

    stats = {
        "agent_paths": list(agent_paths),
        "seed": seed,
        "actions": [[] for _ in range(4)],
        "alive_steps": [0] * 4,
        "deaths": [False] * 4,
        "own_bomb_deaths": [False] * 4,
        "steps": 0,
        "winner": -1,
    }

    for step in range(max_steps):
        actions = []
        for i, agent in enumerate(agents):
            t0 = time.perf_counter()
            try:
                a = int(agent.act(obs))
            except Exception:
                a = 0
            elapsed = time.perf_counter() - t0
            if elapsed > 0.1:
                stats.setdefault("timeout_errors", []).append((i, step, elapsed))
            actions.append(a)
            stats["actions"][i].append(a)

        next_obs, terminated, truncated = env.step(actions)
        done = terminated or truncated

        for i in range(4):
            if int(obs["players"][i][2]):
                stats["alive_steps"][i] += 1

        if done:
            for i in range(4):
                if int(obs["players"][i][2]) and not int(next_obs["players"][i][2]):
                    stats["deaths"][i] = True
                    if _detect_own_bomb_death(obs, next_obs, i):
                        stats["own_bomb_deaths"][i] = True

        obs = next_obs
        if done:
            stats["steps"] = step + 1
            break
    else:
        stats["steps"] = max_steps

    alive = [i for i in range(4) if int(obs["players"][i][2])]
    if len(alive) == 1:
        stats["winner"] = alive[0]
    elif len(alive) == 0:
        stats["winner"] = -2
    else:
        stats["winner"] = -1

    return stats


def _detect_own_bomb_death(prev_obs, obs, agent_id):
    prev_p = prev_obs["players"][agent_id]
    if not int(prev_p[2]):
        return False
    my_r, my_c = int(prev_p[0]), int(prev_p[1])
    bombs_arr = np.asarray(obs["bombs"], dtype=np.int32)
    if bombs_arr.size == 0:
        return False
    if bombs_arr.ndim == 1:
        bombs_arr = bombs_arr.reshape(1, -1)
    for i in range(bombs_arr.shape[0]):
        owner = int(bombs_arr[i, 3])
        if owner == agent_id:
            br, bc = int(bombs_arr[i, 0]), int(bombs_arr[i, 1])
            if abs(my_r - br) + abs(my_c - bc) <= 2:
                return True
    return False


def run_benchmark(agent_path, opponents, num_matches, label, start_seed=1000):
    results = {
        "wins": 0, "draws": 0, "losses": 0,
        "stop_rate": 0.0, "bomb_rate": 0.0,
        "own_bomb_deaths": 0, "deaths": 0,
        "total_matches": 0,
        "total_actions": 0, "total_stop": 0, "total_bomb": 0,
        "variance_stop": [], "variance_bomb": [],
    }

    print(f"\n{'='*60}")
    print(f"BENCHMARK: {label}")
    print(f"  Agent: {agent_path}")
    print(f"  Opponents: {opponents}")
    print(f"  Matches: {num_matches}")
    print(f"{'='*60}")

    for m in range(num_matches):
        seed = start_seed + m
        paths = [agent_path] + list(opponents)
        stats = run_match(paths, seed=seed)

        if stats is None:
            continue

        results["total_matches"] += 1

        if stats["winner"] == 0:
            results["wins"] += 1
        elif stats["winner"] in (-1, -2):
            results["draws"] += 1
        else:
            results["losses"] += 1

        if stats["deaths"][0]:
            results["deaths"] += 1
        if stats["own_bomb_deaths"][0]:
            results["own_bomb_deaths"] += 1

        actions = stats["actions"][0]
        total = len(actions)
        stops = actions.count(0)
        bombs = actions.count(5)

        results["total_actions"] += total
        results["total_stop"] += stops
        results["total_bomb"] += bombs
        results["variance_stop"].append(stops / max(1, total))
        results["variance_bomb"].append(bombs / max(1, total))

        if (m + 1) % 20 == 0:
            n = m + 1
            wr = results["wins"] / n * 100
            print(f"  [{m+1}/{num_matches}] W{results['wins']}/"
                  f"D{results['draws']}/L{results['losses']}  WR={wr:.1f}%")

    n = max(1, results["total_matches"])
    results["stop_rate"] = results["total_stop"] / max(1, results["total_actions"])
    results["bomb_rate"] = results["total_bomb"] / max(1, results["total_actions"])
    results["win_rate"] = results["wins"] / n
    results["death_rate"] = results["deaths"] / n

    if results["variance_stop"]:
        results["stop_std"] = float(np.std(results["variance_stop"]))
        results["bomb_std"] = float(np.std(results["variance_bomb"]))

    return results


def print_results(results, label):
    print(f"\n{'---'*17}")
    print(f"RESULTS: {label}")
    print(f"{'---'*17}")
    n = results["total_matches"]
    print(f"  Matches:      {n}")
    print(f"  Win:          {results['wins']}/{n} ({results['win_rate']*100:.1f}%)")
    print(f"  Draw:         {results['draws']}/{n} ({results['draws']/max(1,n)*100:.1f}%)")
    print(f"  Loss:         {results['losses']}/{n} ({results['losses']/max(1,n)*100:.1f}%)")
    print(f"  Deaths:       {results['deaths']}/{n} ({results['death_rate']*100:.1f}%)")
    print(f"  Own-bomb:     {results['own_bomb_deaths']}")
    sr = results['stop_rate'] * 100
    br = results['bomb_rate'] * 100
    ss = results.get('stop_std', 0) * 100
    bs = results.get('bomb_std', 0) * 100
    print(f"  STOP rate:    {sr:.1f}%  (std={ss:.1f}%)")
    print(f"  BOMB rate:    {br:.1f}%  (std={bs:.1f}%)")
    print(f"  Total actions:{results['total_actions']}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PPO vs imitation vs online_robust")
    parser.add_argument("--num_matches", type=int, default=100,
                        help="Matches per agent-opponent combination")
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--ppo_checkpoint", type=str,
                        default="ml/checkpoints/hybrid_ppo/ppo_finetuned_v4.pt")
    args = parser.parse_args()

    print("=" * 60)
    print("FINAL PPO BENCHMARK")
    print("=" * 60)
    print(f"  Matches per combo: {args.num_matches}")
    print(f"  PPO checkpoint: {args.ppo_checkpoint}")

    all_results = {}
    baselines_set = ["TacticalRuleAgent", "SmarterRuleAgent", "GeniusRuleAgent"]

    # ---- 1. Imitation-only (agent/hybrid_ppo with default checkpoint) ----
    results = run_benchmark(
        "hybrid_ppo", baselines_set, args.num_matches,
        "Imitation (baseline) vs Baselines", start_seed=args.seed
    )
    print_results(results, "Imitation (baseline) vs Baselines")
    all_results["Imitation (baseline)"] = results

    # ---- 2. Online Robust (heuristic) ----
    results = run_benchmark(
        "online_robust", baselines_set, args.num_matches,
        "Online Robust vs Baselines", start_seed=args.seed + 2000
    )
    print_results(results, "Online Robust vs Baselines")
    all_results["Online Robust (heuristic)"] = results

    # ---- 3. PPO-finetuned: patch checkpoint path ----
    print(f"\n{'='*60}")
    print("PPO-FINETUNED BENCHMARK")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.ppo_checkpoint}")

    agent_file = ROOT / "agent/hybrid_ppo/agent.py"
    spec = importlib.util.spec_from_file_location("ppo_eval", str(agent_file))
    ppo_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ppo_mod)
    ppo_mod.CHECKPOINT_PATH = str(ROOT / args.ppo_checkpoint)
    print(f"  Patched to: {ppo_mod.CHECKPOINT_PATH}")

    ppo_results = {
        "wins": 0, "draws": 0, "losses": 0,
        "stop_rate": 0.0, "bomb_rate": 0.0,
        "own_bomb_deaths": 0, "deaths": 0,
        "total_matches": 0,
        "total_actions": 0, "total_stop": 0, "total_bomb": 0,
        "variance_stop": [], "variance_bomb": [],
    }

    for m in range(args.num_matches):
        seed = args.seed + 10000 + m
        agents = []
        # Agent 0: PPO
        agents.append(ppo_mod.Agent(0))
        # Agents 1-3: baselines
        for i, bname in enumerate(baselines_set):
            agents.append(load_agent(bname, i + 1))

        env = BomberEnv(max_steps=500, seed=seed)
        obs = env.reset(seed=seed)

        agent_actions = []
        agent_dead = False
        agent_own_bomb = False

        for step in range(500):
            actions = []
            for i, agent in enumerate(agents):
                try:
                    a = int(agent.act(obs))
                except Exception:
                    a = 0
                actions.append(a)

            if int(obs["players"][0][2]):
                agent_actions.append(actions[0])

            next_obs, terminated, truncated = env.step(actions)
            done = terminated or truncated

            if (not int(next_obs["players"][0][2]) and
                    int(obs["players"][0][2])):
                agent_dead = True
                if _detect_own_bomb_death(obs, next_obs, 0):
                    agent_own_bomb = True

            obs = next_obs
            if done:
                break

        ppo_results["total_matches"] += 1
        alive = [i for i in range(4) if int(obs["players"][i][2])]
        if len(alive) == 1 and alive[0] == 0:
            ppo_results["wins"] += 1
        elif len(alive) == 1:
            ppo_results["losses"] += 1
        else:
            ppo_results["draws"] += 1

        if agent_dead:
            ppo_results["deaths"] += 1
        if agent_own_bomb:
            ppo_results["own_bomb_deaths"] += 1

        total = len(agent_actions)
        stops = agent_actions.count(0)
        bombs = agent_actions.count(5)
        ppo_results["total_actions"] += total
        ppo_results["total_stop"] += stops
        ppo_results["total_bomb"] += bombs
        ppo_results["variance_stop"].append(stops / max(1, total))
        ppo_results["variance_bomb"].append(bombs / max(1, total))

        if (m + 1) % 20 == 0:
            n = m + 1
            wr = ppo_results["wins"] / n * 100
            print(f"  [{m+1}/{args.num_matches}] "
                  f"W{ppo_results['wins']}/D{ppo_results['draws']}/"
                  f"L{ppo_results['losses']}  WR={wr:.1f}%")

    n = max(1, ppo_results["total_matches"])
    ppo_results["stop_rate"] = (ppo_results["total_stop"] /
                                 max(1, ppo_results["total_actions"]))
    ppo_results["bomb_rate"] = (ppo_results["total_bomb"] /
                                 max(1, ppo_results["total_actions"]))
    ppo_results["win_rate"] = ppo_results["wins"] / n
    ppo_results["death_rate"] = ppo_results["deaths"] / n
    if ppo_results["variance_stop"]:
        ppo_results["stop_std"] = float(np.std(ppo_results["variance_stop"]))
        ppo_results["bomb_std"] = float(np.std(ppo_results["variance_bomb"]))

    print_results(ppo_results, "PPO-finetuned (v4) vs Baselines")
    all_results["PPO-finetuned (v4)"] = ppo_results

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("FINAL VERDICT SUMMARY")
    print(f"{'='*60}")
    header = f"{'Agent':<30} {'Win%':>8} {'STOP%':>8} {'BOMB%':>8} {'Death%':>8} {'Own-Bomb':>10}"
    print(header)
    print("-" * len(header))
    for label, r in all_results.items():
        print(f"{label:<30} {r['win_rate']*100:>7.1f}% "
              f"{r['stop_rate']*100:>7.1f}% "
              f"{r['bomb_rate']*100:>7.1f}% "
              f"{r['death_rate']*100:>7.1f}% "
              f"{r['own_bomb_deaths']:>10}")

    # Save
    out_path = ROOT / "ml/results/ppo_benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
