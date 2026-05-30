"""Collect behavior-cloning data for agent/hybrid_agent_rl.

Default teacher is the current production robust agent. The output .npz stores
encoded observations compatible with hybrid_agent_rl/model.py plus teacher
actions.
"""

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.game import BomberEnv
from scripts.participant.run_local_match import make_agents

sys.path.insert(0, str(ROOT / "agent" / "hybrid_agent_rl"))
from encoder import encode


def _load_agent_cls(path):
    p = Path(path)
    if p.is_dir():
        p = p / "agent.py"
    spec = importlib.util.spec_from_file_location(f"bc_teacher_{abs(hash(str(p)))}", str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load teacher: {p}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "Agent"):
        return module.Agent
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and name.endswith("Agent"):
            return obj
    raise AttributeError(f"No Agent class found in {p}")


def _teacher_roster(teacher_path, opponent_paths, seed):
    teacher_cls = _load_agent_cls(teacher_path)
    if opponent_paths:
        agents, names = make_agents(opponent_paths, seed=seed)
        teacher_slots = [0]
        agents[0] = teacher_cls(0)
        names[0] = Path(teacher_path).parent.name
        return agents, names, teacher_slots
    agents = [teacher_cls(i) for i in range(4)]
    return agents, [Path(teacher_path).parent.name] * 4, [0, 1, 2, 3]


def collect(args):
    rng = random.Random(args.seed)
    observations = []
    actions = []
    agent_ids = []
    steps = []
    action_counts = {i: 0 for i in range(6)}
    crashes = 0

    for ep in range(args.episodes):
        ep_seed = args.seed + ep if args.seed is not None else rng.randrange(2**31)
        env = BomberEnv(seed=ep_seed, max_steps=args.max_steps)
        agents, _names, teacher_slots = _teacher_roster(
            args.teacher, args.opponents, ep_seed
        )
        obs = env.reset(seed=ep_seed)
        done = False
        step = 0

        while not done and step < args.max_steps:
            obs_for_agent = dict(obs)
            obs_for_agent["step"] = step
            step_actions = []
            for i, agent in enumerate(agents):
                try:
                    action = int(agent.act(obs_for_agent))
                except Exception:
                    action = 0
                    crashes += 1
                if action < 0 or action > 5:
                    action = 0
                    crashes += 1
                step_actions.append(action)

            for slot in teacher_slots:
                if int(obs["players"][slot][2]) != 1:
                    continue
                observations.append(encode(obs_for_agent, slot))
                actions.append(step_actions[slot])
                agent_ids.append(slot)
                steps.append(step)
                action_counts[step_actions[slot]] += 1

            obs, terminated, truncated = env.step(step_actions)
            done = terminated or truncated
            step += 1

        if (ep + 1) % max(1, args.log_every) == 0:
            print(f"episode {ep + 1}/{args.episodes} samples={len(actions)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "teacher": args.teacher,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "action_counts": action_counts,
        "crashes": crashes,
    }
    np.savez_compressed(
        output,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        agent_ids=np.asarray(agent_ids, dtype=np.int8),
        steps=np.asarray(steps, dtype=np.int16),
        metadata_json=json.dumps(metadata),
    )
    print(f"saved {len(actions)} samples to {output}")
    print(f"action distribution: {action_counts}")
    if crashes:
        print(f"teacher/opponent act failures: {crashes}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="agent/hybrid_agent_online_robust")
    parser.add_argument("--opponents", nargs=4, default=None,
                        help="Optional 4-agent roster; slot 0 is replaced by teacher.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--output", default="ml/datasets/hybrid_agent_rl_bc.npz")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
