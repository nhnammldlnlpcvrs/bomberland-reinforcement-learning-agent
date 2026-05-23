import os
import random
import argparse
import sys
import json
from datetime import datetime
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
# Add parent directory to sys.path if not already present
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine.game import BomberEnv
from agent import RandomAgent, SimpleRuleAgent, SmarterRuleAgent, GeniusRuleAgent, BoxFarmerAgent, TacticalRuleAgent
from competition.evaluation.runtime_guard import load_agent_instance


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def make_agents(agent_paths, seed=None):
    n_players = len(agent_paths)
    agents = [None] * n_players
    names = [None] * n_players

    if seed is not None:
        random.seed(seed)

    for i, path in enumerate(agent_paths):
        if path == "None" or path.lower() == "random":
            # Random rule-based baseline
            x = random.randint(0, 5)
            if x == 0:
                names[i] = "RandomAgent"
                agents[i] = RandomAgent(i)
            elif x == 1:
                names[i] = "SimpleRuleAgent"
                agents[i] = SimpleRuleAgent(i)
            elif x == 2:
                names[i] = "SmarterRuleAgent"
                agents[i] = SmarterRuleAgent(i)
            elif x == 3:
                names[i] = "GeniusRuleAgent"
                agents[i] = GeniusRuleAgent(i)
            elif x == 4:
                names[i] = "BoxFarmerAgent"
                agents[i] = BoxFarmerAgent(i)
            else:
                names[i] = "TacticalRuleAgent"
                agents[i] = TacticalRuleAgent(i)
        elif path == "RandomAgent":
            names[i] = "RandomAgent"
            agents[i] = RandomAgent(i)
        elif path == "SimpleRuleAgent":
            names[i] = "SimpleRuleAgent"
            agents[i] = SimpleRuleAgent(i)
        elif path == "SmarterRuleAgent":
            names[i] = "SmarterRuleAgent"
            agents[i] = SmarterRuleAgent(i)
        elif path == "GeniusRuleAgent":
            names[i] = "GeniusRuleAgent"
            agents[i] = GeniusRuleAgent(i)
        elif path == "BoxFarmerAgent":
            names[i] = "BoxFarmerAgent"
            agents[i] = BoxFarmerAgent(i)
        elif path == "TacticalRuleAgent":
            names[i] = "TacticalRuleAgent"
            agents[i] = TacticalRuleAgent(i)
        else:
            # Custom agent path
            p = Path(path)
            if p.is_dir():
                p = p / "agent.py"
            if not p.exists():
                raise FileNotFoundError(f"Agent file not found: {p}")
            
            try:
                agents[i] = load_agent_instance(str(p), i)
                # If agent has a team_id class attribute, use it, otherwise use folder name
                if hasattr(agents[i], "team_id"):
                    names[i] = agents[i].team_id
                else:
                    names[i] = p.parent.name if p.parent.name else p.name
            except Exception as e:
                raise RuntimeError(f"Failed to load agent from {p}: {e}")

    return agents, names


def _obs_snapshot(obs, step, actions=None):
    return {
        "step": int(step),
        "actions": None if actions is None else [int(action) for action in actions],
        "alive": [bool(player[2]) for player in obs["players"]],
        "map": obs["map"].tolist(),
        "board": obs["map"].tolist(),
        "grid": obs["map"].tolist(),
        "players": obs["players"].tolist(),
        "bombs": obs["bombs"].tolist(),
        "flames": [],
        "explosions": [],
    }


def _final_ranks(n_players, death_groups, survivors):
    ranks = [0] * n_players
    ordered_groups = list(death_groups)
    if survivors:
        ordered_groups.append(list(survivors))
    for rank, group in enumerate(reversed(ordered_groups)):
        for player_id in group:
            ranks[player_id] = rank
    return ranks


def _save_replay(log_dir, seed, episode, names, ranks, survival_steps, total_steps, history):
    json_dir = Path(log_dir) / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    seed_part = "none" if seed is None else str(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = json_dir / f"match_{timestamp}_{seed_part}.json"
    payload = {
        "seed": seed,
        "episode": episode,
        "team_ids": names,
        "agents": names,
        "meta": {"agent_names": names},
        "ranks": ranks,
        "survival_steps": survival_steps,
        "total_steps": int(total_steps),
        "history": history,
        "frames": history,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def run_match(agent_paths, num_episodes=10, max_steps=500, seed=None, save_logs=True, log_dir="logs"):
    env = BomberEnv(max_steps=max_steps, seed=seed)
    n_players = len(agent_paths)
    
    agents, names = make_agents(agent_paths, seed)
    info = [{"name": names[i], "wins": 0} for i in range(n_players)]

    for episode in range(num_episodes):
        episode_seed = None if seed is None else seed + episode
        obs = env.reset(seed=episode_seed)
        done = False
        step = 0
        death_order = []
        death_groups = []
        survival_steps = [0] * n_players
        prev_alive = [bool(p[2]) for p in obs["players"]]
        history = [_obs_snapshot(obs, step, actions=None)] if save_logs else []

        while not done and step < max_steps:
            actions = []
            for i in range(n_players):
                try:
                    action = agents[i].act(obs)
                except Exception as e:
                    print(f"Agent {names[i]} failed to act: {e}")
                    action = 0
                actions.append(action)
                
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1
            if save_logs:
                history.append(_obs_snapshot(obs, step, actions=actions))

            alive_now = [bool(p[2]) for p in obs["players"]]
            deaths_this_step = []
            for i in range(n_players):
                if prev_alive[i] and not alive_now[i]:
                    death_order.append(info[i]["name"])
                    deaths_this_step.append(i)
                    survival_steps[i] = step
            if deaths_this_step:
                death_groups.append(deaths_this_step)
            prev_alive = alive_now
        
        alive_final = [bool(p[2]) for p in obs["players"]]
        survivors = [i for i in range(n_players) if alive_final[i]]
        for player_id in survivors:
            survival_steps[player_id] = step
        ranks = _final_ranks(n_players, death_groups, survivors)
        
        if len(survivors) == 1:
            winner = survivors[0]
            info[winner]["wins"] += 1
            print(f"Episode {episode + 1}: {info[winner]['name']} wins | Died: {death_order}")
        else:
            print(f"Episode {episode + 1}: Draw | Died: {death_order}")

        if save_logs:
            _save_replay(
                log_dir=log_dir,
                seed=episode_seed,
                episode=episode,
                names=names,
                ranks=ranks,
                survival_steps=survival_steps,
                total_steps=step,
                history=history,
            )

    print("\n=== Summary ===")
    for i in range(n_players):
        print(f"{info[i]['name']}: {info[i]['wins']} wins")
    return info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_paths", nargs="+", default=["None", "None", "None", "None"],
                        help="Paths to agent.py files, agent folders, or baseline names (e.g. RandomAgent). Use 'None' for a random baseline.")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--visualize", type=str2bool, default=False)
    parser.add_argument("--autoplay", type=str2bool, default=True)
    parser.add_argument("--save_logs", type=str2bool, default=True)
    parser.add_argument("--log_dir", default="logs")
    args = parser.parse_args()
    
    if args.visualize:
        from scripts.participant.visualizer import run_simple_viewer

        run_simple_viewer(
            agent_paths=args.agent_paths,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            autoplay=args.autoplay,
        )
    else:
        run_match(
            agent_paths=args.agent_paths,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            save_logs=args.save_logs,
            log_dir=args.log_dir,
        )
