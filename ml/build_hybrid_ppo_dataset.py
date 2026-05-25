"""Build balanced imitation dataset for hybrid PPO from heuristic agent replays.

Addresses the 0.1% bomb-rate problem with category-aware sampling:
  - bomb_chosen:       BOMB was actually chosen (keep ALL)
  - bomb_safe_unchosen:  BOMB safe in mask but not chosen (oversample)
  - danger:            agent in danger zone (keep ALL)
  - tactical:          <=3 safe actions, close decision (keep most)
  - routine:           normal movement (subsample)

Data collected ONLY from teacher agents. Opponents provide match diversity.

Usage:
  python -m ml.build_hybrid_ppo_dataset \
    --teachers agent/hybrid_agent_online_robust agent/hybrid_agent_phase_target_bombplan \
    --opponents agent/tactical_rule_agent agent/smarter_rule_agent agent/genius_rule_agent \
    --output ml/datasets/hybrid_ppo/imitation_dataset.npz \
    --num_episodes 60 --target_samples 30000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.game import BomberEnv
from agent.hybrid_ppo.state_encoder import encode_state
from agent.hybrid_ppo.safety_filter import (
    compute_safe_action_mask,
    compute_danger_map,
)

ACTION_NAMES = ["STOP", "LEFT", "RIGHT", "UP", "DOWN", "BOMB"]

CATEGORY_WEIGHTS = {
    "bomb_chosen": 1.0,
    "bomb_safe_unchosen": 0.4,
    "danger": 1.0,
    "tactical": 0.8,
    "routine": 0.15,
}


def load_agent(agent_path):
    """Load Agent class from a directory (agent.py) or standalone .py file.

    Tries 'Agent' first, then any exported class with an 'act' method.
    """
    import importlib
    p = Path(agent_path)
    if p.is_dir():
        target = p / "agent.py"
    else:
        target = p if p.suffix == ".py" else p.with_suffix(".py")
    if not target.exists():
        raise FileNotFoundError(f"Agent not found: {target}")
    spec = importlib.util.spec_from_file_location("collector_agent", str(target))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Try 'Agent' first, then look for any class with an 'act' method
    if hasattr(mod, "Agent"):
        return mod.Agent
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if (isinstance(obj, type) and
                attr_name.endswith("Agent") and
                hasattr(obj, "act")):
            return obj
    raise AttributeError(
        f"No Agent class found in {target}. "
        f"Available: {[a for a in dir(mod) if isinstance(getattr(mod, a), type)]}"
    )


def _classify_transition(mask, action, obs, agent_id):
    bomb_candidate = bool(mask[5])
    bomb_chosen = action == 5
    num_safe = int(mask.sum())

    if bomb_chosen:
        category = "bomb_chosen"
    elif bomb_candidate:
        category = "bomb_safe_unchosen"
    else:
        game_map = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]
        danger = compute_danger_map(game_map, players, bombs)
        my_r = int(players[agent_id][0])
        my_c = int(players[agent_id][1])
        curr_danger = danger[my_r, my_c]

        if curr_danger <= 3:
            category = "danger"
        elif num_safe <= 3:
            category = "tactical"
        else:
            category = "routine"

    return category, bomb_candidate


def collect_transitions(teacher_paths, opponent_paths=None,
                        num_episodes=60, seed=42,
                        target_samples=30000):
    rng = np.random.default_rng(seed)

    teacher_classes = []
    teacher_names = []
    for ap in teacher_paths:
        cls = load_agent(ap)
        teacher_classes.append(cls)
        teacher_names.append(Path(ap).name)

    opponent_classes = []
    opponent_names = []
    if opponent_paths:
        for ap in opponent_paths:
            cls = load_agent(ap)
            opponent_classes.append(cls)
            opponent_names.append(Path(ap).name)

    raw_categories = {c: [] for c in CATEGORY_WEIGHTS}
    raw_stats = {c: 0 for c in CATEGORY_WEIGHTS}
    source_counts = {name: 0 for name in teacher_names}

    episode = 0
    total_raw = 0

    while total_raw < target_samples * 3 and episode < num_episodes:
        # Build roster: 1-2 teachers + 2-3 opponents
        n_teachers = int(rng.integers(1, min(3, len(teacher_classes) + 1)))
        teacher_indices = rng.choice(len(teacher_classes), size=n_teachers,
                                     replace=False)
        teacher_slots = set(rng.choice(4, size=n_teachers, replace=False))

        roster_classes = [None] * 4
        roster_names = [None] * 4
        roster_is_teacher = [False] * 4

        t_idx = 0
        for slot in range(4):
            if slot in teacher_slots and t_idx < n_teachers:
                ci = teacher_indices[t_idx]
                roster_classes[slot] = teacher_classes[ci]
                roster_names[slot] = teacher_names[ci]
                roster_is_teacher[slot] = True
                t_idx += 1
            elif opponent_classes:
                oi = int(rng.integers(0, len(opponent_classes)))
                roster_classes[slot] = opponent_classes[oi]
                roster_names[slot] = opponent_names[oi]
            else:
                ci = int(rng.integers(0, len(teacher_classes)))
                roster_classes[slot] = teacher_classes[ci]
                roster_names[slot] = teacher_names[ci]

        env = BomberEnv(seed=int(rng.integers(0, 2 ** 31)))
        obs = env.reset()
        agents = [cls(agent_id=i) for i, cls in enumerate(roster_classes)]

        episode_transitions = []
        done = False
        step = 0

        while not done and step < 500:
            actions = [agent.act(obs) for agent in agents]

            for i in range(4):
                if not roster_is_teacher[i]:
                    continue
                p = obs["players"][i]
                if not int(p[2]):
                    continue

                mask = compute_safe_action_mask(obs, i)
                action = actions[i]

                if mask.any() and mask[action]:
                    encoded = encode_state(obs, i)
                    category, bomb_candidate = _classify_transition(
                        mask, action, obs, i
                    )
                    episode_transitions.append({
                        "obs": encoded,
                        "mask": mask,
                        "action": action,
                        "agent_id": i,
                        "step": step,
                        "category": category,
                        "bomb_candidate": bomb_candidate,
                        "source_agent": roster_names[i],
                        "outcome": None,
                    })

            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1

        alive_mask = [int(obs["players"][i][2]) for i in range(4)]
        winner = None
        if sum(alive_mask) == 1:
            winner = alive_mask.index(1)

        for t in episode_transitions:
            if winner == t["agent_id"]:
                t["outcome"] = "win"
            elif winner is not None:
                t["outcome"] = "loss"
            else:
                t["outcome"] = "draw"

            raw_categories[t["category"]].append(t)
            raw_stats[t["category"]] += 1
            source_counts[t["source_agent"]] += 1

        total_raw = sum(len(v) for v in raw_categories.values())
        episode += 1

        if episode % 5 == 0:
            print(f"  match {episode:3d}: raw={total_raw:5d} "
                  f"bomb_chosen={raw_stats['bomb_chosen']} "
                  f"bomb_safe={raw_stats['bomb_safe_unchosen']} "
                  f"danger={raw_stats['danger']} "
                  f"tactical={raw_stats['tactical']} "
                  f"routine={raw_stats['routine']}")

    # ---- Balanced sampling ----
    all_obs = []
    all_masks = []
    all_actions = []
    all_agent_ids = []
    all_episodes = []
    all_steps = []
    all_outcomes = []
    all_bomb_candidates = []
    all_categories = []
    all_sources = []

    kept_stats = {}
    for category, weight in CATEGORY_WEIGHTS.items():
        pool = raw_categories[category]
        n_keep = max(1, int(len(pool) * weight))
        indices = rng.choice(len(pool), size=n_keep, replace=False)
        kept_stats[category] = n_keep

        for idx in indices:
            t = pool[idx]
            all_obs.append(t["obs"])
            all_masks.append(t["mask"])
            all_actions.append(t["action"])
            all_agent_ids.append(t["agent_id"])
            all_episodes.append(episode)
            all_steps.append(t["step"])
            all_outcomes.append(t["outcome"])
            all_bomb_candidates.append(t["bomb_candidate"])
            all_categories.append(category)
            all_sources.append(t["source_agent"])

        if len(all_actions) >= target_samples:
            break

    total = len(all_actions)
    if total == 0:
        raise RuntimeError("No transitions collected")

    observations = np.stack(all_obs, axis=0).astype(np.float32)
    masks = np.stack(all_masks, axis=0).astype(bool)
    actions = np.array(all_actions, dtype=np.int64)
    agent_ids = np.array(all_agent_ids, dtype=np.int64)
    episodes_arr = np.array(all_episodes, dtype=np.int64)
    steps_arr = np.array(all_steps, dtype=np.int64)
    outcomes_arr = np.array(all_outcomes)
    bomb_candidates_arr = np.array(all_bomb_candidates, dtype=bool)
    categories_arr = np.array(all_categories)
    sources_arr = np.array(all_sources)

    metadata = {
        "source_agents": teacher_names,
        "opponent_agents": opponent_names,
        "action_names": ACTION_NAMES,
        "schema": "hybrid_ppo_imitation_v2",
        "num_channels": observations.shape[1],
        "total_matches": episode,
        "category_weights_used": CATEGORY_WEIGHTS,
        "raw_counts": raw_stats,
        "kept_counts": kept_stats,
        "source_breakdown_raw": source_counts,
    }

    stats = {
        "bomb_chosen": int(np.sum(actions == 5)),
        "bomb_candidates": int(np.sum(bomb_candidates_arr)),
        "matches": episode,
    }

    return (observations, masks, actions, agent_ids, episodes_arr, steps_arr,
            outcomes_arr, bomb_candidates_arr, categories_arr, sources_arr,
            metadata, stats)


def _print_distribution(label, values, total):
    counts = np.bincount(values, minlength=6)
    print(f"\n{label}:")
    for i, name in enumerate(ACTION_NAMES):
        pct = 100 * counts[i] / max(1, total)
        print(f"  {i} {name}: {counts[i]:5d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Build balanced imitation dataset for hybrid PPO.")
    parser.add_argument("--teachers", nargs="+", required=True,
                        help="Teacher agent paths (data collected from these)")
    parser.add_argument("--opponents", nargs="+", default=None,
                        help="Opponent agent paths (match diversity, no data)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_episodes", type=int, default=60)
    parser.add_argument("--target_samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.opponents:
        print(f"Teachers: {args.teachers}")
        print(f"Opponents: {args.opponents}")
    else:
        print(f"Teachers: {args.teachers} (no opponents — teacher-only matches)")
    print(f"Episodes: {args.num_episodes}, Target: {args.target_samples}")
    print(f"Category weights: {CATEGORY_WEIGHTS}")
    print()

    (observations, masks, actions, agent_ids, episodes_arr, steps_arr,
     outcomes_arr, bomb_candidates_arr, categories_arr, sources_arr,
     metadata, stats) = collect_transitions(
        args.teachers, args.opponents, args.num_episodes, args.seed,
        args.target_samples)

    total = len(actions)

    # 1. Raw distribution
    print(f"\n{'='*60}")
    print("RAW COLLECTION")
    print(f"{'='*60}")
    raw_total = sum(metadata["raw_counts"].values())
    print(f"Total raw transitions: {raw_total}")
    print(f"Matches: {metadata['total_matches']}")
    for cat in CATEGORY_WEIGHTS:
        count = metadata["raw_counts"].get(cat, 0)
        print(f"  {cat}: {count} ({100*count/max(1,raw_total):.1f}%)")

    # 2. Kept samples
    print(f"\n{'='*60}")
    print("AFTER BALANCED SAMPLING")
    print(f"{'='*60}")
    print(f"Total kept: {total}")
    for cat in CATEGORY_WEIGHTS:
        count = metadata["kept_counts"].get(cat, 0)
        print(f"  {cat}: {count} ({100*count/max(1,total):.1f}%)")

    # 3. File size
    out_path = Path(args.output)
    print(f"\nFile: {out_path}")

    # 4. Action distribution
    _print_distribution("Action distribution (post-balance)", actions, total)

    # 5. Category distribution
    unique_cats, cat_counts = np.unique(categories_arr, return_counts=True)
    print(f"\nCategory distribution (final):")
    for cat, count in zip(unique_cats, cat_counts):
        print(f"  {cat}: {count} ({100*count/total:.1f}%)")

    # 6-7. Bomb stats
    n_bomb_chosen = int(np.sum(actions == 5))
    n_bomb_candidate = int(bomb_candidates_arr.sum())
    print(f"\nBomb analysis:")
    print(f"  BOMB safe: {n_bomb_candidate} ({100*n_bomb_candidate/total:.1f}%)")
    print(f"  BOMB chosen: {n_bomb_chosen} ({100*n_bomb_chosen/total:.1f}%)")
    if n_bomb_candidate > 0:
        bomb_chosen_in_candidate = int(
            np.sum(bomb_candidates_arr & (actions == 5))
        )
        print(f"  Expert chooses BOMB when safe: {bomb_chosen_in_candidate}/"
              f"{n_bomb_candidate} ({100*bomb_chosen_in_candidate/n_bomb_candidate:.1f}%)")

    # 8. Source-agent breakdown
    print(f"\nSource-agent breakdown (raw):")
    for name, count in metadata["source_breakdown_raw"].items():
        print(f"  {name}: {count} ({100*count/max(1,raw_total):.1f}%)")

    # 9. Opponent-pool breakdown (which opponents were available)
    opps = metadata.get("opponent_agents", [])
    print(f"\nOpponent pool: {opps if opps else '(none — teachers only)'}")

    # 10. Outcome distribution
    unique_outcomes, outcome_counts = np.unique(outcomes_arr, return_counts=True)
    print(f"\nOutcome distribution:")
    for outcome, count in zip(unique_outcomes, outcome_counts):
        print(f"  {outcome}: {count} ({100*count/total:.1f}%)")

    # 11. Safe action availability
    mask_counts = masks.sum(axis=0)
    print(f"\nSafe action availability:")
    for i, name in enumerate(ACTION_NAMES):
        pct = 100 * mask_counts[i] / max(1, total)
        print(f"  {i} {name}: {int(mask_counts[i]):5d} ({pct:5.1f}%)")

    # Save
    metadata["total_count"] = int(total)
    metadata["bomb_chosen_count"] = int(n_bomb_chosen)
    metadata["bomb_candidate_count"] = int(n_bomb_candidate)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        observations=observations,
        safe_action_masks=masks,
        actions=actions,
        agent_ids=agent_ids,
        episodes=episodes_arr,
        steps=steps_arr,
        outcomes=outcomes_arr,
        bomb_candidates=bomb_candidates_arr,
        categories=categories_arr.astype(str),
        source_agents=sources_arr,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    obs_gb = observations.nbytes / (1024 ** 3)
    print(f"\nSaved: {out_path}")
    print(f"  Compressed: {file_size_mb:.1f} MB")
    print(f"  Raw observations: {obs_gb:.2f} GB")
    print(f"  Ratio: {obs_gb * 1024 / max(0.01, file_size_mb):.0f}x compression")


if __name__ == "__main__":
    main()
