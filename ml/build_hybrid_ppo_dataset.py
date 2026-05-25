"""Build balanced imitation dataset for hybrid PPO from heuristic agent replays.

Addresses the 0.1% bomb-rate problem with category-aware sampling:
  - bomb_chosen:       BOMB was actually chosen (keep ALL)
  - bomb_safe_unchosen:  BOMB safe in mask but not chosen (oversample)
  - danger:            agent in danger zone (keep ALL)
  - tactical:          <=3 safe actions, close decision (keep most)
  - routine:           normal movement (subsample)

Includes bomb_candidate flag so training can apply bomb-suppression loss
to teach the model when NOT to bomb even though it's safe.

Usage:
  python -m ml.build_hybrid_ppo_dataset \
    --agents agent/hybrid_agent_online_robust agent/hybrid_agent_phase_target_bombplan \
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

# Sampling weights per category (probability of keeping a transition)
CATEGORY_WEIGHTS = {
    "bomb_chosen": 1.0,         # keep all — rarest, most important
    "bomb_safe_unchosen": 0.4,  # oversample — learn bomb discrimination
    "danger": 1.0,              # keep all — safety-critical
    "tactical": 0.8,            # keep most — close decisions
    "routine": 0.15,            # subsample — abundant, least informative
}


def load_agent(agent_path):
    """Import an agent module by path and return its Agent class."""
    import importlib

    spec = importlib.util.spec_from_file_location(
        "collector_agent", str(Path(agent_path) / "agent.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Agent


def _classify_transition(mask, action, obs, agent_id):
    """Classify a transition into one of five categories.

    Returns category string and whether BOMB is a safe candidate.
    """
    bomb_candidate = bool(mask[5])
    bomb_chosen = action == 5
    num_safe = int(mask.sum())

    if bomb_chosen:
        category = "bomb_chosen"
    elif bomb_candidate:
        category = "bomb_safe_unchosen"
    else:
        # Check danger level at agent's position
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


def collect_transitions(agent_paths, num_episodes=60, seed=42,
                        target_samples=30000):
    """Run matches and collect balanced transitions from specified agents."""
    rng = np.random.default_rng(seed)

    agent_classes = []
    agent_names = []
    for ap in agent_paths:
        cls = load_agent(ap)
        agent_classes.append(cls)
        agent_names.append(Path(ap).name)

    # Collect into category buckets first, then sample
    raw_categories = {c: [] for c in CATEGORY_WEIGHTS}
    raw_stats = {c: 0 for c in CATEGORY_WEIGHTS}

    episode = 0
    total_raw = 0

    while total_raw < target_samples * 3 and episode < num_episodes:
        # Build 4-agent roster from pool (with replacement if pool < 4)
        perm = rng.permutation(len(agent_classes))
        selected_indices = [perm[i % len(perm)] for i in range(4)]
        selected_classes = [agent_classes[i] for i in selected_indices]
        selected_names = [agent_names[i] for i in selected_indices]

        env = BomberEnv(seed=int(rng.integers(0, 2 ** 31)))
        obs = env.reset()
        agents = [cls(agent_id=i) for i, cls in enumerate(selected_classes)]

        episode_transitions = []
        done = False
        step = 0

        while not done and step < 500:
            actions = []
            for i, agent in enumerate(agents):
                act = agent.act(obs)
                actions.append(act)

            for i in range(4):
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
                        "outcome": None,  # filled after episode
                    })

            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1

        # Determine winner
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

        total_raw = sum(len(v) for v in raw_categories.values())
        episode += 1

        if episode % 5 == 0:
            bomb_total = raw_stats["bomb_chosen"] + raw_stats["bomb_safe_unchosen"]
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

        # If we already have enough after this category, stop
        if len(all_actions) >= target_samples:
            break

    total = len(all_actions)
    if total == 0:
        raise RuntimeError("No transitions collected — check agent paths")

    observations = np.stack(all_obs, axis=0).astype(np.float32)
    masks = np.stack(all_masks, axis=0).astype(bool)
    actions = np.array(all_actions, dtype=np.int64)
    agent_ids = np.array(all_agent_ids, dtype=np.int64)
    episodes_arr = np.array(all_episodes, dtype=np.int64)
    steps_arr = np.array(all_steps, dtype=np.int64)
    outcomes_arr = np.array(all_outcomes)
    bomb_candidates_arr = np.array(all_bomb_candidates, dtype=bool)
    categories_arr = np.array(all_categories)

    metadata = {
        "source_agents": agent_names,
        "action_names": ACTION_NAMES,
        "schema": "hybrid_ppo_imitation_v2",
        "num_channels": observations.shape[1],
        "total_matches": episode,
        "category_weights_used": CATEGORY_WEIGHTS,
        "raw_counts": raw_stats,
        "kept_counts": kept_stats,
    }

    stats = {
        "bomb_chosen": int(np.sum(actions == 5)),
        "bomb_candidates": int(np.sum(bomb_candidates_arr)),
        "matches": episode,
    }

    return (observations, masks, actions, agent_ids, episodes_arr, steps_arr,
            outcomes_arr, bomb_candidates_arr, categories_arr, metadata, stats)


def _print_distribution(label, values, total):
    counts = np.bincount(values, minlength=6)
    print(f"\n{label}:")
    for i, name in enumerate(ACTION_NAMES):
        pct = 100 * counts[i] / max(1, total)
        print(f"  {i} {name}: {counts[i]:5d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Build balanced imitation dataset for hybrid PPO.")
    parser.add_argument("--agents", nargs="+", required=True,
                        help="Agent paths to collect data from")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_episodes", type=int, default=60)
    parser.add_argument("--target_samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Agents: {args.agents}")
    print(f"Episodes: {args.num_episodes}, Target: {args.target_samples}")
    print(f"Category weights: {CATEGORY_WEIGHTS}")
    print()

    (observations, masks, actions, agent_ids, episodes_arr, steps_arr,
     outcomes_arr, bomb_candidates_arr, categories_arr,
     metadata, stats) = collect_transitions(
        args.agents, args.num_episodes, args.seed, args.target_samples)

    total = len(actions)

    # ---- Raw distribution (pre-balancing) ----
    print(f"\n{'='*60}")
    print("RAW COLLECTION")
    print(f"{'='*60}")
    print(f"Total raw transitions: {sum(metadata['raw_counts'].values())}")
    print(f"Matches: {metadata['total_matches']}")
    for cat, count in metadata["raw_counts"].items():
        print(f"  {cat}: {count}")

    # ---- After balancing ----
    print(f"\n{'='*60}")
    print("AFTER BALANCED SAMPLING")
    print(f"{'='*60}")
    print(f"Kept: {total}")
    for cat, count in metadata["kept_counts"].items():
        print(f"  {cat}: {count}")

    # ---- Chosen action distribution ----
    _print_distribution("Chosen action distribution (post-balance)", actions, total)

    # ---- Bomb candidate analysis ----
    bomb_candidate_mask = bomb_candidates_arr
    n_bomb_candidate = bomb_candidate_mask.sum()
    print(f"\nBomb candidate analysis:")
    print(f"  States where BOMB is safe: {n_bomb_candidate} "
          f"({100*n_bomb_candidate/total:.1f}%)")
    n_bomb_chosen = int(np.sum(actions == 5))
    print(f"  States where BOMB was chosen: {n_bomb_chosen} "
          f"({100*n_bomb_chosen/total:.1f}%)")

    if n_bomb_candidate > 0:
        # Among bomb-candidate states, how often was bomb actually chosen?
        bomb_chosen_in_candidate = int(
            np.sum((bomb_candidate_mask) & (actions == 5))
        )
        print(f"  Bomb chosen when safe: {bomb_chosen_in_candidate} / "
              f"{n_bomb_candidate} "
              f"({100*bomb_chosen_in_candidate/max(1,n_bomb_candidate):.1f}%)")

    # ---- Safe action availability ----
    mask_counts = masks.sum(axis=0)
    print(f"\nSafe action availability:")
    for i, name in enumerate(ACTION_NAMES):
        pct = 100 * mask_counts[i] / max(1, total)
        print(f"  {i} {name}: {int(mask_counts[i]):5d} ({pct:5.1f}%)")

    # ---- Outcome distribution ----
    unique_outcomes, outcome_counts = np.unique(outcomes_arr, return_counts=True)
    print(f"\nOutcome distribution:")
    for outcome, count in zip(unique_outcomes, outcome_counts):
        print(f"  {outcome}: {count} ({100*count/total:.1f}%)")

    # ---- Category distribution in final set ----
    unique_cats, cat_counts = np.unique(categories_arr, return_counts=True)
    print(f"\nCategory distribution (final):")
    for cat, count in zip(unique_cats, cat_counts):
        print(f"  {cat}: {count} ({100*count/total:.1f}%)")

    # ---- Save ----
    metadata["total_count"] = int(total)
    metadata["bomb_chosen_count"] = int(n_bomb_chosen)
    metadata["bomb_candidate_count"] = int(n_bomb_candidate)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        observations=observations,
        safe_action_masks=masks,
        actions=actions,
        agent_ids=agent_ids,
        episodes=episodes_arr,
        steps=steps_arr,
        outcomes=outcomes_arr,
        bomb_candidates=bomb_candidates_arr,
        categories=categories_arr.astype(str),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    file_size_mb = out.stat().st_size / (1024 * 1024)
    obs_gb = observations.nbytes / (1024 ** 3)
    print(f"\nSaved: {out}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Observations raw: {obs_gb:.2f} GB uncompressed")
    print(f"  Compression ratio: {file_size_mb / (obs_gb * 1024):.2f}x")


if __name__ == "__main__":
    main()
