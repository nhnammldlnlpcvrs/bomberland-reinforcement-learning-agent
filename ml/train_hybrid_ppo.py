"""Safety-constrained PPO fine-tuning for hybrid Bomberland agent.

Refines the imitation-pretrained CNN policy using PPO with shaped rewards
while maintaining all safety invariants from the safety filter.

PPO only chooses among safe actions — safety_filter remains authoritative.
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.game import BomberEnv

from agent.hybrid_ppo.ppo_policy import PPOPolicy, NUM_CHANNELS, NUM_ACTIONS
from agent.hybrid_ppo.state_encoder import encode_state
from agent.hybrid_ppo.safety_filter import compute_safe_action_mask
from agent.hybrid_ppo.reward import compute_reward

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_NAMES = ["STOP", "LEFT", "RIGHT", "UP", "DOWN", "BOMB"]


def _load_opponent_cls(path):
    import importlib.util
    p = Path(path)
    if p.is_dir():
        target = p / "agent.py"
    else:
        target = p if p.suffix == ".py" else p.with_suffix(".py")
    spec = importlib.util.spec_from_file_location("ppo_opponent", str(target))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "Agent"):
        return mod.Agent
    for an in dir(mod):
        obj = getattr(mod, an)
        if isinstance(obj, type) and an.endswith("Agent") and hasattr(obj, "act"):
            return obj
    raise AttributeError(f"No Agent class in {target}")


class RolloutBuffer:
    """Stores transitions with pre-computed advantages/returns."""

    def __init__(self):
        self.obs = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.advantages = []
        self.returns = []

    def add(self, obs, mask, action, log_prob, advantage, ret):
        self.obs.append(obs)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.advantages.append(advantage)
        self.returns.append(ret)

    def size(self):
        return len(self.actions)

    def extend(self, other):
        self.obs.extend(other.obs)
        self.masks.extend(other.masks)
        self.actions.extend(other.actions)
        self.log_probs.extend(other.log_probs)
        self.advantages.extend(other.advantages)
        self.returns.extend(other.returns)

    def get_all(self):
        obs = np.stack(self.obs, axis=0).astype(np.float32)
        masks = np.stack(self.masks, axis=0).astype(bool)
        actions = np.array(self.actions, dtype=np.int64)
        log_probs = np.array(self.log_probs, dtype=np.float32)
        advantages = np.array(self.advantages, dtype=np.float32)
        returns = np.array(self.returns, dtype=np.float32)
        return obs, masks, actions, log_probs, advantages, returns


def _compute_gae(rewards, values, dones, gamma, lam):
    """Compute GAE advantages and returns for a single trajectory."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    rets = np.zeros(n, dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(n)):
        mask_val = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask_val - values[t]
        gae = delta + gamma * lam * mask_val * gae
        advantages[t] = gae
        rets[t] = gae + values[t]
        next_value = values[t]
    return advantages, rets


def _build_roster(ppo_model, n_ppo, opponent_cls_list, device, seed):
    """Build a 4-agent roster with n_ppo PPO agents + rule-based opponents."""
    rng = np.random.default_rng(seed)
    ppo_slots = set(rng.choice(4, size=n_ppo, replace=False))
    roster = []
    ppo_ids = []
    for slot in range(4):
        if slot in ppo_slots:
            roster.append(_PPOAgentWrapper(ppo_model, slot, device))
            ppo_ids.append(slot)
        else:
            cls = opponent_cls_list[int(rng.integers(0, len(opponent_cls_list)))]
            roster.append(cls(agent_id=slot))
    return roster, ppo_ids


class _PPOAgentWrapper:
    """Wraps the PPO model to expose the standard Agent interface."""

    def __init__(self, model, agent_id, device):
        self.model = model
        self.agent_id = agent_id
        self.device = device
        self._pos_history = deque(maxlen=20)
        self._prev_obs = None
        self._prev_position = None

    def act(self, obs):
        p = obs["players"][self.agent_id]
        alive = bool(int(p[2]))
        if not alive:
            return 0

        mask = compute_safe_action_mask(obs, self.agent_id)

        state = encode_state(obs, self.agent_id)
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).bool()

        with torch.no_grad():
            action, log_prob, value, _entropy = self.model.act(state_t, mask_t)

        return action

    def reset_episode(self):
        self._pos_history.clear()
        self._prev_obs = None
        self._prev_position = None


def collect_rollouts(ppo_model, opponent_cls_list, device, num_episodes,
                     gamma, lam, seed):
    """Run episodes and collect transitions from PPO agents.

    GAE is computed per-agent per-episode to maintain correct temporal
    ordering. Transitions are stored with pre-computed advantages/returns.
    """
    rng = np.random.default_rng(seed)
    buffer = RolloutBuffer()
    episode_returns = []
    action_counts = {}
    total_deaths = 0
    own_bomb_deaths = 0
    total_bomb_actions = 0
    total_actions = 0

    for ep in range(num_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        n_ppo = int(rng.integers(1, 3))
        roster, ppo_ids = _build_roster(ppo_model, n_ppo, opponent_cls_list,
                                        device, ep_seed)

        env = BomberEnv(seed=ep_seed)
        obs = env.reset()
        done = False
        step = 0

        # Per-agent trajectory buffers: aid -> list of (state, mask, action,
        # log_prob, value, prev_obs, pos_history)
        agent_trajs = {aid: [] for aid in ppo_ids}
        agent_pos_hist = {aid: deque(maxlen=20) for aid in ppo_ids}
        agent_prev_obs = {aid: None for aid in ppo_ids}

        while not done and step < 500:
            # Collect actions from all agents
            agent_step_data = {}  # aid -> (state, mask, action, log_prob, value)
            actions = []

            for agent in roster:
                if isinstance(agent, _PPOAgentWrapper):
                    aid = agent.agent_id
                    p_agent = obs["players"][aid]
                    alive = bool(int(p_agent[2]))

                    if not alive:
                        actions.append(0)
                        continue

                    mask = compute_safe_action_mask(obs, aid)
                    state = encode_state(obs, aid)
                    state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    mask_t = torch.from_numpy(mask).bool()

                    with torch.no_grad():
                        action, log_prob, value, _entropy = ppo_model.act(
                            state_t, mask_t
                        )
                    actions.append(action)
                    agent_step_data[aid] = (state, mask, action, log_prob, value)
                else:
                    actions.append(agent.act(obs))

            prev_obs = obs
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1

            # Compute rewards and record transitions per PPO agent
            for aid in ppo_ids:
                if aid not in agent_step_data:
                    continue

                state, mask, action, log_prob, value = agent_step_data[aid]
                p_curr = obs["players"][aid]
                alive_now = bool(int(p_curr[2]))
                my_r, my_c = int(p_curr[0]), int(p_curr[1])

                # Update position history
                agent_pos_hist[aid].append((my_r, my_c))
                pos_hist_list = list(agent_pos_hist[aid])

                # Reward: use prev_obs from this agent, or prev_obs of match
                prev_for_reward = agent_prev_obs.get(aid) or prev_obs

                reward = compute_reward(
                    obs, prev_for_reward, aid,
                    action, done and not alive_now, step,
                    position_history=pos_hist_list,
                )

                # Record trajectory step
                agent_trajs[aid].append({
                    "state": state,
                    "mask": mask,
                    "action": action,
                    "log_prob": log_prob,
                    "value": value,
                    "reward": reward,
                    "done": done or not alive_now,
                })

                # Stats
                action_counts[action] = action_counts.get(action, 0) + 1
                total_actions += 1
                if action == 5:
                    total_bomb_actions += 1

                if not alive_now:
                    total_deaths += 1
                    bombs = np.asarray(obs["bombs"], dtype=np.int32)
                    if bombs.size > 0:
                        if bombs.ndim == 1:
                            bombs = bombs.reshape(1, -1)
                        for bi in range(bombs.shape[0]):
                            if int(bombs[bi, 3]) == aid:
                                br, bc = int(bombs[bi, 0]), int(bombs[bi, 1])
                                if abs(my_r - br) + abs(my_c - bc) <= 2:
                                    own_bomb_deaths += 1
                                    break

                # Update prev_obs for this agent
                agent_prev_obs[aid] = prev_obs

        # ---- After episode: compute GAE per agent ----
        ep_return = 0.0
        for aid in ppo_ids:
            traj = agent_trajs[aid]
            if not traj:
                continue

            rewards = [t["reward"] for t in traj]
            values = [t["value"] for t in traj]
            dones = [t["done"] for t in traj]

            advantages, rets = _compute_gae(rewards, values, dones, gamma, lam)

            for i, t in enumerate(traj):
                buffer.add(
                    t["state"], t["mask"], t["action"],
                    t["log_prob"], float(advantages[i]), float(rets[i]),
                )

            ep_return += sum(rewards)

        episode_returns.append(ep_return)

    return (buffer, episode_returns, action_counts,
            total_deaths, own_bomb_deaths, total_bomb_actions, total_actions)


def ppo_update(model, frozen_model, optimizer, buffer, clip_eps, value_coef,
               entropy_coef, kl_coef, epochs, batch_size, device,
               max_grad_norm=0.5):
    """Perform PPO clipped update with KL penalty to frozen imitation prior."""
    if buffer.size() == 0:
        return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "kl": 0,
                "total_loss": 0}

    obs, masks, actions, old_log_probs, advantages, returns = buffer.get_all()

    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    obs_t = torch.from_numpy(obs).to(device)
    masks_t = torch.from_numpy(masks).to(device)
    actions_t = torch.from_numpy(actions).long().to(device)
    old_log_probs_t = torch.from_numpy(old_log_probs).float().to(device)
    advantages_t = torch.from_numpy(advantages).float().to(device)
    returns_t = torch.from_numpy(returns).float().to(device)

    n = buffer.size()
    indices = np.arange(n)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    n_batches = 0

    for epoch in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            b_obs = obs_t[batch_idx]
            b_masks = masks_t[batch_idx]
            b_actions = actions_t[batch_idx]
            b_old_lp = old_log_probs_t[batch_idx]
            b_adv = advantages_t[batch_idx]
            b_ret = returns_t[batch_idx]

            # Current policy forward pass
            logits, values_raw = model(b_obs)
            logits = logits.clone()
            logits[~b_masks] = -1e9
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(b_actions)
            entropy = dist.entropy()

            # Frozen imitation reference for KL penalty
            with torch.no_grad():
                ref_logits, _ = frozen_model(b_obs)
                ref_logits = ref_logits.clone()
                ref_logits[~b_masks] = -1e9
                ref_probs = F.softmax(ref_logits, dim=-1)

            # KL(current || reference)
            kl = (probs * (torch.log(probs + 1e-9) -
                           torch.log(ref_probs + 1e-9))).sum(-1).mean()

            # PPO clipped objective
            ratio = torch.exp(log_probs - b_old_lp)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values_raw.squeeze(-1), b_ret)
            entropy_loss = entropy.mean()

            loss = (policy_loss + value_coef * value_loss
                    - entropy_coef * entropy_loss + kl_coef * kl)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_loss.item()
            total_kl += kl.item()
            n_batches += 1

    denom = max(1, n_batches)
    return {
        "policy_loss": total_policy_loss / denom,
        "value_loss": total_value_loss / denom,
        "entropy": total_entropy / denom,
        "kl": total_kl / denom,
        "total_loss": (total_policy_loss + total_value_loss
                       - total_entropy + total_kl) / denom,
    }


@torch.no_grad()
def evaluate_policy(model, opponent_cls_list, device, num_episodes=20, seed=999):
    """Quick evaluation against baseline opponents."""
    rng = np.random.default_rng(seed)
    wins = 0
    draws = 0
    losses = 0
    deaths = 0
    action_counts = {a: 0 for a in range(6)}
    times = []

    for ep in range(num_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        n_ppo = 1
        roster, ppo_ids = _build_roster(model, n_ppo, opponent_cls_list,
                                        device, ep_seed)

        env = BomberEnv(seed=ep_seed)
        obs = env.reset()
        done = False
        step = 0

        while not done and step < 500:
            actions = []
            for agent in roster:
                if isinstance(agent, _PPOAgentWrapper):
                    aid = agent.agent_id
                    p = obs["players"][aid]
                    if not int(p[2]):
                        actions.append(0)
                        continue
                    mask = compute_safe_action_mask(obs, aid)
                    state = encode_state(obs, aid)
                    st = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    mt = torch.from_numpy(mask).bool()
                    t0 = time.perf_counter()
                    action, _ = model.get_action_logits(st, mt)
                    times.append(time.perf_counter() - t0)
                    actions.append(action)
                    action_counts[action] = action_counts.get(action, 0) + 1
                else:
                    actions.append(agent.act(obs))

            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1

        ppo_aid = ppo_ids[0] if ppo_ids else 0
        p_final = obs["players"][ppo_aid]
        alive = int(p_final[2])
        alive_mask = [int(obs["players"][i][2]) for i in range(4)]
        n_alive = sum(alive_mask)

        if not alive:
            deaths += 1
            losses += 1
        elif n_alive == 1:
            winner = alive_mask.index(1)
            if winner == ppo_aid:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1

    results = {
        "wins": wins, "draws": draws, "losses": losses,
        "deaths": deaths, "episodes": num_episodes,
    }
    if times:
        arr = np.array(times) * 1000
        results["latency_mean_ms"] = float(arr.mean())
        results["latency_p95_ms"] = float(np.percentile(arr, 95))

    total_acts = sum(action_counts.values())
    results["stop_pct"] = 100 * action_counts.get(0, 0) / max(1, total_acts)
    results["bomb_pct"] = 100 * action_counts.get(5, 0) / max(1, total_acts)
    results["actions"] = {ACTION_NAMES[k]: v for k, v in action_counts.items()}

    return results


def train(args):
    device = torch.device("cpu")
    print(f"Device: {device}")

    # ---- Load imitation checkpoint ----
    print(f"\nLoading imitation checkpoint: {args.checkpoint}")
    model = PPOPolicy(input_channels=NUM_CHANNELS, num_actions=NUM_ACTIONS)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.train()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')}")

    # ---- Frozen imitation model for KL penalty ----
    frozen_model = PPOPolicy(input_channels=NUM_CHANNELS, num_actions=NUM_ACTIONS)
    frozen_model.load_state_dict(ckpt["model_state_dict"])
    frozen_model.to(device)
    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad = False

    # ---- Load opponents ----
    print(f"\nLoading opponents...")
    opponent_cls_list = [_load_opponent_cls(p) for p in args.opponents]
    opponent_names = [Path(p).name for p in args.opponents]
    print(f"  Opponents: {opponent_names}")

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # ---- Training loop ----
    best_eval_wins = -1
    round_results = []

    print(f"\n{'='*60}")
    print(f"PPO FINE-TUNING")
    print(f"{'='*60}")
    print(f"Rounds: {args.rounds}")
    print(f"Episodes/round: {args.episodes_per_round}")
    print(f"PPO epochs: {args.ppo_epochs}")
    print(f"LR: {args.lr}, Clip: {args.clip_eps}, KL coef: {args.kl_coef}")
    print(f"Value coef: {args.value_coef}, Entropy coef: {args.entropy_coef}")
    print(f"Gamma: {args.gamma}, Lambda: {args.gae_lambda}")

    # ---- Baseline eval ----
    print(f"\n--- Baseline (imitation only) ---")
    baseline_eval = evaluate_policy(model, opponent_cls_list, device,
                                    num_episodes=args.eval_episodes, seed=100)
    _print_eval(baseline_eval)
    round_results.append({"round": 0, "eval": baseline_eval})

    # ---- PPO rounds ----
    for round_idx in range(1, args.rounds + 1):
        t0 = time.perf_counter()

        # Collect rollouts
        (buffer, ep_returns, act_counts,
         deaths, ob_deaths, bomb_acts, total_acts) = collect_rollouts(
            model, opponent_cls_list, device,
            num_episodes=args.episodes_per_round,
            gamma=args.gamma, lam=args.gae_lambda,
            seed=1000 + round_idx,
        )

        # PPO update
        metrics = ppo_update(
            model, frozen_model, optimizer, buffer,
            clip_eps=args.clip_eps,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            kl_coef=args.kl_coef,
            epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            device=device,
            max_grad_norm=args.max_grad_norm,
        )

        scheduler.step()
        elapsed = time.perf_counter() - t0

        stop_pct = 100 * act_counts.get(0, 0) / max(1, total_acts)
        bomb_pct = 100 * bomb_acts / max(1, total_acts)

        # Print round summary
        mean_ret = np.mean(ep_returns) if ep_returns else 0.0
        print(f"\nRound {round_idx:3d}/{args.rounds} | "
              f"trans={buffer.size():5d} | "
              f"ret={mean_ret:7.2f} | "
              f"stop={stop_pct:.1f}% bomb={bomb_pct:.1f}% | "
              f"deaths={deaths} ob={ob_deaths} | "
              f"p_loss={metrics['policy_loss']:.4f} "
              f"v_loss={metrics['value_loss']:.4f} "
              f"H={metrics['entropy']:.3f} "
              f"KL={metrics['kl']:.4f} | "
              f"{elapsed:.1f}s")

        # Periodic eval
        if round_idx % args.eval_every == 0 or round_idx == args.rounds:
            eval_results = evaluate_policy(
                model, opponent_cls_list, device,
                num_episodes=args.eval_episodes,
                seed=1000 + round_idx,
            )
            round_results.append({"round": round_idx, "eval": eval_results})
            _print_eval(eval_results)

            if eval_results["wins"] > best_eval_wins:
                best_eval_wins = eval_results["wins"]
                _save_checkpoint(model, optimizer, round_idx, eval_results,
                                 args.output)

    # ---- Final save ----
    _save_checkpoint(model, optimizer, args.rounds, round_results[-1]["eval"],
                     args.output)

    # ---- Final eval ----
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION")
    print(f"{'='*60}")
    final_eval = evaluate_policy(model, opponent_cls_list, device,
                                 num_episodes=args.eval_episodes * 2, seed=9999)
    _print_eval(final_eval)

    # Comparison to baseline
    print(f"\n--- Comparison ---")
    print(f"Baseline wins: {baseline_eval['wins']}/{baseline_eval['episodes']}")
    print(f"Final    wins: {final_eval['wins']}/{final_eval['episodes']}")
    print(f"Baseline STOP: {baseline_eval['stop_pct']:.1f}%")
    print(f"Final    STOP: {final_eval['stop_pct']:.1f}%")
    print(f"Baseline BOMB: {baseline_eval['bomb_pct']:.1f}%")
    print(f"Final    BOMB: {final_eval['bomb_pct']:.1f}%")
    print(f"Baseline deaths: {baseline_eval['deaths']}")
    print(f"Final    deaths: {final_eval['deaths']}")

    return round_results


def _print_eval(results):
    print(f"  W{results['wins']}/D{results['draws']}/L{results['losses']} "
          f"(win rate: {results['wins']/max(1,results['episodes'])*100:.0f}%) | "
          f"deaths={results['deaths']} | "
          f"STOP={results['stop_pct']:.1f}% BOMB={results['bomb_pct']:.1f}% | "
          f"lat={results.get('latency_mean_ms', 0):.2f}ms")


def _save_checkpoint(model, optimizer, round_idx, eval_results, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "round": round_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "eval_results": eval_results,
    }, path)


def main():
    parser = argparse.ArgumentParser(
        description="Safety-constrained PPO fine-tuning for hybrid PPO agent.")
    parser.add_argument("--checkpoint", required=True,
                        help="Imitation checkpoint to fine-tune from")
    parser.add_argument("--output", default="ml/checkpoints/hybrid_ppo/ppo_finetuned.pt")
    parser.add_argument("--opponents", nargs="+",
                        default=["agent/tactical_rule_agent.py",
                                 "agent/smarter_rule_agent.py",
                                 "agent/genius_rule_agent.py"],
                        help="Rule-based opponents for match diversity")
    parser.add_argument("--rounds", type=int, default=40,
                        help="Number of PPO rounds")
    parser.add_argument("--episodes_per_round", type=int, default=8,
                        help="Episodes collected per round")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                        help="PPO update epochs per round")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--kl_coef", type=float, default=1.0,
                        help="KL penalty coefficient (keeps policy near imitation)")
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate every N rounds")
    parser.add_argument("--eval_episodes", type=int, default=20,
                        help="Episodes per evaluation")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
