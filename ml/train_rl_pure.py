"""Train the isolated pure RL Bomberland Dueling Double-DQN track."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AGENT_DIR = ROOT / "agent" / "hybrid_agent_rl_pure"
if str(AGENT_DIR) not in sys.path:
    sys.path.append(str(AGENT_DIR))

from agent import BoxFarmerAgent, GeniusRuleAgent, RandomAgent, SimpleRuleAgent, SmarterRuleAgent, TacticalRuleAgent
from competition.evaluation.runtime_guard import load_agent_instance
from engine.game import BomberEnv
from model import DuelingDQN, encode_observation, fallback_policy, safe_action_mask


BASELINES = {
    "random": RandomAgent,
    "RandomAgent": RandomAgent,
    "simple": SimpleRuleAgent,
    "SimpleRuleAgent": SimpleRuleAgent,
    "box_farmer": BoxFarmerAgent,
    "BoxFarmerAgent": BoxFarmerAgent,
    "smarter": SmarterRuleAgent,
    "SmarterRuleAgent": SmarterRuleAgent,
    "tactical": TacticalRuleAgent,
    "TacticalRuleAgent": TacticalRuleAgent,
    "genius": GeniusRuleAgent,
    "GeniusRuleAgent": GeniusRuleAgent,
}


class ReplayBuffer:
    def __init__(self, capacity, spatial_shape, scalar_dim):
        self.capacity = int(capacity)
        self.pos = 0
        self.size = 0
        self.spatial = np.zeros((capacity, *spatial_shape), dtype=np.float32)
        self.scalar = np.zeros((capacity, scalar_dim), dtype=np.float32)
        self.next_spatial = np.zeros((capacity, *spatial_shape), dtype=np.float32)
        self.next_scalar = np.zeros((capacity, scalar_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, spatial, scalar, action, reward, next_spatial, next_scalar, done):
        self.spatial[self.pos] = spatial
        self.scalar[self.pos] = scalar
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_spatial[self.pos] = next_spatial
        self.next_scalar[self.pos] = next_scalar
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.spatial[idx]),
            torch.from_numpy(self.scalar[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_spatial[idx]),
            torch.from_numpy(self.next_scalar[idx]),
            torch.from_numpy(self.dones[idx]),
        )


class QPolicyAgent:
    def __init__(self, agent_id, q_net, epsilon, device="cpu"):
        self.agent_id = int(agent_id)
        self.q_net = q_net
        self.epsilon = float(epsilon)
        self.device = device
        self.last_q = None

    def act(self, obs):
        mask = safe_action_mask(obs, self.agent_id)
        if random.random() < self.epsilon:
            safe = np.flatnonzero(mask)
            return int(random.choice(safe.tolist())) if safe.size else 0
        spatial, scalar = encode_observation(obs, self.agent_id)
        with torch.no_grad():
            q = self.q_net(
                torch.from_numpy(spatial).unsqueeze(0).to(self.device),
                torch.from_numpy(scalar).unsqueeze(0).to(self.device),
            ).squeeze(0).cpu().numpy()
        self.last_q = q
        return fallback_policy(obs, self.agent_id, q)


def load_config():
    return json.loads((AGENT_DIR / "config.json").read_text(encoding="utf-8"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def epsilon_at(step, cfg):
    start = cfg["epsilon_start"]
    final = cfg["epsilon_final"]
    decay = max(1, int(cfg["epsilon_decay_steps"]))
    frac = min(1.0, step / decay)
    return final + (start - final) * (1.0 - frac)


def make_baseline(name, agent_id):
    if name in BASELINES:
        return BASELINES[name](agent_id)
    path = Path(name)
    if path.is_dir():
        path = path / "agent.py"
    return load_agent_instance(str(path), agent_id)


def stage_opponents(stage, config, self_play_pool=None):
    if stage == "a":
        return ["random", "simple"]
    if stage == "b":
        return ["simple", "box_farmer", "smarter", "tactical", "genius"]
    if stage == "d":
        return config["curriculum"]["stage_d_eval"]
    pool = []
    if self_play_pool:
        pool = [str(p) for p in Path(self_play_pool).glob("*.pth")]
    return ["simple", "box_farmer", "smarter", "tactical", "genius", *pool]


def box_count(obs):
    return int((np.asarray(obs["map"]) == 2).sum())


def item_power(players, agent_id):
    p = players[agent_id]
    return int(p[3]) + int(p[4])


def reward_fn(prev_obs, obs, agent_id, action, done, history, weights):
    prev_players = np.asarray(prev_obs["players"])
    players = np.asarray(obs["players"])
    reward = 0.0
    if int(players[agent_id, 2]):
        reward += weights["survival"]
    else:
        reward += weights["death"]
    prev_enemy_alive = sum(int(p[2]) for i, p in enumerate(prev_players) if i != agent_id)
    enemy_alive = sum(int(p[2]) for i, p in enumerate(players) if i != agent_id)
    reward += max(0, prev_enemy_alive - enemy_alive) * weights["kill_enemy"]
    reward += max(0, box_count(prev_obs) - box_count(obs)) * weights["box_destroyed"]
    reward += max(0, item_power(players, agent_id) - item_power(prev_players, agent_id)) * weights["item_collected"]
    if action == 5:
        mask = safe_action_mask(prev_obs, agent_id)
        reward += weights["bomb_with_target"] if mask[5] else weights["unsafe_bomb"]
    if not safe_action_mask(prev_obs, agent_id)[action]:
        reward += weights["invalid_action"]
    try:
        spatial, _ = encode_observation(obs, agent_id)
        danger_now = spatial[10][int(players[agent_id, 0]), int(players[agent_id, 1])]
        reward += float(danger_now) * weights["danger"]
    except Exception:
        pass
    pos = (int(players[agent_id, 0]), int(players[agent_id, 1]))
    history.append(pos)
    if len(history) == history.maxlen:
        common = Counter(history).most_common(1)[0][1]
        if common >= max(4, len(history) // 2):
            reward += weights["loop"]
    if done:
        alive = [i for i, p in enumerate(players) if int(p[2])]
        if alive == [agent_id]:
            reward += weights["win"]
        elif agent_id in alive:
            reward += weights["draw_survive"]
        rank_proxy = len([i for i, p in enumerate(players) if i != agent_id and not int(p[2])])
        reward += weights["rank_scale"] * (rank_proxy / 3.0)
    return float(reward)


def optimize(q_net, target_net, optimizer, replay, cfg, device):
    spatial, scalar, actions, rewards, next_spatial, next_scalar, dones = replay.sample(cfg["batch_size"])
    spatial = spatial.to(device)
    scalar = scalar.to(device)
    actions = actions.to(device).unsqueeze(1)
    rewards = rewards.to(device).unsqueeze(1)
    next_spatial = next_spatial.to(device)
    next_scalar = next_scalar.to(device)
    dones = dones.to(device).unsqueeze(1)
    q = q_net(spatial, scalar).gather(1, actions)
    with torch.no_grad():
        next_actions = q_net(next_spatial, next_scalar).argmax(dim=1, keepdim=True)
        next_q = target_net(next_spatial, next_scalar).gather(1, next_actions)
        target = rewards + cfg["gamma"] * next_q * (1.0 - dones)
    loss = F.smooth_l1_loss(q, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), cfg["gradient_clip_norm"])
    optimizer.step()
    return float(loss.item())


def save_checkpoint(path, q_net, target_net, optimizer, cfg, step, episode, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": q_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": cfg,
        "global_step": step,
        "episode": episode,
        "metrics": metrics,
    }
    torch.save(payload, path)
    latest = path.parent / "latest.pth"
    shutil.copyfile(path, latest)


def train(args):
    cfg = load_config()
    seed_all(args.seed)
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    q_net = DuelingDQN(cfg["spatial_channels"], cfg["scalar_dim"], cfg["num_actions"], cfg["hidden_dim"]).to(device)
    target_net = DuelingDQN(cfg["spatial_channels"], cfg["scalar_dim"], cfg["num_actions"], cfg["hidden_dim"]).to(device)
    optimizer = optim.Adam(q_net.parameters(), lr=cfg["learning_rate"])
    global_step = 0
    start_episode = 0
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        q_net.load_state_dict(ckpt["model_state_dict"])
        target_net.load_state_dict(ckpt.get("target_state_dict", ckpt["model_state_dict"]))
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = int(ckpt.get("global_step", 0))
        start_episode = int(ckpt.get("episode", 0))
    target_net.load_state_dict(q_net.state_dict())
    replay = ReplayBuffer(cfg["replay_capacity"], (cfg["spatial_channels"], 13, 13), cfg["scalar_dim"])
    env = BomberEnv(max_steps=args.max_steps, seed=args.seed)
    opponents = stage_opponents(args.stage, cfg, args.self_play_pool)
    metrics = {"episodes": [], "loss": []}

    for episode in range(start_episode, start_episode + args.episodes):
        agent_id = random.randint(0, 3)
        obs = env.reset(seed=args.seed + episode)
        epsilon = epsilon_at(global_step, cfg)
        agents = []
        for idx in range(4):
            if idx == agent_id:
                agents.append(QPolicyAgent(idx, q_net, epsilon, device=device))
            else:
                agents.append(make_baseline(random.choice(opponents), idx))
        history = deque(maxlen=10)
        total_reward = 0.0
        losses = []
        for _ in range(args.max_steps):
            spatial, scalar = encode_observation(obs, agent_id)
            actions = [int(agent.act(obs)) for agent in agents]
            action = int(actions[agent_id])
            next_obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            reward = reward_fn(obs, next_obs, agent_id, action, done, history, cfg["reward"])
            next_spatial, next_scalar = encode_observation(next_obs, agent_id)
            replay.push(spatial, scalar, action, reward, next_spatial, next_scalar, done)
            total_reward += reward
            global_step += 1
            if len(replay) >= cfg["warmup_steps"] and global_step % cfg["train_every_steps"] == 0:
                losses.append(optimize(q_net, target_net, optimizer, replay, cfg, device))
            if global_step % cfg["target_update_steps"] == 0:
                target_net.load_state_dict(q_net.state_dict())
            obs = next_obs
            if done:
                break
        alive = bool(obs["players"][agent_id][2])
        metrics["episodes"].append({"episode": episode + 1, "reward": total_reward, "alive": alive, "epsilon": epsilon})
        if losses:
            metrics["loss"].append(float(np.mean(losses)))
        if (episode + 1) % args.checkpoint_every == 0 or episode == start_episode + args.episodes - 1:
            out = Path(args.output_dir) / f"rl_pure_ep{episode + 1}_step{global_step}.pth"
            save_checkpoint(out, q_net, target_net, optimizer, cfg, global_step, episode + 1, metrics)
        print(f"episode={episode + 1} reward={total_reward:.2f} alive={alive} eps={epsilon:.3f} replay={len(replay)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["a", "b", "c", "d"], default="a")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--self_play_pool", default=None)
    parser.add_argument("--output_dir", default="ml/checkpoints/rl_pure")
    parser.add_argument("--checkpoint_every", type=int, default=50)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
