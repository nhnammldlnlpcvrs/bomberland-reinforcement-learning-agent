from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rl_agent_pure.aux_model import BomberAuxModel
from agent.rl_agent_pure.constants import PLACE_BOMB
from agent.rl_agent_pure.model import BomberFeaturesExtractor
from agent.rl_agent_pure.utils import boxes_in_blast, normalize_obs
from ml.envs.bomber_gym_env import BomberGymEnv


def _load_aux(path: str, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = BomberAuxModel(features_dim=int(checkpoint.get("features_dim", 256))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def analyze(args):
    device = torch.device(args.device)
    aux = _load_aux(args.aux_model, args.device)
    if args.thresholds and Path(args.thresholds).exists():
        thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
        args.death_threshold = float(thresholds.get("death", args.death_threshold))
        args.safe_escape_threshold = float(thresholds.get("escape_available", args.safe_escape_threshold))
        args.bomb_escape_threshold = float(thresholds.get("bomb_escape_available", args.bomb_escape_threshold))
        args.trapped_threshold = float(thresholds.get("trapped_if_bomb", args.trapped_threshold))
        args.future_blast_threshold = float(thresholds.get("future_blast", args.future_blast_threshold))
    policy_kwargs = {
        "features_extractor_class": BomberFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    policy = PPO.load(args.policy, device=args.device, custom_objects={"policy_kwargs": policy_kwargs})
    env = BomberGymEnv(agent_id=args.agent_id, opponent_pool=args.opponents, max_steps=args.max_steps, seed=args.seed)

    counters = {
        "states": 0,
        "high_death_risk": 0,
        "no_escape_available": 0,
        "no_bomb_escape_available": 0,
        "trapped_if_bomb": 0,
        "future_blast": 0,
        "bombs_on_high_risk": 0,
        "safe_useful_bomb_avoided": 0,
        "bomb_actions": 0,
    }
    examples = []
    with torch.no_grad():
        for episode in range(args.episodes):
            obs, _info = env.reset(seed=args.seed + episode)
            done = False
            truncated = False
            while not (done or truncated):
                action, _state = policy.predict(obs, deterministic=True)
                action = int(np.asarray(action).reshape(-1)[0])
                tensor = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
                out = aux(tensor)
                death_prob = float(torch.sigmoid(out["death_logit"])[0].cpu())
                escape_available_prob = float(torch.sigmoid(out["escape_available_logit"])[0].cpu())
                bomb_escape_prob = float(torch.sigmoid(out["bomb_escape_available_logit"])[0].cpu())
                trapped_prob = float(torch.sigmoid(out["trapped_if_bomb_logit"])[0].cpu())
                future_blast_prob = float(torch.sigmoid(out["future_blast_logit"])[0].cpu())
                obs_dict = env.last_obs
                board, players, bombs, _ = normalize_obs(obs_dict)
                pos = (int(players[args.agent_id, 0]), int(players[args.agent_id, 1]))
                boxes = boxes_in_blast(board, players, pos[0], pos[1], args.agent_id)
                counters["states"] += 1
                if death_prob >= args.death_threshold:
                    counters["high_death_risk"] += 1
                if escape_available_prob < args.safe_escape_threshold:
                    counters["no_escape_available"] += 1
                if bomb_escape_prob < args.bomb_escape_threshold:
                    counters["no_bomb_escape_available"] += 1
                if trapped_prob >= args.trapped_threshold:
                    counters["trapped_if_bomb"] += 1
                if future_blast_prob >= args.future_blast_threshold:
                    counters["future_blast"] += 1
                if action == PLACE_BOMB:
                    counters["bomb_actions"] += 1
                    if (
                        death_prob >= args.death_threshold
                        or bomb_escape_prob < args.bomb_escape_threshold
                        or trapped_prob >= args.trapped_threshold
                    ):
                        counters["bombs_on_high_risk"] += 1
                elif boxes > 0 and death_prob < args.safe_death_threshold and bomb_escape_prob >= args.bomb_escape_threshold:
                    counters["safe_useful_bomb_avoided"] += 1
                    if len(examples) < args.max_examples:
                        examples.append({
                            "episode": episode,
                            "step": int(env.env.current_step),
                            "pos": pos,
                            "boxes_in_blast": int(boxes),
                            "death_prob": death_prob,
                            "escape_available_prob": escape_available_prob,
                            "bomb_escape_prob": bomb_escape_prob,
                            "trapped_prob": trapped_prob,
                            "future_blast_prob": future_blast_prob,
                            "action": action,
                        })
                obs, _reward, done, truncated, _info = env.step(action)

    report = {"counters": counters, "examples": examples}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Analyze normal-env rollouts with auxiliary risk/value model.")
    parser.add_argument("--policy", default="ml/checkpoints/rl_agent_pure/ppo_bomb_selector_best.zip")
    parser.add_argument("--aux_model", default="ml/checkpoints/rl_agent_pure/aux_curriculum_model.pt")
    parser.add_argument("--output", default="logs/aux_failure_analysis.json")
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=6300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponents", nargs="*", default=["random", "simple"])
    parser.add_argument("--death_threshold", type=float, default=0.7)
    parser.add_argument("--escape_failure_threshold", type=float, default=0.3)
    parser.add_argument("--safe_death_threshold", type=float, default=0.3)
    parser.add_argument("--safe_escape_threshold", type=float, default=0.7)
    parser.add_argument("--bomb_escape_threshold", type=float, default=0.7)
    parser.add_argument("--trapped_threshold", type=float, default=0.5)
    parser.add_argument("--future_blast_threshold", type=float, default=0.5)
    parser.add_argument("--max_examples", type=int, default=20)
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
