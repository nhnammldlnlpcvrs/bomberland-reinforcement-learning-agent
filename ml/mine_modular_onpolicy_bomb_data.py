from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import RandomAgent, SimpleRuleAgent
from agent.rl_agent_pure.action_mask import highest_prob_valid, legal_action_mask
from agent.rl_agent_pure.constants import PLACE_BOMB
from agent.rl_agent_pure.encoder import encode_observation
from agent.rl_agent_pure.utils import boxes_in_blast, has_escape_after_bomb, normalize_obs, reachable_area, compute_danger_map, bomb_positions
from agent.rl_agent_recurrent.modular_model import ModularBomberCnnLstm
from engine.game import BomberEnv


SOURCE_NAMES = np.asarray(
    [
        "onpolicy_zero_value_bomb",
        "onpolicy_death_after_bomb",
        "onpolicy_useful_bomb",
        "onpolicy_safe_nonbomb",
        "onpolicy_potential_bomb_avoided",
    ],
    dtype=object,
)

OPPONENTS = {
    "random": RandomAgent,
    "simple": SimpleRuleAgent,
}


def load_model(path: str, device: torch.device) -> ModularBomberCnnLstm:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    model = ModularBomberCnnLstm(
        in_channels=int(config.get("in_channels", 19)),
        embedding_dim=int(config.get("embedding_dim", 128)),
        hidden_size=int(config.get("hidden_size", 128)),
        num_lstm_layers=int(config.get("num_lstm_layers", 1)),
        dropout=float(config.get("dropout", 0.0)),
        layer_norm=bool(config.get("layer_norm", False)),
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    model.eval()
    return model


def make_seq(obs_plane: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    seq = np.zeros((seq_len, obs_plane.shape[0], obs_plane.shape[1], obs_plane.shape[2]), dtype=np.float32)
    mask = np.zeros(seq_len, dtype=bool)
    seq[0] = obs_plane.astype(np.float32)
    mask[0] = True
    return seq, mask


def first_valid_score(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    first = mask.float().argmax(dim=1)
    return logits[torch.arange(logits.shape[0], device=logits.device), first]


@torch.no_grad()
def policy_step(model, hidden, obs, agent_id, safety_threshold, value_threshold, device):
    encoded = encode_observation(obs, agent_id)
    tensor = torch.as_tensor(encoded[None, None], dtype=torch.float32, device=device)
    out, hidden = model(tensor, hidden)
    movement_probs = torch.softmax(out["movement_logits"][0, -1], dim=-1).cpu().numpy()
    escape_probs = torch.softmax(out["escape_logits"][0, -1], dim=-1).cpu().numpy()
    safety_prob = float(torch.sigmoid(out["bomb_logit"][0, -1]).item())
    value_prob = float(torch.sigmoid(out["bomb_value_logit"][0, -1]).item())
    mask = legal_action_mask(obs, agent_id)
    if safety_prob >= safety_threshold and value_prob >= value_threshold and mask[PLACE_BOMB]:
        action = PLACE_BOMB
    else:
        action = highest_prob_valid(np.r_[movement_probs, -np.inf], obs, agent_id)
    return int(action), hidden, encoded, safety_prob, value_prob, mask


def append_record(records, obs_seq, valid_mask, safety_prob, value_prob, action, legal, label_value, label_zero, label_death, label_useful, boxes, survived, death7, threshold_id, opponent_id, seed, step, source_type, reachable_delta=0.0, enemy_pressure=0.0):
    records["observations"].append(obs_seq)
    records["valid_mask"].append(valid_mask)
    records["safety_prob"].append(float(safety_prob))
    records["value_prob"].append(float(value_prob))
    records["chosen_action"].append(int(action))
    records["legal_actions"].append(legal.astype(bool))
    records["label_value_now"].append(float(label_value))
    records["label_zero_value"].append(float(label_zero))
    records["label_death_after_bomb"].append(float(label_death))
    records["label_useful_bomb"].append(float(label_useful))
    records["boxes_destroyed"].append(float(boxes))
    records["survived_after_bomb"].append(float(survived))
    records["death_within_7"].append(float(death7))
    records["threshold_config"].append(int(threshold_id))
    records["opponent"].append(int(opponent_id))
    records["seed"].append(int(seed))
    records["step"].append(int(step))
    records["source_type"].append(int(source_type))
    records["reachable_delta"].append(float(reachable_delta))
    records["enemy_pressure_proxy"].append(float(enemy_pressure))


def initial_records():
    return {
        "observations": [],
        "valid_mask": [],
        "safety_prob": [],
        "value_prob": [],
        "chosen_action": [],
        "legal_actions": [],
        "label_value_now": [],
        "label_zero_value": [],
        "label_death_after_bomb": [],
        "label_useful_bomb": [],
        "boxes_destroyed": [],
        "survived_after_bomb": [],
        "death_within_7": [],
        "threshold_config": [],
        "opponent": [],
        "seed": [],
        "step": [],
        "source_type": [],
        "reachable_delta": [],
        "enemy_pressure_proxy": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine on-policy modular bomb decision states with actual outcomes.")
    parser.add_argument("--checkpoint", default="ml/checkpoints/rl_agent_recurrent/modular_bomb_value_now.pt")
    parser.add_argument("--output", default="ml/datasets/modular_onpolicy_bomb_data.npz")
    parser.add_argument("--opponents", nargs="+", default=["random", "simple"])
    parser.add_argument("--thresholds", default="0.5:0.6,0.5:0.7")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    threshold_pairs = [tuple(float(x) for x in item.split(":")) for item in args.thresholds.split(",")]
    records = initial_records()
    action_counts = {str(i): 0 for i in range(6)}
    env = BomberEnv(max_steps=args.max_steps, seed=args.seed)

    for threshold_id, (safety_threshold, value_threshold) in enumerate(threshold_pairs):
        for opponent_id, opponent_name in enumerate(args.opponents):
            opponent_cls = OPPONENTS[opponent_name]
            for ep in range(args.episodes):
                episode_seed = args.seed + threshold_id * 100000 + opponent_id * 10000 + ep
                rng = random.Random(episode_seed)
                slot = rng.randrange(4)
                agents = [opponent_cls(i) for i in range(4)]
                hidden = None
                obs = {**env.reset(seed=episode_seed), "step": 0}
                done = False
                step = 0
                pending_bombs = []
                while not done and step < args.max_steps:
                    prev_obs = obs
                    action, hidden, encoded, safety_prob, value_prob, legal = policy_step(
                        model, hidden, obs, slot, safety_threshold, value_threshold, device
                    )
                    actions = []
                    for idx, agent in enumerate(agents):
                        actions.append(action if idx == slot else int(agent.act(obs)))
                    board, players, bombs, _ = normalize_obs(prev_obs)
                    row, col = int(players[slot, 0]), int(players[slot, 1])
                    would_boxes = boxes_in_blast(board, players, row, col, slot) if legal[PLACE_BOMB] else 0
                    escape_available = has_escape_after_bomb(board, players, bombs, slot) if legal[PLACE_BOMB] else False
                    pos = (row, col)
                    area_before = float(reachable_area(board, bomb_positions(bombs), compute_danger_map(board, players, bombs), pos).sum())
                    seq, seq_mask = make_seq(encoded, args.seq_len)

                    if action == PLACE_BOMB:
                        pending_bombs.append(
                            {
                                "obs_seq": seq,
                                "valid_mask": seq_mask,
                                "safety_prob": safety_prob,
                                "value_prob": value_prob,
                                "action": action,
                                "legal": legal,
                                "start_step": step,
                                "initial_boxes": int((board == 2).sum()),
                                "area_before": area_before,
                                "escape_available": escape_available,
                                "seed": episode_seed,
                                "opponent_id": opponent_id,
                                "threshold_id": threshold_id,
                                "step": step,
                                "resolved": False,
                            }
                        )
                    elif legal[PLACE_BOMB] and escape_available:
                        # Keep real non-bomb states as negatives/avoided positives.
                        source = 4 if would_boxes > 0 else 3
                        append_record(
                            records,
                            seq,
                            seq_mask,
                            safety_prob,
                            value_prob,
                            action,
                            legal,
                            0.0,
                            1.0 if would_boxes <= 0 else 0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            threshold_id,
                            opponent_id,
                            episode_seed,
                            step,
                            source,
                        )

                    obs, terminated, truncated = env.step(actions)
                    obs = {**obs, "step": step + 1}
                    done = terminated or truncated
                    step += 1
                    action_counts[str(action)] += 1
                    alive_now = bool(obs["players"][slot, 2])
                    board_now, players_now, bombs_now, _ = normalize_obs(obs)
                    pos_now = (int(players_now[slot, 0]), int(players_now[slot, 1]))
                    area_now = float(reachable_area(board_now, bomb_positions(bombs_now), compute_danger_map(board_now, players_now, bombs_now), pos_now).sum()) if alive_now else 0.0
                    for event in pending_bombs:
                        if event["resolved"]:
                            continue
                        age = step - event["start_step"]
                        if age < 8 and alive_now:
                            continue
                        boxes_destroyed = max(0, event["initial_boxes"] - int((board_now == 2).sum()))
                        death7 = (not alive_now) and age <= 7
                        survived = alive_now and age >= 8
                        useful = boxes_destroyed > 0 and survived and not death7
                        zero_value = boxes_destroyed <= 0 and not useful
                        if death7:
                            source_type = 1
                        elif useful:
                            source_type = 2
                        else:
                            source_type = 0
                        append_record(
                            records,
                            event["obs_seq"],
                            event["valid_mask"],
                            event["safety_prob"],
                            event["value_prob"],
                            event["action"],
                            event["legal"],
                            1.0 if useful else 0.0,
                            1.0 if zero_value else 0.0,
                            1.0 if death7 else 0.0,
                            1.0 if useful else 0.0,
                            boxes_destroyed,
                            1.0 if survived else 0.0,
                            1.0 if death7 else 0.0,
                            event["threshold_id"],
                            event["opponent_id"],
                            event["seed"],
                            event["step"],
                            source_type,
                            reachable_delta=area_now - event["area_before"],
                        )
                        event["resolved"] = True

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "observations": np.asarray(records["observations"], dtype=np.float32),
        "valid_mask": np.asarray(records["valid_mask"], dtype=bool),
        "safety_prob": np.asarray(records["safety_prob"], dtype=np.float32),
        "value_prob": np.asarray(records["value_prob"], dtype=np.float32),
        "chosen_action": np.asarray(records["chosen_action"], dtype=np.int64),
        "legal_actions": np.asarray(records["legal_actions"], dtype=bool),
        "label_value_now": np.asarray(records["label_value_now"], dtype=np.float32),
        "label_zero_value": np.asarray(records["label_zero_value"], dtype=np.float32),
        "label_death_after_bomb": np.asarray(records["label_death_after_bomb"], dtype=np.float32),
        "label_useful_bomb": np.asarray(records["label_useful_bomb"], dtype=np.float32),
        "boxes_destroyed": np.asarray(records["boxes_destroyed"], dtype=np.float32),
        "survived_after_bomb": np.asarray(records["survived_after_bomb"], dtype=np.float32),
        "death_within_7": np.asarray(records["death_within_7"], dtype=np.float32),
        "threshold_config": np.asarray(records["threshold_config"], dtype=np.int16),
        "opponent": np.asarray(records["opponent"], dtype=np.int16),
        "seed": np.asarray(records["seed"], dtype=np.int64),
        "step": np.asarray(records["step"], dtype=np.int16),
        "source_type": np.asarray(records["source_type"], dtype=np.int16),
        "reachable_delta": np.asarray(records["reachable_delta"], dtype=np.float32),
        "enemy_pressure_proxy": np.asarray(records["enemy_pressure_proxy"], dtype=np.float32),
        "source_type_names": SOURCE_NAMES,
    }
    np.savez_compressed(output, **arrays)
    source = arrays["source_type"]
    report = {
        "output": str(output),
        "total_states_collected": int(len(source)),
        "bomb_decisions": int(np.sum(arrays["chosen_action"] == PLACE_BOMB)),
        "zero_value_bombs": int(np.sum(source == 0)),
        "death_after_bomb": int(np.sum(source == 1)),
        "useful_bombs": int(np.sum(source == 2)),
        "safe_nonbomb_candidates": int(np.sum(source == 3)),
        "potential_bomb_avoided": int(np.sum(source == 4)),
        "action_distribution": action_counts,
        "label_value_positive": int(np.sum(arrays["label_value_now"] > 0.5)),
        "label_zero_value": int(np.sum(arrays["label_zero_value"] > 0.5)),
        "label_death_after_bomb": int(np.sum(arrays["label_death_after_bomb"] > 0.5)),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
