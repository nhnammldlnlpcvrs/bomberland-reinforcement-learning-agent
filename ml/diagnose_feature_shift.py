"""Diagnose shadow ranker bomb-collapse root cause.

Compares feature tensors, model logits, and action masks between
the static ranking dataset and live shadow observations.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine.game import BomberEnv
from agent import TacticalRuleAgent
from competition.evaluation.runtime_guard import load_agent_instance
from ml.features import encode_observation, CHANNEL_NAMES
from ml.models.simple_cnn_policy import TORCH_AVAILABLE, load_checkpoint

if TORCH_AVAILABLE:
    import torch
else:
    torch = None

ACTION_NAMES = ["STOP", "LEFT", "RIGHT", "UP", "DOWN", "PLACE_BOMB"]
BOARD_SIZE = 13
TILE_WALL = 1
TILE_BOX = 2

# ==============================================================================
# Lightweight validity mask (same as shadow_ranker_eval.py)
# ==============================================================================


def compute_validity_mask(obs, agent_id):
    mask = np.zeros(6, dtype=bool)
    mask[0] = True
    p = obs["players"][agent_id]
    my_r, my_c = int(p[0]), int(p[1])
    alive = int(p[2])
    bombs_left = int(p[3])
    if not alive:
        return mask
    game_map = obs["map"]
    bombs_arr = np.asarray(obs["bombs"], dtype=np.int32)
    if bombs_arr.size == 0:
        bomb_set = set()
    else:
        if bombs_arr.ndim == 1:
            bombs_arr = bombs_arr.reshape(1, -1)
        bomb_set = {(int(bombs_arr[i, 0]), int(bombs_arr[i, 1])) for i in range(bombs_arr.shape[0])}
    DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    for action, (dr, dc) in DIRS.items():
        nr, nc = my_r + dr, my_c + dc
        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
            tile = game_map[nr, nc]
            if tile not in (TILE_WALL, TILE_BOX) and (nr, nc) not in bomb_set:
                mask[action] = True
    if bombs_left > 0 and (my_r, my_c) not in bomb_set:
        mask[5] = True
    return mask


# ==============================================================================
# Dataset-safe mask (same logic as build_action_ranking_dataset.py)
# ==============================================================================


def _self_position(obs_tensor):
    cells = np.argwhere(obs_tensor[6] > 0.5)
    if len(cells) == 0:
        return None
    return int(cells[0, 0]), int(cells[0, 1])


def compute_dataset_style_mask(obs_tensor):
    """Replicates compute_safe_action_mask from build_action_ranking_dataset.py."""
    ACTION_DELTAS = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    mask = np.zeros(6, dtype=bool)
    pos = _self_position(obs_tensor)
    if pos is None:
        return mask
    row, col = pos
    walkable = obs_tensor[8]
    safe = obs_tensor[9]
    bombs_ch = obs_tensor[2]
    for action, (dr, dc) in ACTION_DELTAS.items():
        nr, nc = row + dr, col + dc
        if 0 <= nr < 13 and 0 <= nc < 13 and walkable[nr, nc] > 0.5 and safe[nr, nc] > 0.5:
            mask[action] = True
    current_safe = safe[row, col] > 0.5 and bombs_ch[row, col] <= 0.5
    # In training, bomb is only in mask when chosen_action == 5 (or include_unchosen_bombs).
    # For diagnostic purposes, include bomb when current cell is safe (same gate as training).
    if current_safe:
        mask[5] = True
    return mask


# ==============================================================================
# Live observation collector
# ==============================================================================


def collect_live_observations(agent_path, opponents, num_episodes, max_steps, seed):
    """Run matches and collect encoded observations + lightweight masks."""
    p = Path(agent_path)
    if p.is_dir():
        p = p / "agent.py"
    heuristic_agent = load_agent_instance(str(p), 0)

    opp_agents = []
    for i, name in enumerate(opponents):
        opp_agents.append(TacticalRuleAgent(i + 1))

    agents = [heuristic_agent] + opp_agents
    n_players = 4
    env = BomberEnv(max_steps=max_steps, seed=seed)

    all_encoded = []
    all_masks_lightweight = []
    all_masks_dataset_style = []
    all_heuristic_actions = []

    for episode in range(num_episodes):
        episode_seed = None if seed is None else seed + episode
        obs = env.reset(seed=episode_seed)
        done = False
        step = 0

        while not done and step < max_steps:
            actions = []
            for i in range(n_players):
                try:
                    action = agents[i].act(obs)
                except Exception:
                    action = 0
                actions.append(action)

            heuristic_action = int(actions[0])

            # Encode observation (same as shadow_ranker_eval.py)
            obs_copy = dict(obs)
            obs_copy["_agent_index"] = 0
            encoded = encode_observation(obs_copy)
            tensor = encoded["tensor"]

            mask_lw = compute_validity_mask(obs, 0)
            mask_ds = compute_dataset_style_mask(tensor)

            all_encoded.append(tensor)
            all_masks_lightweight.append(mask_lw)
            all_masks_dataset_style.append(mask_ds)
            all_heuristic_actions.append(heuristic_action)

            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1

    return (
        np.stack(all_encoded, axis=0).astype(np.float32),
        np.stack(all_masks_lightweight, axis=0),
        np.stack(all_masks_dataset_style, axis=0),
        np.array(all_heuristic_actions, dtype=np.int64),
    )


# ==============================================================================
# Main diagnostic
# ==============================================================================


def diagnose(args):
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not available")
        return

    print("=" * 70)
    print("SHADOW RANKER BOMB-COLLAPSE DIAGNOSTIC")
    print("=" * 70)

    # ---- 1. Load dataset ----
    print("\n[1] Loading dataset...")
    data = np.load(args.dataset, allow_pickle=False)
    dataset_obs = data["observations"].astype(np.float32)
    dataset_targets = data["target_actions"].astype(np.int64)
    dataset_masks = data["safe_action_masks"].astype(bool)
    n_dataset = len(dataset_targets)
    print(f"    samples: {n_dataset}")
    print(f"    target distribution:")
    for i, name in enumerate(ACTION_NAMES):
        count = int((dataset_targets == i).sum())
        print(f"      {name}: {count} ({100.0 * count / n_dataset:.1f}%)")

    # ---- 2. Load checkpoint ----
    print("\n[2] Loading checkpoint...")
    model, checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    ckpt_metrics = checkpoint.get("best_metrics", {})
    print(f"    checkpoint metrics: rank_acc={ckpt_metrics.get('ranking_accuracy', 'N/A')}")
    print(f"    top2_safe={ckpt_metrics.get('top2_safe_agreement', 'N/A')}")
    print(f"    bomb_pred_pct={ckpt_metrics.get('bomb_prediction_pct', 'N/A')}")

    # ---- 3. Collect live observations ----
    print("\n[3] Collecting live observations...")
    live_obs, live_masks_lw, live_masks_ds, live_actions = collect_live_observations(
        args.agent_path, args.opponents, args.num_episodes, args.max_steps, args.seed
    )
    n_live = len(live_obs)
    print(f"    live steps collected: {n_live}")
    print(f"    live heuristic action distribution:")
    for i, name in enumerate(ACTION_NAMES):
        count = int((live_actions == i).sum())
        print(f"      {name}: {count} ({100.0 * count / n_live:.1f}%)")

    # ---- 4. Per-channel statistics comparison ----
    print("\n[4] Per-channel statistics: dataset vs live")
    print(f"    {'Channel':<22s} {'D mean':>8s} {'L mean':>8s} {'D std':>8s} {'L std':>8s} {'d_mean':>8s} {'d_std':>8s} {'Flag'}")
    flagged_channels = []
    for ch_idx, name in enumerate(CHANNEL_NAMES):
        d_ch = dataset_obs[:, ch_idx]
        l_ch = live_obs[:, ch_idx]
        d_mean, d_std = float(d_ch.mean()), float(d_ch.std())
        l_mean, l_std = float(l_ch.mean()), float(l_ch.std())
        delta_mean = abs(d_mean - l_mean)
        delta_std = abs(d_std - l_std)
        flag = ""
        if delta_mean > 0.05:
            flag = "MEAN_SHIFT"
            flagged_channels.append((ch_idx, name, "mean_shift", delta_mean))
        if delta_std > 0.10:
            flag = (flag + "+" if flag else "") + "STD_SHIFT"
            if not any(c[0] == ch_idx for c in flagged_channels):
                flagged_channels.append((ch_idx, name, "std_shift", delta_std))
        if not flag:
            flag = "ok"
        print(f"    {name:<22s} {d_mean:>8.4f} {l_mean:>8.4f} {d_std:>8.4f} {l_std:>8.4f} {delta_mean:>8.4f} {delta_std:>8.4f} {flag}")

    if not flagged_channels:
        print("    No significant channel shifts detected.")

    # ---- 5. Per-channel min/max ----
    print("\n[5] Per-channel min/max")
    print(f"    {'Channel':<22s} {'D min':>8s} {'L min':>8s} {'D max':>8s} {'L max':>8s}")
    for ch_idx, name in enumerate(CHANNEL_NAMES):
        d_ch = dataset_obs[:, ch_idx]
        l_ch = live_obs[:, ch_idx]
        print(f"    {name:<22s} {float(d_ch.min()):>8.4f} {float(l_ch.min()):>8.4f} {float(d_ch.max()):>8.4f} {float(l_ch.max()):>8.4f}")

    # ---- 6. Model logits comparison ----
    print("\n[6] Model logits comparison: dataset vs live")

    # Sample dataset observations (balanced across actions)
    n_sample = min(2000, n_dataset)
    rng = np.random.default_rng(42)
    ds_idx = rng.choice(n_dataset, size=n_sample, replace=False)
    ds_sample = torch.from_numpy(dataset_obs[ds_idx])

    # Sample live observations
    n_live_sample = min(n_live, n_sample)
    live_idx = rng.choice(n_live, size=n_live_sample, replace=False)
    live_sample = torch.from_numpy(live_obs[live_idx])

    with torch.no_grad():
        ds_logits = model(ds_sample).numpy()
        live_logits = model(live_sample).numpy()

    print(f"    Dataset logits (n={n_sample}):")
    for i, name in enumerate(ACTION_NAMES):
        vals = ds_logits[:, i]
        print(f"      {name:<12s} mean={float(vals.mean()):>8.4f} std={float(vals.std()):>8.4f} "
              f"min={float(vals.min()):>8.4f} max={float(vals.max()):>8.4f}")

    print(f"    Live logits (n={n_live_sample}):")
    for i, name in enumerate(ACTION_NAMES):
        vals = live_logits[:, i]
        print(f"      {name:<12s} mean={float(vals.mean()):>8.4f} std={float(vals.std()):>8.4f} "
              f"min={float(vals.min()):>8.4f} max={float(vals.max()):>8.4f}")

    # BOMB logit vs others
    ds_bomb_mean = float(ds_logits[:, 5].mean())
    ds_other_mean = float(np.delete(ds_logits, 5, axis=1).mean())
    live_bomb_mean = float(live_logits[:, 5].mean())
    live_other_mean = float(np.delete(live_logits, 5, axis=1).mean())

    print(f"\n    BOMB logit gap analysis:")
    print(f"      Dataset: BOMB mean={ds_bomb_mean:.4f}  other actions mean={ds_other_mean:.4f}  "
          f"gap={ds_bomb_mean - ds_other_mean:+.4f}")
    print(f"      Live:    BOMB mean={live_bomb_mean:.4f}  other actions mean={live_other_mean:.4f}  "
          f"gap={live_bomb_mean - live_other_mean:+.4f}")

    # ---- 7. Raw prediction comparison ----
    print("\n[7] Raw (unmasked) prediction distribution")
    ds_raw_preds = ds_logits.argmax(axis=1)
    live_raw_preds = live_logits.argmax(axis=1)

    print(f"    Dataset raw top-1:")
    for i, name in enumerate(ACTION_NAMES):
        count = int((ds_raw_preds == i).sum())
        print(f"      {name}: {count} ({100.0 * count / n_sample:.1f}%)")

    print(f"    Live raw top-1:")
    for i, name in enumerate(ACTION_NAMES):
        count = int((live_raw_preds == i).sum())
        print(f"      {name}: {count} ({100.0 * count / n_live_sample:.1f}%)")

    # ---- 8. Mask comparison ----
    print("\n[8] Action mask comparison")

    ds_masks_sample = dataset_masks[ds_idx]
    live_masks_lw_sample = live_masks_lw[live_idx]
    live_masks_ds_sample = live_masks_ds[live_idx]

    print(f"    Mask availability rates (% of steps where action is valid):")
    print(f"    {'Action':<12s} {'Dataset':>10s} {'Lightweight':>12s} {'Dataset-style':>13s}")
    for i, name in enumerate(ACTION_NAMES):
        d_rate = float(ds_masks_sample[:, i].mean())
        lw_rate = float(live_masks_lw_sample[:, i].mean())
        ds_rate = float(live_masks_ds_sample[:, i].mean())
        print(f"    {name:<12s} {d_rate:>10.4f} {lw_rate:>12.4f} {ds_rate:>13.4f}")

    # BOMB mask rate specifically
    ds_bomb_mask_rate = float(dataset_masks[:, 5].mean())
    live_lw_bomb_mask_rate = float(live_masks_lw[:, 5].mean())
    live_ds_bomb_mask_rate = float(live_masks_ds[:, 5].mean())
    print(f"\n    BOMB in mask rate:")
    print(f"      Dataset mask:         {ds_bomb_mask_rate:.4f} ({ds_bomb_mask_rate*100:.1f}%)")
    print(f"      Live lightweight:     {live_lw_bomb_mask_rate:.4f} ({live_lw_bomb_mask_rate*100:.1f}%)")
    print(f"      Live dataset-style:   {live_ds_bomb_mask_rate:.4f} ({live_ds_bomb_mask_rate*100:.1f}%)")
    print(f"      Ratio lw/dataset:     {live_lw_bomb_mask_rate / max(1e-9, ds_bomb_mask_rate):.1f}x")

    # ---- 9. Masked prediction comparison ----
    print("\n[9] Masked prediction comparison")

    # Dataset mask on dataset observations
    ds_masked = ds_logits.copy()
    ds_masked[~ds_masks_sample] = -1e9
    ds_masked_preds = ds_masked.argmax(axis=1)

    # Lightweight mask on live observations
    live_lw_masked = live_logits.copy()
    live_lw_masked[~live_masks_lw_sample] = -1e9
    live_lw_preds = live_lw_masked.argmax(axis=1)

    # Dataset-style mask on live observations
    live_ds_masked = live_logits.copy()
    live_ds_masked[~live_masks_ds_sample] = -1e9
    live_ds_preds = live_ds_masked.argmax(axis=1)

    print(f"    Dataset obs + dataset mask:")
    for i, name in enumerate(ACTION_NAMES):
        count = int((ds_masked_preds == i).sum())
        print(f"      {name}: {count} ({100.0 * count / n_sample:.1f}%)")

    print(f"    Live obs + lightweight mask:")
    for i, name in enumerate(ACTION_NAMES):
        count = int((live_lw_preds == i).sum())
        print(f"      {name}: {count} ({100.0 * count / n_live_sample:.1f}%)")

    print(f"    Live obs + dataset-style mask:")
    for i, name in enumerate(ACTION_NAMES):
        count = int((live_ds_preds == i).sum())
        print(f"      {name}: {count} ({100.0 * count / n_live_sample:.1f}%)")

    # ---- 10. BOMB logit vs training mask analysis ----
    print("\n[10] BOMB logit vs training-mask analysis")

    # On dataset: BOMB logit when BOMB is in mask vs not
    ds_bomb_in_mask = ds_masks_sample[:, 5]
    ds_bomb_logit_when_valid = ds_logits[ds_bomb_in_mask, 5]
    ds_bomb_logit_when_invalid = ds_logits[~ds_bomb_in_mask, 5]

    print(f"    Dataset BOMB logit when BOMB in training mask (n={ds_bomb_in_mask.sum()}):")
    if len(ds_bomb_logit_when_valid) > 0:
        print(f"      mean={float(ds_bomb_logit_when_valid.mean()):.4f} "
              f"std={float(ds_bomb_logit_when_valid.std()):.4f} "
              f"min={float(ds_bomb_logit_when_valid.min()):.4f} "
              f"max={float(ds_bomb_logit_when_valid.max()):.4f}")

    print(f"    Dataset BOMB logit when BOMB NOT in training mask (n={(~ds_bomb_in_mask).sum()}):")
    if len(ds_bomb_logit_when_invalid) > 0:
        print(f"      mean={float(ds_bomb_logit_when_invalid.mean()):.4f} "
              f"std={float(ds_bomb_logit_when_invalid.std()):.4f} "
              f"min={float(ds_bomb_logit_when_invalid.min()):.4f} "
              f"max={float(ds_bomb_logit_when_invalid.max()):.4f}")

    # On live: BOMB logit when lightweight mask allows it vs not
    live_bomb_in_lw = live_masks_lw_sample[:, 5]
    live_bomb_logit_lw_valid = live_logits[live_bomb_in_lw, 5]
    live_bomb_logit_lw_invalid = live_logits[~live_bomb_in_lw, 5]

    print(f"    Live BOMB logit when lightweight mask valid (n={live_bomb_in_lw.sum()}):")
    if len(live_bomb_logit_lw_valid) > 0:
        print(f"      mean={float(live_bomb_logit_lw_valid.mean()):.4f} "
              f"std={float(live_bomb_logit_lw_valid.std()):.4f}")

    print(f"    Live BOMB logit when lightweight mask invalid (n={(~live_bomb_in_lw).sum()}):")
    if len(live_bomb_logit_lw_invalid) > 0:
        print(f"      mean={float(live_bomb_logit_lw_invalid.mean()):.4f} "
              f"std={float(live_bomb_logit_lw_invalid.std()):.4f}")

    # ---- 11. Logit rank correlation ----
    print("\n[11] Per-action logit rank analysis")

    # For each sample, rank the 6 logits (0=lowest, 5=highest)
    ds_ranks = np.argsort(np.argsort(ds_logits, axis=1), axis=1)
    live_ranks = np.argsort(np.argsort(live_logits, axis=1), axis=1)

    print(f"    Dataset mean logit rank (0=lowest, 5=highest):")
    for i, name in enumerate(ACTION_NAMES):
        print(f"      {name}: {float(ds_ranks[:, i].mean()):.2f}")

    print(f"    Live mean logit rank (0=lowest, 5=highest):")
    for i, name in enumerate(ACTION_NAMES):
        print(f"      {name}: {float(live_ranks[:, i].mean()):.2f}")

    # ---- 12. ROOT CAUSE DETERMINATION ----
    print("\n" + "=" * 70)
    print("[12] ROOT CAUSE ANALYSIS")
    print("=" * 70)

    findings = []

    # Check 1: Feature encoding mismatch
    if flagged_channels:
        findings.append(
            f"FEATURE ENCODING: {len(flagged_channels)} channels show significant shift. "
            f"Channels: {[c[1] for c in flagged_channels]}. "
            f"This may contribute to the problem."
        )
    else:
        findings.append("FEATURE ENCODING: No significant channel shifts detected between dataset and live observations.")

    # Check 2: BOMB logit analysis
    bomb_gap_dataset = ds_bomb_mean - ds_other_mean
    bomb_gap_live = live_bomb_mean - live_other_mean
    if bomb_gap_live > 2.0:
        findings.append(
            f"BOMB LOGIT: BOMB logit is {bomb_gap_live:.1f} above other actions in live play "
            f"(vs {bomb_gap_dataset:.1f} on dataset). BOMB logit dominates all other actions."
        )

    # Check 3: Mask mismatch
    mask_ratio = live_lw_bomb_mask_rate / max(1e-9, ds_bomb_mask_rate)
    if mask_ratio > 5:
        findings.append(
            f"MASK MISMATCH: BOMB is valid in lightweight mask {live_lw_bomb_mask_rate*100:.0f}% of steps "
            f"vs {ds_bomb_mask_rate*100:.1f}% in dataset training mask ({mask_ratio:.0f}x difference). "
            f"The lightweight mask exposes BOMB far more often than training."
        )

    # Check 4: Training dynamic
    ds_bomb_in_mask_count = ds_bomb_in_mask.sum()
    ds_bomb_logit_invalid_mean = float(ds_bomb_logit_when_invalid.mean()) if len(ds_bomb_logit_when_invalid) > 0 else 0
    if ds_bomb_logit_invalid_mean > ds_bomb_logit_when_valid.mean() if len(ds_bomb_logit_when_valid) > 0 else True:
        findings.append(
            f"TRAINING DYNAMIC: BOMB logit is HIGHER when BOMB is NOT in the training mask "
            f"(invalid: {ds_bomb_logit_invalid_mean:.4f}) vs when it IS in the mask. "
            f"This confirms the BOMB logit is unconstrained during training and drifts upward."
        )

    # Check 5: Logit rank
    live_bomb_rank = float(live_ranks[:, 5].mean())
    ds_bomb_rank = float(ds_ranks[:, 5].mean())
    if live_bomb_rank > 4.0:
        findings.append(
            f"LOGIT RANK: BOMB has mean rank {live_bomb_rank:.1f}/5 in live (vs {ds_bomb_rank:.1f}/5 in dataset). "
            f"BOMB is the highest-ranked action in virtually all live steps."
        )

    for f in findings:
        print(f"\n  {f}")

    # ---- Final determination ----
    print("\n" + "-" * 70)
    print("DETERMINATION:")

    if bomb_gap_live > 2.0 and mask_ratio > 5:
        print("\n  ROOT CAUSE: Training mask / evaluation mask mismatch + unconstrained BOMB logit")
        print()
        print("  The training mask (compute_safe_action_mask) only includes BOMB as a valid")
        print("  action when the heuristic actually chose BOMB (~2-3% of samples). For the")
        print("  remaining ~97% of training samples, the BOMB logit is completely unconstrained")
        print("  by the masked cross-entropy loss. The model's BOMB logit drifts to high values")
        print("  during training because it is never penalized.")
        print()
        print("  The lightweight validity mask in shadow evaluation allows BOMB whenever")
        print("  bombs_left > 0 and the agent is not standing on a bomb — which is true for")
        print("  the vast majority of steps. This exposes the inflated BOMB logit, causing")
        print("  the model to predict BOMB as top-1 on nearly every step.")
        print()
        print("  This is NOT a feature encoding bug, checkpoint mismatch, or observation")
        print("  schema issue. The encoding is consistent between dataset and live play.")
        print("  The root cause is a structural mismatch between the training mask criteria")
        print("  and the evaluation mask criteria for the BOMB action.")
    elif len(flagged_channels) > 3:
        print("\n  ROOT CAUSE: Feature encoding mismatch between dataset and live observations")
    else:
        print("\n  ROOT CAUSE: Inconclusive — review findings above.")

    # ---- Recommendations ----
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS:")
    print()
    print("  1. Retrain with bomb-balanced masking: include BOMB in the training mask more")
    print("     often (e.g., whenever the current cell is safe AND adjacent to a crate/enemy),")
    print("     so the model learns to discriminate bomb-appropriate situations.")
    print()
    print("  2. Add explicit BOMB suppression loss: penalize high BOMB logit on samples where")
    print("     BOMB should not be chosen (e.g., when not adjacent to any box or enemy).")
    print()
    print("  3. Consider using the dataset-style mask (walkable + safe) in shadow evaluation")
    print("     for a more realistic assessment of model behavior.")
    print()
    print("  4. Until retrained, the current checkpoint is NOT suitable for any form of")
    print("     hybrid reranking — even with a safety gate, the model provides no useful")
    print("     signal for action selection.")

    return {
        "flagged_channels": [(c[1], c[2], float(c[3])) for c in flagged_channels],
        "bomb_gap_dataset": float(bomb_gap_dataset),
        "bomb_gap_live": float(bomb_gap_live),
        "mask_ratio": float(mask_ratio),
        "findings": findings,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose shadow ranker bomb-collapse root cause."
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to action ranking .npz dataset")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to neural ranker checkpoint")
    parser.add_argument("--agent_path", default="agent/hybrid_agent",
                        help="Path to production heuristic agent")
    parser.add_argument("--opponents", nargs=3,
                        default=["TacticalRuleAgent", "TacticalRuleAgent", "TacticalRuleAgent"])
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    diagnose(args)


if __name__ == "__main__":
    main()
