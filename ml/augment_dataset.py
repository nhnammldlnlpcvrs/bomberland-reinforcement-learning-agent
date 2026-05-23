"""Symmetry augmentation for Bomberland imitation and ranking datasets."""

import argparse
import json
from pathlib import Path

import numpy as np

from ml.train_imitation import ACTION_NAMES


VALID_MODES = {"hflip", "vflip", "rot180"}


def _load_metadata(data):
    raw = data.get("metadata_json")
    if raw is None:
        return {}
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {}


def _remap_action(action, mode):
    action = int(action)
    if action in (0, 5):
        return action
    if mode == "hflip":
        return {1: 2, 2: 1, 3: 3, 4: 4}[action]
    if mode == "vflip":
        return {1: 1, 2: 2, 3: 4, 4: 3}[action]
    if mode == "rot180":
        return {1: 2, 2: 1, 3: 4, 4: 3}[action]
    raise ValueError(f"unknown augmentation mode: {mode}")


def _remap_action_array(actions, mode):
    return np.array([_remap_action(action, mode) for action in actions], dtype=np.int64)


def _augment_observations(observations, mode):
    if mode == "hflip":
        return np.flip(observations, axis=2).copy()
    if mode == "vflip":
        return np.flip(observations, axis=3).copy()
    if mode == "rot180":
        return np.flip(np.flip(observations, axis=2), axis=3).copy()
    raise ValueError(f"unknown augmentation mode: {mode}")


def _augment_masks(masks, mode):
    remapped = np.zeros_like(masks)
    for action in range(6):
        remapped[:, _remap_action(action, mode)] = masks[:, action]
    return remapped


def _pct(count, total):
    return 0.0 if total <= 0 else 100.0 * float(count) / float(total)


def _print_distribution(title, actions):
    counts = np.bincount(actions.astype(np.int64), minlength=6)
    total = max(1, len(actions))
    print(f"=== {title} ===")
    print(f"samples: {len(actions)}")
    for idx, name in enumerate(ACTION_NAMES):
        print(f"  {idx} {name}: {int(counts[idx])} ({_pct(counts[idx], total):.1f}%)")
    move_total = int(counts[1:5].sum())
    print("direction distribution:")
    for idx in range(1, 5):
        print(f"  {ACTION_NAMES[idx]}: {_pct(counts[idx], max(1, move_total)):.1f}% of moves")
    print(f"STOP ratio: {_pct(counts[0], total):.1f}%")
    print(f"PLACE_BOMB ratio: {_pct(counts[5], total):.1f}%")


def _metadata_samples(metadata, repeat_count, modes, total):
    samples = metadata.get("samples", [])
    output = []
    labels = ["identity"] + list(modes)
    for label in labels:
        for idx in range(total):
            if idx < len(samples) and isinstance(samples[idx], dict):
                item = dict(samples[idx])
            else:
                item = {"source_index": int(idx)}
            item["augmentation"] = label
            output.append(item)
    if len(output) != repeat_count * total:
        raise RuntimeError("metadata augmentation length mismatch")
    return output


def augment_dataset(args):
    modes = list(args.modes)
    invalid = [mode for mode in modes if mode not in VALID_MODES]
    if invalid:
        raise ValueError(f"invalid augmentation modes: {invalid}")

    data = np.load(args.input, allow_pickle=False)
    metadata = _load_metadata(data)
    observations = data["observations"].astype(np.float32)

    if "actions" in data:
        action_key = "actions"
    elif "target_actions" in data:
        action_key = "target_actions"
    else:
        raise RuntimeError("dataset must contain actions or target_actions")
    actions = data[action_key].astype(np.int64)

    _print_distribution("Before augmentation", actions)

    obs_parts = [observations]
    action_parts = [actions]
    mask_parts = []
    if "safe_action_masks" in data:
        mask_parts.append(data["safe_action_masks"].astype(bool))

    for mode in modes:
        obs_parts.append(_augment_observations(observations, mode))
        action_parts.append(_remap_action_array(actions, mode))
        if "safe_action_masks" in data:
            mask_parts.append(_augment_masks(data["safe_action_masks"].astype(bool), mode))

    output_data = {}
    output_data["observations"] = np.concatenate(obs_parts, axis=0).astype(np.float32)
    output_data[action_key] = np.concatenate(action_parts, axis=0).astype(np.int64)

    repeat_count = 1 + len(modes)
    for key in data.files:
        if key in {"observations", action_key, "metadata_json"}:
            continue
        if key == "safe_action_masks":
            output_data[key] = np.concatenate(mask_parts, axis=0).astype(bool)
            continue
        arr = data[key]
        if arr.shape and arr.shape[0] == len(actions):
            output_data[key] = np.concatenate([arr] * repeat_count, axis=0)
        else:
            output_data[key] = arr

    output_actions = output_data[action_key]
    _print_distribution("After augmentation", output_actions)

    metadata_out = dict(metadata)
    metadata_out["augmentation"] = {
        "source": str(args.input),
        "modes": modes,
        "repeat_count": repeat_count,
        "original_samples": int(len(actions)),
        "augmented_samples": int(len(output_actions)),
        "action_key": action_key,
        "action_remap": {
            "hflip": "LEFT<->RIGHT",
            "vflip": "UP<->DOWN",
            "rot180": "LEFT<->RIGHT and UP<->DOWN",
        },
    }
    metadata_out["samples"] = _metadata_samples(metadata, repeat_count, modes, len(actions))
    output_data["metadata_json"] = json.dumps(metadata_out, sort_keys=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output_data)
    print(f"Saved augmented dataset: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Augment Bomberland imitation/ranking datasets by symmetry.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modes", nargs="+", default=["hflip", "vflip", "rot180"])
    args = parser.parse_args()
    augment_dataset(args)


if __name__ == "__main__":
    main()
