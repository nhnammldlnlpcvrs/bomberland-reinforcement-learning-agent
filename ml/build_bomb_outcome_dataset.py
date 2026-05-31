from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SOURCE_NAMES = np.asarray(
    [
        "useful_positive",
        "zero_value_negative",
        "death_negative",
        "safe_nonbomb",
        "counterfactual",
    ],
    dtype=object,
)


def _append(parts, *, observations, valid_mask, expected_boxes, death, zero, survived, escape, trapped, reachable, source):
    count = len(expected_boxes)
    if count == 0:
        return
    parts["observations"].append(observations.astype(np.float32))
    parts["valid_mask"].append(valid_mask.astype(bool))
    parts["expected_boxes_destroyed"].append(np.asarray(expected_boxes, dtype=np.float32))
    parts["death_after_bomb"].append(np.asarray(death, dtype=np.float32))
    parts["zero_value"].append(np.asarray(zero, dtype=np.float32))
    parts["survived_after_bomb"].append(np.asarray(survived, dtype=np.float32))
    parts["has_escape_after_bomb"].append(np.asarray(escape, dtype=np.float32))
    parts["trapped_if_bomb"].append(np.asarray(trapped, dtype=np.float32))
    parts["reachable_delta"].append(np.asarray(reachable, dtype=np.float32))
    parts["source_type"].append(np.full(count, source, dtype=np.int16))


def add_onpolicy(parts, path: str) -> dict:
    if not path or not Path(path).exists():
        return {}
    data = np.load(path, allow_pickle=True)
    obs = data["observations"]
    mask = data["valid_mask"]
    source = data["source_type"].astype(np.int16)
    boxes = data["boxes_destroyed"].astype(np.float32)
    death = data["death_within_7"].astype(np.float32)
    zero = data["label_zero_value"].astype(np.float32)
    useful = data["label_useful_bomb"].astype(np.float32)
    survived = data["survived_after_bomb"].astype(np.float32)
    reachable = data["reachable_delta"].astype(np.float32) if "reachable_delta" in data.files else np.zeros(len(source), dtype=np.float32)

    # Existing on-policy source mapping from mine_modular_onpolicy_bomb_data.py:
    # 0 zero-value bomb, 1 death-after-bomb, 2 useful bomb, 3 safe non-bomb, 4 potential avoided.
    mappings = [
        (source == 2, 0),
        (source == 0, 1),
        (source == 1, 2),
        (source == 3, 3),
        (source == 4, 3),
    ]
    for idx, new_source in mappings:
        _append(
            parts,
            observations=obs[idx],
            valid_mask=mask[idx],
            expected_boxes=boxes[idx],
            death=death[idx],
            zero=zero[idx] if new_source != 0 else np.zeros(np.sum(idx), dtype=np.float32),
            survived=survived[idx] if new_source in {0, 1, 2} else np.ones(np.sum(idx), dtype=np.float32),
            escape=np.ones(np.sum(idx), dtype=np.float32),
            trapped=np.zeros(np.sum(idx), dtype=np.float32),
            reachable=reachable[idx],
            source=new_source,
        )
    return {
        "onpolicy_count": int(len(source)),
        "onpolicy_useful": int(np.sum(useful > 0.5)),
        "onpolicy_zero": int(np.sum(zero > 0.5)),
        "onpolicy_death": int(np.sum(death > 0.5)),
    }


def add_value_v2(parts, path: str) -> dict:
    if not path or not Path(path).exists():
        return {}
    data = np.load(path, allow_pickle=True)
    obs = data["observations"]
    mask = data["valid_mask"]
    source = data["source_type"].astype(np.int16)
    boxes = data["expected_boxes"].astype(np.float32)
    labels = data["label_value_now"].astype(np.float32)

    # v2 source mapping:
    # 0 safe positive, 1 synthetic safe zero, 2 synthetic unsafe, 3 onpolicy zero,
    # 4 onpolicy death, 5 onpolicy useful, 6/7 safe non-bomb negatives.
    mappings = [
        (np.isin(source, [0, 5]) & (labels > 0.5), 0),
        (np.isin(source, [1, 3]), 1),
        (source == 4, 2),
        (np.isin(source, [6, 7]), 3),
        (source == 2, 4),
    ]
    for idx, new_source in mappings:
        count = int(np.sum(idx))
        if not count:
            continue
        _append(
            parts,
            observations=obs[idx],
            valid_mask=mask[idx],
            expected_boxes=boxes[idx],
            death=np.ones(count, dtype=np.float32) if new_source == 2 else np.zeros(count, dtype=np.float32),
            zero=np.ones(count, dtype=np.float32) if new_source in {1, 3, 4} else np.zeros(count, dtype=np.float32),
            survived=np.zeros(count, dtype=np.float32) if new_source in {2, 4} else np.ones(count, dtype=np.float32),
            escape=np.zeros(count, dtype=np.float32) if new_source in {2, 4} else np.ones(count, dtype=np.float32),
            trapped=np.ones(count, dtype=np.float32) if new_source == 4 else np.zeros(count, dtype=np.float32),
            reachable=np.zeros(count, dtype=np.float32),
            source=new_source,
        )
    return {
        "value_v2_count": int(len(source)),
        "value_v2_positive": int(np.sum(labels > 0.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-outcome bomb dataset from on-policy and offline value data.")
    parser.add_argument("--onpolicy_dataset", default="ml/datasets/modular_onpolicy_bomb_data.npz")
    parser.add_argument("--value_dataset", default="ml/datasets/bomb_value_now_dataset_v2.npz")
    parser.add_argument("--output", default="ml/datasets/bomb_outcome_dataset.npz")
    args = parser.parse_args()

    parts = {
        "observations": [],
        "valid_mask": [],
        "expected_boxes_destroyed": [],
        "death_after_bomb": [],
        "zero_value": [],
        "survived_after_bomb": [],
        "has_escape_after_bomb": [],
        "trapped_if_bomb": [],
        "reachable_delta": [],
        "source_type": [],
    }
    input_report = {}
    input_report.update(add_onpolicy(parts, args.onpolicy_dataset))
    input_report.update(add_value_v2(parts, args.value_dataset))
    arrays = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, source_type_names=SOURCE_NAMES, **arrays)

    source = arrays["source_type"]
    boxes = arrays["expected_boxes_destroyed"]
    death = arrays["death_after_bomb"]
    zero = arrays["zero_value"]
    useful = source == 0
    report = {
        "output": str(output),
        **input_report,
        "sample_count": int(len(source)),
        "onpolicy_count": int(input_report.get("onpolicy_count", 0)),
        "useful_count": int(np.sum(useful)),
        "zero_value_count": int(np.sum(zero > 0.5)),
        "death_count": int(np.sum(death > 0.5)),
        "boxes_destroyed_mean": float(np.mean(boxes)) if len(boxes) else 0.0,
        "boxes_destroyed_mean_useful": float(np.mean(boxes[useful])) if np.any(useful) else 0.0,
        "positive_negative_overlap": {
            "useful_and_death": int(np.sum(useful & (death > 0.5))),
            "useful_and_zero": int(np.sum(useful & (zero > 0.5))),
            "zero_and_death": int(np.sum((zero > 0.5) & (death > 0.5))),
        },
        "source_distribution": {str(SOURCE_NAMES[i]): int(np.sum(source == i)) for i in range(len(SOURCE_NAMES))},
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
