from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SOURCE_NAMES = np.asarray(
    [
        "safe_value_positive",
        "synthetic_safe_zero_negative",
        "synthetic_unsafe_negative",
        "onpolicy_zero_value_negative",
        "onpolicy_death_negative",
        "onpolicy_useful_positive",
        "onpolicy_safe_nonbomb_negative",
        "onpolicy_potential_bomb_avoided_negative",
    ],
    dtype=object,
)


def add(parts, observations, valid_mask, labels, source, expected_boxes, zero_reason=None):
    count = len(labels)
    if count == 0:
        return
    parts["observations"].append(observations.astype(np.float32))
    parts["valid_mask"].append(valid_mask.astype(bool))
    parts["label_value_now"].append(np.asarray(labels, dtype=np.float32))
    parts["source_type"].append(np.full(count, source, dtype=np.int16))
    parts["expected_boxes"].append(np.asarray(expected_boxes, dtype=np.float32))
    parts["zero_value_reason"].append(
        np.asarray(zero_reason if zero_reason is not None else np.zeros(count), dtype=np.int16)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge synthetic and on-policy bomb_value_now data.")
    parser.add_argument("--base_dataset", default="ml/datasets/bomb_value_now_dataset.npz")
    parser.add_argument("--onpolicy_dataset", default="ml/datasets/modular_onpolicy_bomb_data.npz")
    parser.add_argument("--output", default="ml/datasets/bomb_value_now_dataset_v2.npz")
    args = parser.parse_args()

    parts = {k: [] for k in ["observations", "valid_mask", "label_value_now", "source_type", "expected_boxes", "zero_value_reason"]}
    base = np.load(args.base_dataset, allow_pickle=True)
    base_source = base["source_type"].astype(np.int16)
    base_labels = base["label_value_now"].astype(np.float32)
    base_boxes = base["expected_boxes"].astype(np.float32)
    add(parts, base["observations"][base_source == 0], base["valid_mask"][base_source == 0], base_labels[base_source == 0], 0, base_boxes[base_source == 0])
    add(parts, base["observations"][base_source == 1], base["valid_mask"][base_source == 1], base_labels[base_source == 1], 1, base_boxes[base_source == 1], np.ones(np.sum(base_source == 1)))
    add(parts, base["observations"][base_source == 2], base["valid_mask"][base_source == 2], base_labels[base_source == 2], 2, base_boxes[base_source == 2], np.full(np.sum(base_source == 2), 2))

    on = np.load(args.onpolicy_dataset, allow_pickle=True)
    on_source = on["source_type"].astype(np.int16)
    on_boxes = on["boxes_destroyed"].astype(np.float32)
    mapping = {
        0: 3,
        1: 4,
        2: 5,
        3: 6,
        4: 7,
    }
    for old_source, new_source in mapping.items():
        idx = on_source == old_source
        labels = (new_source == 5) * np.ones(np.sum(idx), dtype=np.float32)
        add(parts, on["observations"][idx], on["valid_mask"][idx], labels, new_source, on_boxes[idx], np.full(np.sum(idx), new_source))

    arrays = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, source_type_names=SOURCE_NAMES, **arrays)
    source = arrays["source_type"]
    labels = arrays["label_value_now"]
    report = {
        "output": str(output),
        "total_count": int(len(labels)),
        "positive_count": int(np.sum(labels > 0.5)),
        "synthetic_negative_count": int(np.sum(np.isin(source, [1, 2]))),
        "onpolicy_zero_value_negative_count": int(np.sum(source == 3)),
        "onpolicy_death_negative_count": int(np.sum(source == 4)),
        "onpolicy_useful_positive_count": int(np.sum(source == 5)),
        "onpolicy_safe_nonbomb_count": int(np.sum(source == 6)),
        "onpolicy_potential_bomb_avoided_count": int(np.sum(source == 7)),
        "positive_fraction": float(np.mean(labels > 0.5)) if len(labels) else 0.0,
        "source_distribution": {str(SOURCE_NAMES[i]): int(np.sum(source == i)) for i in range(len(SOURCE_NAMES))},
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
