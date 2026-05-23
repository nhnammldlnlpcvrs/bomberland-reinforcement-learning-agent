# ML Skeleton

This folder contains safe placeholders for future ML/DL/RL work.

Current status:

- No model training is implemented.
- No extra dependencies are required.
- Scripts are argparse skeletons only.
- Production agents must keep a rule-based safety filter.

Suggested future flow:

1. Build datasets from replay JSON.
2. Encode observations with `features.py`.
3. Train imitation learning from heuristic labels.
4. Evaluate with action masks.
5. Export a small CPU-friendly model only after validation.

