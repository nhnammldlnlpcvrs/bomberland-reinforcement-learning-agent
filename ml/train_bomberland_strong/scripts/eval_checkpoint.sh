#!/usr/bin/env bash
set -euo pipefail

python -m ml.train_bomberland_strong.evaluate \
  --agent_path "${AGENT_PATH:?set AGENT_PATH}" \
  --frame_stack "${FRAME_STACK:-4}" \
  --opponents random simple online_robust \
  --episodes "${EPISODES:-300}" \
  --max_steps 500 \
  --seed "${SEED:-2026}" \
  --output "${OUTPUT:-/kaggle/working/eval_checkpoint.json}"
