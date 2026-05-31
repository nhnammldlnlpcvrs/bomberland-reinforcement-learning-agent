#!/usr/bin/env bash
set -euo pipefail

SAVE_DIR="${SAVE_DIR:-/kaggle/working/bomberland_strong_stage1}"

python -m ml.train_bomberland_strong.run_kaggle_train \
  --stage "${STAGE:-stage1}" \
  --resume "${SAVE_DIR}/latest.zip" \
  --total_timesteps "${TOTAL_TIMESTEPS:-500000}" \
  --n_envs "${N_ENVS:-4}" \
  --frame_stack "${FRAME_STACK:-4}" \
  --device "${DEVICE:-auto}" \
  --save_dir "${SAVE_DIR}"
