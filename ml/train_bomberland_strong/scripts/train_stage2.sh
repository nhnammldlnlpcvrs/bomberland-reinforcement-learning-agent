#!/usr/bin/env bash
set -euo pipefail

python -m ml.train_bomberland_strong.run_kaggle_train \
  --stage stage2 \
  --resume "${RESUME:-/kaggle/working/bomberland_strong_stage1/best_overall.zip}" \
  --total_timesteps 1000000 \
  --n_envs 4 \
  --frame_stack 4 \
  --device auto \
  --save_dir "${SAVE_DIR:-/kaggle/working/bomberland_strong_stage2}" \
  --eval_opponents random simple online_robust
