#!/usr/bin/env bash
set -euo pipefail

python -m ml.train_bomberland_strong.run_kaggle_train \
  --stage stage1 \
  --resume ml/checkpoints/rl_agent_temporal/ppo_framestack4_lr1e4_ent002_100k.zip \
  --total_timesteps 1000000 \
  --n_envs 4 \
  --frame_stack 4 \
  --device auto \
  --save_dir "${SAVE_DIR:-/kaggle/working/bomberland_strong_stage1}"
