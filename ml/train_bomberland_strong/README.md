# Bomberland Strong RL Training Track

This folder is a clean research-only training path for serious Kaggle RL runs.
It does not modify production submission code and does not promote checkpoints
automatically.

## Why This Track Exists

Previous experiments covered pure PPO, frame-stack PPO, RecurrentPPO, BC
warm-starts, modular recurrent BC, bomb selectors, bomb value heads, outcome
heads, auxiliary critic pretraining, and curriculum training.

Main findings:

- Frame-stack4 PPO is currently the strongest survival-oriented RL baseline.
- RecurrentPPO from scratch learns nonzero bombing, but survival is catastrophic.
- Modular bomb/value heads work offline but have not transferred reliably to
  gameplay.
- Direct bomb-logit nudging, selector distillation, tiny BC loops, and modular
  inference thresholds should not be continued as the main leaderboard path.

The current best research starting checkpoint is:

```text
ml/checkpoints/rl_agent_temporal/ppo_framestack4_lr1e4_ent002_100k.zip
```

It has strong movement/survival, high win rate versus random, competitive simple
baseline performance, and `bomb_rate = 0`. This track starts there and uses real
RL training with strict checkpoint gates rather than more offline logit edits.

## Kaggle Stage 1 Command

```bash
python -m ml.train_bomberland_strong.run_kaggle_train \
  --stage stage1 \
  --resume ml/checkpoints/rl_agent_temporal/ppo_framestack4_lr1e4_ent002_100k.zip \
  --total_timesteps 1000000 \
  --n_envs 4 \
  --frame_stack 4 \
  --device auto \
  --save_dir /kaggle/working/bomberland_strong_stage1
```

## Evaluation Command

```bash
python -m ml.train_bomberland_strong.evaluate \
  --agent_path /kaggle/working/bomberland_strong_stage1/best_overall.zip \
  --frame_stack 4 \
  --opponents random simple online_robust \
  --episodes 300 \
  --max_steps 500 \
  --seed 2026 \
  --output /kaggle/working/eval_best_overall.json
```

## Checkpoint Selection

Use `checkpoint_gate.py`; do not select by reward alone. A candidate must have:

- `crash/timeout/invalid = 0`
- random win rate at least 95%
- simple death not regressed badly
- simple win competitive or improved
- if bombing occurs, `boxes/bomb > 0.5` and bomb suicide is not excessive
- if no bombing occurs, survival/simple win must improve meaningfully
- multi-seed confirmation before `best_overall`

## What Not To Promote

Do not promote:

- recurrent-from-scratch checkpoints with high death
- modular/selector/value-head inference checkpoints
- checkpoints selected only by training reward
- any checkpoint without 300+ episode benchmark against random, simple, and
  stronger baselines

Promotion to `submission/agent.py` is a manual decision outside this folder.
