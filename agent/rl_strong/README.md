# rl_strong Experimental Track

Separate leaderboard research track for frame-stacked PPO. It does not modify
the production root `agent.py`.

## Train

```bash
python -m ml.train_rl_strong --stage stage1 --frame_stack 4 --n_envs 4 --total_timesteps 1000000 --save_dir ml/checkpoints/rl_strong
```

Resume from an existing frame-stacked checkpoint:

```bash
python -m ml.train_rl_strong --stage stage2 --frame_stack 4 --resume agent/rl_strong/policy.zip --save_dir ml/checkpoints/rl_strong --total_timesteps 1000000
```

## Evaluate

```bash
python -m scripts.participant.benchmark_rl_strong --candidate agent/rl_strong --episodes 300 --opponents random simple tactical online_robust hybrid_agent_rl
```

## Export

```bash
python -m ml.export_rl_strong_submission --source agent/rl_strong --output_dir submission_rl_strong --zip_path dist/rl_strong_submission.zip
```

Promote only if repeated multi-seed benchmarks clearly beat the current
production agent with zero crashes, zero timeouts, and no regression in survival.
