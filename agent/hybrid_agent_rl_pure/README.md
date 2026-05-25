# hybrid_agent_rl_pure

Experimental pure RL-style Bomberland track. This folder is isolated from
`hybrid_agent_online_robust` and should not be promoted unless benchmarks show
a clear improvement.

## Algorithm

- Compact Dueling Double-DQN.
- CPU inference with a small convolutional board encoder and scalar branch.
- Replay buffer, target network, epsilon schedule, gradient clipping, and
  checkpointing are implemented in `ml/train_rl_pure.py`.
- A hard safety layer masks illegal and immediately lethal actions. The model
  still ranks actions through Q-values; the mask only rejects invalid/suicidal
  choices.

## Observation Features

The encoder emits `18 x 13 x 13` spatial features:

- wall, box, grass, radius item, capacity item
- own position
- alive enemy positions
- bomb occupancy, normalized timer, own bomb
- danger horizon channels for explosion risk in 1 through 7 steps
- BFS safe reachable cells

Scalar features include bombs left, radius bonus, alive enemy ratio, normalized
step, current danger timer, reachable-safe area, remaining box density, and bomb
density.

## Reward

Reward weights live in `config.json`. The trainer rewards wins, better final
rank, enemy deaths, box destruction, item pickup, useful bomb placement, and
non-camping survival. It penalizes death, standing in danger, blocked moves,
unsafe bomb attempts, and short position loops.

## Training

```powershell
python -m ml.train_rl_pure --stage a --episodes 200 --max_steps 300 --seed 7
python -m ml.train_rl_pure --stage b --episodes 500 --checkpoint ml/checkpoints/rl_pure/latest.pth
python -m ml.train_rl_pure --stage c --episodes 1000 --self_play_pool ml/checkpoints/rl_pure
```

## Evaluation

```powershell
python -m ml.eval_rl_pure --episodes 100 --opponents random simple box_farmer smarter tactical genius online_robust
python -m ml.eval_rl_pure --episodes 100 --head_to_head_online_robust
```

## Export

```powershell
python -m ml.export_rl_pure_submission --checkpoint ml/checkpoints/rl_pure/latest.pth
```

The exporter writes `dist/rl_pure_submission.zip` with one `agent.py`,
`model.py`, `config.json`, and optional `rl_pure_model.pth`.

## Current Benchmark

Initial smoke benchmark should be treated as untrained fallback performance
until a checkpoint is produced. Do not replace `online_robust` unless
`ml.eval_rl_pure` shows a stable win/rank improvement over at least 100 matches
and inference p95 remains below 100ms.

## Known Weaknesses

- Untrained fallback is conservative and mostly survival/box oriented.
- DQN may overfit to local baseline policies without a diverse self-play pool.
- Dense reward terms approximate box destruction and self-kills from state
  deltas because the engine does not expose event info directly.
