# Bomberland Pure RL Repository Analysis

Date: 2026-05-31

This report summarizes the current research state and the recommended pure-RL
training path. It is research-only. It does not promote or modify production.

## Scope And Production Constraints

Protected paths for production remain out of scope:

- `submission/agent.py`
- `agent/hybrid_agent_online_robust/`
- `engine/`
- `competition/`

Current working tree note: `git status` shows `submission/agent.py` deleted
before this report work. This training track did not modify, repair, or promote
that path. Manual cleanup is required before any production release.

## Repository State

Important RL and research agents:

- `agent/rl_agent_pure/`: single-frame PPO research agent with minimal illegal
  action masking.
- `agent/rl_agent_temporal/`: frame-stack PPO research agent with minimal
  illegal action masking. This is the cleanest pure-RL deployment package.
- `agent/rl_agent_recurrent/`: RecurrentPPO and standalone recurrent BC
  experiments.
- `agent/rl_agent_recurrent_modular/`: modular recurrent BC inference branch.
  Not a pure-RL final solution.
- `agent/rl_strong/`: frame-stack experimental export, but its inference mask
  includes danger and bomb-usefulness filtering beyond minimal legality, so it
  should not be treated as the clean pure-RL final target.
- `agent/hybrid_rule_rl/`, `agent/hybrid_agent_rl/`, `hybrid_ppo/`: hybrid or
  assisted policies, useful for baselines/teachers but not pure-RL final agents.

Important training/evaluation code:

- `ml/train_sb3_ppo.py`: main SB3 PPO training path with frame-stack support.
- `ml/evaluate_rl_pure.py`: main normal-env evaluation metrics.
- `ml/train_recurrent_ppo.py`, `ml/evaluate_recurrent_rl.py`: RecurrentPPO.
- `ml/train_curriculum_ppo.py`, `ml/run_curriculum_pipeline.py`: curriculum PPO.
- `ml/train_bomberland_strong/`: new Kaggle-oriented pure-RL training track with
  chunked training, evaluation, checkpoint gates, curated self-play pool, and
  stage configs.

Important datasets/artifacts:

- BC and bomb datasets under `ml/datasets/rl_bc_*`, `recurrent_bc_*`,
  `bomb_value_*`, `bomb_outcome_*`.
- Auxiliary/curriculum datasets under `ml/datasets/curriculum_*` and
  `aux_*`.
- Checkpoints under `ml/checkpoints/rl_agent_pure`,
  `ml/checkpoints/rl_agent_temporal`, and `ml/checkpoints/rl_agent_recurrent`.

## Best Evidence From Logs

### Strongest Survival RL Policy

Checkpoint:

```text
ml/checkpoints/rl_agent_temporal/ppo_framestack4_lr1e4_ent002_100k.zip
```

Evidence:

- `logs/rl_temporal_framestack4_lr1e4_ent002_100k_seed9300.json`
- `logs/rl_temporal_framestack4_lr1e4_ent002_100k_seed9301.json`

Metrics, 100 episodes:

- Random: win `98%`, death `2%`, crash/timeout/invalid `0`
- Simple: win `50%`, draw `14%`, death `36%`, crash/timeout/invalid `0`
- Bomb rate: `0`

Conclusion: this is the strongest clean pure-RL survival base.

### Single-Frame Pure PPO Baseline

Log:

- `logs/rl_singleframe_baseline_seed9300.json`

Metrics, 100 episodes:

- Random: win `98%`, death `2%`
- Simple: win `33%`, death `58%`
- Bomb rate tiny, with poor bomb safety: `bomb_suicide_rate 75%`,
  `boxes/bomb 0.4`

Conclusion: weaker than frame-stack for gameplay.

### Strongest Bombing Behavior

Recurrent PPO from scratch learns bombs, but loses the game.

Logs:

- `logs/rl_recurrent_stage1_seed9700.json`
- `logs/rl_recurrent_fromscratch_50k_seed9800.json`

Representative metrics:

- Recurrent stage1 vs simple: win `0%`, death `100%`, bomb rate `2.39%`,
  `boxes/bomb 3.14`, bomb suicide `54.3%`
- Recurrent 50k vs simple: win `1%`, death `99%`, bomb rate `2.38%`,
  `boxes/bomb 2.91`, bomb suicide `45.2%`

Conclusion: recurrent exploration discovers tactical bombing but survival is
catastrophic.

### Frame-Stack Curriculum / BC Attempts

Logs:

- `logs/rl_temporal_fs4_bomb_box_25k_seed9500.json`
- `logs/rl_temporal_fs4_bomb_then_escape_25k_seed9500.json`
- `logs/rl_temporal_fs4_bc001_seed9600.json` and variants

Result:

- Survival is mostly preserved.
- Bomb rate remains `0`.

Conclusion: curriculum/BC did not restore a bomb prior once the frame-stack
policy collapsed to no-bomb survival.

### Modular Recurrent BC

Logs:

- `logs/modular_recurrent_eval_seed9800.json`
- `logs/modular_value_aware_eval_seed9900.json`
- `logs/modular_value_now_v1_onpolicy_offline_sweep.json`
- `logs/modular_value_now_v2_onpolicy_offline_sweep.json`
- `logs/bomb_outcome_decision_sweep.json`

Findings:

- Bomb safety head separates safe-escape from unsafe contexts offline.
- Gameplay over-bombs: threshold `0.50` vs simple gives win `43%`, death `51%`,
  bomb rate `6.77%`, zero-value bombs `1068`.
- Value-aware variants reduce zero-value bombs only slightly and do not improve
  win/death.
- Multi-outcome bomb model fails offline gate. Best useful precision/recall
  tradeoffs still exceed death-risk constraints.

Conclusion: modular heads are useful diagnostics but not a gameplay solution.

### Auxiliary Critic / Representation

Log:

- `logs/value_quality_report.json`

Finding:

- Actor drift after aux critic pretrain: `0`.
- Value quality did not materially improve over baseline; explained variance
  remains near zero on important subsets.

Conclusion: stop aux-critic integration for now.

### Curriculum PPO

Logs:

- `logs/curriculum_pipeline_report.json`
- `logs/curriculum_mixed_retain_smoke_report.json`

Finding:

- Direct actor fine-tuning on curriculum caused normal-env survival regression.
- Full-game retain/KL reduced but did not eliminate regression.

Conclusion: curriculum is useful for data generation, not direct actor PPO yet.

## What Worked

- Frame-stack PPO improved survival and simple win/death substantially over
  single-frame PPO.
- Minimal legality-mask frame-stack packaging in `agent/rl_agent_temporal/`
  stays compatible with pure-RL constraints.
- Recurrent PPO proves that memory increases bomb exploration, but needs a much
  stronger survival curriculum/league before it is viable.
- Checkpoint-gated workflows correctly reject regressions.

## What Failed Or Should Stop

- Direct PLACE_BOMB logit nudging.
- Selector/value-head distillation into the actor.
- Modular recurrent inference as final policy.
- Recurrent PPO from scratch without a survival curriculum.
- Aux critic warm-start as currently formulated.
- Short curriculum actor fine-tuning on narrow scenarios.
- Evaluating by offline classification metrics instead of gameplay.

## Recommended Pure-RL Strategy

Continue from the frame-stack4 survival checkpoint with real PPO and strict
multi-seed gameplay gates.

Do not try to force bombs immediately. The current leaderboard-relevant signal is
survival and win rate. Bombing should be allowed to emerge only if it improves
gameplay under checkpoint gates.

Use this training track:

```text
ml/train_bomberland_strong/
```

Stage progression:

1. Stage 1: frame-stack4 PPO continuation vs random + simple.
   Objective: improve simple win/death while maintaining random win >= 95%.
2. Stage 2: add `online_robust` and available stronger baselines.
   Objective: robustness, not bomb frequency.
3. Stage 3: curated self-play with only accepted checkpoints.
   Objective: leaderboard-style robustness.

## Launch Commands

Kaggle Stage 1:

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

Evaluation:

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

Resume latest:

```bash
SAVE_DIR=/kaggle/working/bomberland_strong_stage1 \
TOTAL_TIMESTEPS=500000 \
bash ml/train_bomberland_strong/scripts/resume_latest.sh
```

## Checkpoint Selection

Accept only gameplay improvements:

- crash/timeout/invalid = `0`
- random win >= `95%`
- simple death does not regress badly
- simple win competitive or improved
- if bomb rate > 0, boxes/bomb > `0.5` and bomb suicide not excessive
- if bomb rate = 0, simple win/death must improve meaningfully
- multi-seed evaluation before `best_overall`

Best checkpoint path is currently still:

```text
ml/checkpoints/rl_agent_temporal/ppo_framestack4_lr1e4_ent002_100k.zip
```

No new long-run checkpoint has surpassed it yet.

## Promotion Recommendation

Do not promote anything now.

Continue pure-RL Kaggle training from the frame-stack4 survival checkpoint using
`ml/train_bomberland_strong`. Promote manually only after a candidate beats the
current strongest production/online_robust baseline in 300+ episode multi-seed
benchmarks with zero crash/timeout/invalid and no death-rate regression.
