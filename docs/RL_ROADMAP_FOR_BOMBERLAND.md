# RL Roadmap for Bomberland

## Phase 1: Strong Heuristic Agent

Build reliable safety, BFS movement, bomb rules, danger maps, and territory heuristics. This is the production baseline and the teacher for future learning.

## Phase 2: Evaluation Infrastructure

Maintain stability benchmark, compare agents, replay analyzer, local logs, and online metrics. Do not trust one run.

## Phase 3: Dataset Collection

Collect observation-action pairs from the strong heuristic. Collect successful replays, label actions, and filter low-quality decisions such as emergency mistakes or known loops.

## Phase 4: Imitation Learning

Train a small supervised policy.

- Input: multi-channel 13x13 board tensor.
- Output: 6 action logits.
- Teacher: heuristic agent.
- Deployment rule: never deploy without a rule-based safety filter.

## Phase 5: Tiny DQN / Q-head

Learn `Q(s,a)` using handcrafted features or a small CNN encoder. Keep rule-based action filtering. The Q-head should improve action ordering, not replace safety.

## Phase 6: PPO / Actor-Critic

Use PPO only after the simulator, dataset, logging, and evaluation are stable. Use GAE, clipped updates, self-play curriculum, and a safety action mask.

## Phase 7: Hybrid Deployment

The neural policy suggests an action. The rule-based safety layer vetoes unsafe actions. If the neural policy is invalid, fallback to the heuristic.

## Why PPO Is Not First

PPO is sensitive to reward shaping, simulator variance, and online meta mismatch. A raw PPO policy can learn unsafe shortcuts. Start with heuristic and imitation learning so the model inherits stable behavior.

## Why Online Robustness Matters

Online opponents are not the same as local baselines. Robustness means stable average rank, lower draw without hyper-aggression, and reasonable average steps.

## Why Action Filtering Is Mandatory

Bomberland has hard-death actions. A learned policy must not be allowed to choose immediate death, unsafe bomb placement, or invalid movement.

## Dataset Collection Pipeline

The first ML pipeline is replay logs to imitation dataset. The teacher is the current online-robust heuristic agent, because it already encodes safety, expansion, bomb escape checks, and online robustness lessons.

Dataset flow:

1. Generate replay logs with `run_local_match --save_logs true`.
2. Parse `logs/json/*.json`.
3. Extract `obs -> action` pairs for `HybridAgent`.
4. Encode each frame into a compact 13x13 multi-channel tensor.
5. Save compressed `.npz` files for future supervised imitation learning.

PPO is still not the next step. Before PPO, collect cleaner datasets, train a small CNN policy to imitate the heuristic, and evaluate it behind the deterministic safety filter. The future CNN policy should rank safe actions; it should not own safety decisions.

## Imitation Learning Baseline

The first learned baseline is supervised behavior cloning from the online-robust heuristic teacher. The model uses the replay dataset tensors and predicts one of the six Bomberland actions with cross-entropy loss.

This comes before PPO because it verifies that replay parsing, feature encoding, action labels, and class balance are usable. A tiny CNN can expose dataset problems quickly without introducing reward-design instability or self-play variance.

Future deployment should stay hybrid: the neural policy ranks candidate actions, then the deterministic safety layer applies action masks, bomb escape validation, and fallback to the heuristic agent.

## Dataset Curation and Policy Bias

Teacher policies have bias. A strong survival heuristic can still produce replay datasets that are draw-heavy, STOP-heavy, or sparse in meaningful `PLACE_BOMB` examples. If copied directly, the neural policy can learn passive survival instead of useful action ranking.

Before PPO or DQN work, curate the imitation dataset and measure policy behavior. Track action distribution, win/draw/loss mix, bomb ratio, STOP ratio, prediction entropy, movement diversity, and confusion around `PLACE_BOMB`.

Curated datasets should improve training signal without forcing aggression. The goal is not bomb spam; it is enough meaningful bomb representation for the supervised model to learn when bomb actions are plausible. Policy entropy matters because low-entropy collapse can hide behind acceptable accuracy.
