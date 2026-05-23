# ML/DL/RL Glossary for Bomberland

## MDP

General: a Markov Decision Process defines states, actions, transitions, and rewards.

Bomberland: state is the board observation, action is one of six moves, transition is the engine step, reward is match outcome or shaped signal.

## State

General: the information used for decision making.

Bomberland: `map`, `players`, `bombs`, danger map, and derived features.

## Action

General: a choice available to the agent.

Bomberland: STOP, LEFT, RIGHT, UP, DOWN, PLACE_BOMB.

## Reward

General: immediate feedback for an action.

Bomberland: win, kill, survival, safe expansion, or penalties for death.

## Return

General: discounted sum of future rewards.

Bomberland: the long-term value of surviving, gaining territory, and winning.

## Policy

General: a mapping from state to action probabilities.

Bomberland: a model or heuristic that chooses among six actions.

## Value Function

General: expected return from a state.

Bomberland: how favorable a board position is.

## Q-function

General: expected return from taking action `a` in state `s`.

Bomberland: value of moving into a cell or placing a bomb now.

## Advantage

General: action value minus baseline value.

Bomberland: how much better a move is than the average valid move.

## Monte Carlo

General: learns from complete returns after episodes.

Bomberland: update from full match results.

## TD(0)

General: bootstraps one step into the future.

Bomberland: update value from current state, reward, and next-state value.

## TD(lambda)

General: mixes multi-step returns with decay.

Bomberland: credits decisions across several future bomb and movement steps.

## GAE

General: Generalized Advantage Estimation for actor-critic training.

Bomberland: smoother advantage estimates for PPO after stable rewards exist.

## Q-learning

General: off-policy learning of `Q(s,a)`.

Bomberland: learn which safe action has best long-term outcome.

## DQN

General: neural Q-learning with replay buffer.

Bomberland: CNN maps board to six Q-values, filtered by safety.

## REINFORCE

General: policy-gradient method using sampled returns.

Bomberland: direct policy update from match outcomes, usually high variance.

## PPO

General: clipped policy-gradient actor-critic algorithm.

Bomberland: useful later with action masks and stable curriculum.

## AWR

General: Advantage-Weighted Regression for offline policy improvement.

Bomberland: can learn from heuristic replay datasets.

## Imitation Learning

General: supervised learning from expert actions.

Bomberland: train a small policy to copy the heuristic agent.

## Self-play

General: train against versions of yourself.

Bomberland: expose policy to stronger trap and anti-camping behaviors.

## Curriculum Learning

General: gradually increase task difficulty.

Bomberland: start vs baselines, then mixed agents, then self-play.

## Action Mask

General: disallow invalid actions before sampling.

Bomberland: mask unsafe moves and non-escapable bombs.

## Safety Filter

General: deterministic layer that vetoes bad model choices.

Bomberland: danger map and bomb escape checks override neural policy.

## CNN Encoder

General: convolutional network for spatial features.

Bomberland: encode 13x13 board channels.

## LSTM

General: recurrent network for temporal memory.

Bomberland: may track enemy patterns, but increases complexity.

## Replay Buffer

General: stored transitions for training.

Bomberland: saved obs-action-reward-next_obs samples from matches.

