# Online vs Local Lessons

## 1. Why Local Score Can Mislead

- Opponent pool mismatch: local baselines do not represent hidden leaderboard agents.
- Hidden leaderboard agents may punish passive loops or unsafe pressure differently.
- Local overfitting can reward behavior that does not transfer online.
- Execution semantics can differ, including reused agent instances and state persistence.
- A state persistence bug can push an agent into the wrong internal phase across games.

## 2. Case Study

The online-best conservative version performed better online than later local-strong variants. The less-draw variant produced strong local scores, but online it had lower win rate and worse average rank. Average steps and average rank were more informative than local score alone.

Approximate history:

- Online-best v1: score about 114.1, win rate about 35%, average rank about 0.95, average steps about 408.
- less_draw: score about 112.6, win rate about 22%, average rank about 1.16, average steps about 469.
- online_robust: local multi-run comparison improved score and draw rate, but still needs online validation.

## 3. Metrics to Track

- Score
- Win Rate
- Draw Rate
- Avg Rank
- Avg Steps
- Games
- Sigma

## 4. Acceptance Rules

- Accept only if behavior improves, not just local score.
- Reject if average steps drift near 500 while win rate is low.
- Reject if average rank worsens.
- Reject if replay analysis reports stuck-loop behavior.
- Compare against online-best, not only against local benchmark history.

## 5. Submission Strategy

Submit controlled variants. Wait for at least 50 online games before drawing strong conclusions. Compare new variants against the online-best version and record results in `docs/EXPERIMENT_LOG.md`.

