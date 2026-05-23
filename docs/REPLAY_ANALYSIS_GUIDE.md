# Replay Analysis Guide

## 1. Why Replay Analysis Matters

Aggregate scores hide failure modes. Replay analysis shows whether the agent loses to self-bombs, enemy traps, passive loops, poor expansion, or bad late-game conversion.

## 2. How to Generate Local Logs

```bash
python -m scripts.participant.run_local_match --agent_paths agent/hybrid_agent TacticalRuleAgent --num_episodes 5 --save_logs true
```

Logs are written to `logs/json/`.

## 3. How to Analyze Logs

```bash
python -m scripts.participant.analyze_matches --log_dir logs/json --team_name HybridAgent
```

## 4. What to Look For

- Early suicide
- Self bomb
- Enemy bomb
- Stuck loop
- Spawn camping
- Corridor trap
- Draw near 500
- Low bomb usage
- Passive movement

## 5. Mapping Failure to Fix

| Failure | Candidate Fix |
|---|---|
| stuck loop | loop breaker |
| average steps high | controlled expansion |
| self bomb | strengthen safety layer |
| low win rate | territory pressure or enemy pressure |
| online collapse | reduce aggression and re-check state reset |

