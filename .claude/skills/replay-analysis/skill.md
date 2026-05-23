# Replay Analysis Skill

Use this skill before changing heuristics or interpreting leaderboard failures.

Workflow:

1. Generate logs with `run_local_match --save_logs true`.
2. Run `python -m scripts.participant.analyze_matches --log_dir logs/json --team_name HybridAgent`.
3. Inspect wins, draws, losses, average steps, death source, and stuck-loop signals.
4. Map symptoms to fixes before editing the agent.

Look for:

- self-bomb deaths
- enemy-bomb deaths
- corridor/dead-end deaths
- spawn camping
- passive movement
- draw near 500 steps
- low bomb usage
- repeated local positions

Do not trust aggregate score without replay evidence.

