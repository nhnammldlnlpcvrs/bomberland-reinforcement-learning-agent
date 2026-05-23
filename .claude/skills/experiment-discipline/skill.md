# Experiment Discipline Skill

Use this skill for Bomberland optimization tasks.

Rules:

- Create variants; never overwrite the champion first.
- Compare multiple runs, not one lucky run.
- Record meaningful experiments in `docs/EXPERIMENT_LOG.md`.
- Reject local-only improvements if they look online-risky.
- Do not commit poor variants.
- Commit accepted tools, docs, and controlled experiment variants only.

Acceptance signals:

- score improves or draw falls without average-rank damage
- average steps do not drift toward 500
- replay analyzer does not show stuck loops
- win rate does not collapse
- safety invariants remain intact

