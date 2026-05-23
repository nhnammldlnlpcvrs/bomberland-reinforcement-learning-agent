# RL Agent Design Skill

Use this skill when changing Bomberland agent strategy or planning learned policies.

Guidelines:

- Prefer hybrid safety-filtered agents.
- Do not deploy a raw neural policy.
- Use reward, value, Q-value, and advantage concepts carefully.
- Optimize online robustness over local score.
- Action masking is mandatory for learned policies.
- `PLACE_BOMB` must still pass escape validation.
- Reject changes that improve one local run but increase online-risk signals.

Default design:

1. Enumerate valid safe actions.
2. Score long-term territory and pressure.
3. Let ML rank safe actions only.
4. Veto unsafe actions with deterministic rules.
5. Fallback to the heuristic agent.

