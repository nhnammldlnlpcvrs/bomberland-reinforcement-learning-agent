# Feature Specification: Conversion Optimization

**Feature Branch**: `conversion-optimization`

**Created**: 2026-06-10

**Status**: Draft

**Input**: Analyze why the current production Bomberland agent survives for a long time but does not finish first often enough, then design a conservative rule-first hybrid conversion candidate without modifying production.

## Context

Current production is `submission/agent.py`, representing `hybrid_agent_model_optimized`.

Observed leaderboard results after 169 games:

- Rank: 336
- Mu: 109.7095
- Sigma: 0.6903
- Score: 107.6386
- Win rate: 0.2367
- Average rank: 1.2899
- Average steps: 464.0059

The production agent's long average survival indicates that basic survival is not the primary bottleneck. The feature must investigate conversion failures: games in which production remains alive late but fails to secure rank 1.

The previous aggressive endgame/search candidate is rejected evidence and must not be revived. It produced average rank 0.145 versus production 0.110, loss rate 0.120 versus 0.085, self-bomb deaths 23 versus 14, and win rate 0.000 versus 0.015.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Diagnose Conversion Failures (Priority: P1)

As the agent developer, I need a compact, evidence-based analysis of production matches so that optimization targets are based on actual missed conversion patterns rather than speculative aggression.

**Why this priority**: No new candidate should be designed until the repository's logs and replays identify where rank-1 conversion is being lost.

**Independent Test**: Run the analysis against available production logs and verify that `docs/PRODUCTION_CONVERSION_ANALYSIS.md` contains all required aggregate metrics, documented data coverage, and representative failure categories without dumping full replay data.

**Acceptance Scenarios**:

1. **Given** production match logs or replays, **When** the analysis is run, **Then** it reports rank distribution and first-, second-, third-, and fourth-place frequencies.
2. **Given** games that reach late stages, **When** the analysis is run, **Then** it reports timeout finishes, games reaching step greater than 400, and alive-player counts after step 350.
3. **Given** production action histories, **When** conversion opportunities are analyzed, **Then** the report estimates bomb placement frequency, missed kill opportunities, and declined safe-bomb opportunities with explicit definitions.
4. **Given** incomplete or heterogeneous logs, **When** metrics cannot be measured reliably, **Then** the report marks the limitation and does not fabricate values.

---

### User Story 2 - Design a Conservative Conversion Candidate (Priority: P2)

As the agent developer, I need a new isolated candidate that acts only in low-risk conversion states so that it can improve rank-1 frequency without weakening production safety.

**Why this priority**: Leaderboard upside requires better conversion, but prior aggressive endgame logic increased losses and self-bomb deaths.

**Independent Test**: Inspect the candidate design and verify that every intervention is gated by low survival risk, cannot override emergency escape, cannot introduce unsafe bomb placement, and falls back exactly to production when uncertain.

**Acceptance Scenarios**:

1. **Given** an emergency escape state, **When** the candidate acts, **Then** it returns the same emergency action as production.
2. **Given** a state with active or near-future blast risk, low safe area, recent own-bomb exposure, or uncertain evaluation, **When** the conversion layer evaluates actions, **Then** it falls back to production.
3. **Given** multiple production-approved safe actions in a low-risk conversion state, **When** the candidate has clear evidence of improved rank-1 conversion potential, **Then** it may rerank only those safe actions.
4. **Given** a `PLACE_BOMB` action not already approved as safe and useful by production, **When** the conversion layer evaluates it, **Then** it cannot select that action.

---

### User Story 3 - Validate Without Production Risk (Priority: P3)

As the production owner, I need deterministic promotion gates so that a candidate cannot replace production based on anecdotal wins or a small favorable seed sample.

**Why this priority**: Production safety and repeatable benchmark evidence are mandatory for leaderboard optimization.

**Independent Test**: Run disabled parity and staged benchmarks, confirming that failed smoke validation prevents final validation and that the final report ends with exactly one promotion status.

**Acceptance Scenarios**:

1. **Given** conversion logic disabled, **When** at least 6000 decisions are compared with production, **Then** mismatches equal zero.
2. **Given** a candidate that worsens loss rate or self-bomb deaths in the 100-episode smoke benchmark, **When** validation completes, **Then** final 5 x 300 validation is not run and status is `REJECT_PROMOTION`.
3. **Given** a candidate that passes smoke, **When** five independent 300-episode blocks complete, **Then** only compact block summaries are persisted and the promotion gate is evaluated over all blocks.
4. **Given** any timeout, error, invalid-action regression, or evidence that conversion logic never activated, **When** the final verdict is produced, **Then** promotion is rejected.

## Edge Cases

- Production logs may not contain explicit final ranks, bomb ownership, or action histories; analysis must derive only defensible metrics and identify unavailable metrics.
- Simultaneous deaths and multiple survivors at timeout must have a documented rank/draw convention.
- Safe-bomb opportunity detection must distinguish a declined opportunity from a bomb correctly rejected because escape, value, or timing was uncertain.
- Kill-opportunity detection must account for enemy escape cells, walls, boxes, blast radius, bomb timers, and chain reactions.
- The candidate may encounter states without a `step` field; it must use a conservative fallback rather than assume an endgame phase.
- Model/checkpoint load or inference failure must return production behavior.
- Wall-clock latency variance must not create false disabled-parity mismatches; parity methodology must isolate the conversion feature from unrelated timing gates.
- Candidate interventions that improve rank 1 while increasing fourth-place outcomes must be rejected under the loss-rate gate.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The feature MUST analyze the current production logs and replay files available under repository log locations before designing candidate behavior.
- **FR-002**: The analysis MUST report rank distribution, including first-, second-, third-, and fourth-place frequency and percentage.
- **FR-003**: The analysis MUST report timeout finishes and the frequency of games reaching step greater than 400.
- **FR-004**: The analysis MUST report alive-player-count distributions after step 350, subject to available replay coverage.
- **FR-005**: The analysis MUST report production bomb placement frequency using a documented denominator.
- **FR-006**: The analysis MUST define and estimate missed kill opportunities using reproducible tactical criteria.
- **FR-007**: The analysis MUST define and estimate safe-bomb opportunities declined using production-equivalent escape and usefulness checks.
- **FR-008**: The analysis MUST be saved as `docs/PRODUCTION_CONVERSION_ANALYSIS.md` during implementation of this feature.
- **FR-009**: The implementation candidate MUST live at `agent/hybrid_agent_conversion_candidate/`.
- **FR-010**: The candidate MUST be based on current production behavior from `submission/agent.py` and MUST not modify `submission/agent.py` during experimentation.
- **FR-011**: The candidate MUST NOT modify `agent/hybrid_agent_online_robust/`.
- **FR-012**: The candidate MUST activate conversion behavior only when deterministic safety checks classify survival risk as low.
- **FR-013**: The candidate MUST NOT override emergency escape or path-to-safety behavior.
- **FR-014**: The candidate MUST NOT introduce a `PLACE_BOMB` action unless current production already marks that bomb safe and useful.
- **FR-015**: Models, if used, MAY only rerank actions already accepted as safe by production rules.
- **FR-016**: The candidate MUST fall back exactly to production on uncertainty, unavailable data, model failure, inference failure, or latency-budget risk.
- **FR-017**: The feature MUST NOT include pure RL, PPO training, or open-ended reinforcement-learning research.
- **FR-018**: The feature MUST NOT reuse or continue the rejected aggressive endgame/search strategy.
- **FR-019**: Disabled candidate mode MUST produce zero mismatches over at least 6000 production decision comparisons.
- **FR-020**: The candidate MUST complete a 100-episode smoke benchmark against current production before final validation is permitted.
- **FR-021**: The smoke benchmark MUST reject the candidate if loss rate increases, self-bomb deaths increase, timeout/error/invalid actions appear, or conversion logic does not activate.
- **FR-022**: Final validation MAY run only after smoke passes and MUST consist of five independent blocks of 300 episodes each.
- **FR-023**: Validation logs MUST contain compact summaries only and MUST NOT store complete frames, replays, or per-action runtime rows by default.
- **FR-024**: The benchmark result MUST be saved as `logs/hybrid_conversion_candidate_benchmark.json`.
- **FR-025**: The candidate report MUST be saved as `docs/HYBRID_CONVERSION_CANDIDATE_REPORT.md`.
- **FR-026**: The final candidate report MUST end with exactly one status: `PROMOTE_CANDIDATE` or `REJECT_PROMOTION`.
- **FR-027**: No submission zip may be generated as part of this feature.
- **FR-028**: No production promotion may occur without a separate explicit promotion task after final validation passes.

### Analysis Definitions

- **Rank Distribution**: Count and percentage of evaluated agent appearances ending in each final rank, using the competition rank convention for simultaneous deaths and timeouts.
- **Timeout Finish**: A match reaching its configured maximum step without a unique normal-game winner.
- **Late Game**: A match state at step greater than 350; step greater than 400 is tracked separately.
- **Missed Kill Opportunity**: A state where production had a rule-safe tactical action with a defensible high probability of reducing an enemy's escape area or causing an unavoidable blast, but selected another action. The implementation plan must specify the exact reproducible detector before coding.
- **Declined Safe-Bomb Opportunity**: A state where production's own safety, escape, and usefulness checks approve bomb placement, but production chooses a non-bomb action. This metric does not imply the bomb was strategically correct.
- **Low Survival Risk**: A state with no emergency escape, no current or near-future blast exposure, sufficient reachable safe area, and no recent own-bomb escape commitment. Exact thresholds must be specified in the implementation plan and benchmarked conservatively.

### Key Entities

- **Production Match Sample**: A match or replay used for conversion analysis, including seed, final ranks, terminal step, survival steps, player-count timeline, and action/bomb events when available.
- **Conversion Opportunity**: A reproducibly detected low-risk state in which an already-safe action may improve rank-1 potential.
- **Candidate Intervention**: A decision where the conversion layer changes the production action, including the production action, candidate action, safety gates, reason, and confidence/evidence summary.
- **Validation Block Summary**: Compact aggregate results for one 300-episode seed block, including rank distribution, wins/draws/losses, survival, bomb deaths, runtime failures, latency, and intervention counters.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `docs/PRODUCTION_CONVERSION_ANALYSIS.md` reports every required metric or explicitly marks it unavailable with a reason and data-coverage count.
- **SC-002**: Disabled parity produces exactly `0 / 6000` mismatches or better coverage with zero mismatches.
- **SC-003**: The 100-episode smoke benchmark completes with zero timeout, error, and invalid-action events for the candidate.
- **SC-004**: Final validation runs only if smoke passes all safety gates.
- **SC-005**: Across the final 5 x 300 validation, candidate average rank is lower than production average rank.
- **SC-006**: Across final validation, candidate rank-1 frequency is higher than production rank-1 frequency.
- **SC-007**: Across final validation, candidate loss rate does not exceed production loss rate.
- **SC-008**: Across final validation, candidate self-bomb deaths do not exceed production self-bomb deaths.
- **SC-009**: Candidate timeout, error, and invalid-action counts remain zero and do not regress against production.
- **SC-010**: Candidate counters prove conversion logic activated and changed actions in evaluated low-risk states.
- **SC-011**: All generated benchmark JSON artifacts remain compact summaries and avoid unbounded growth from raw replay storage.

## Non-Goals

- Training or evaluating a new pure RL/PPO policy.
- Aggressive endgame chasing, broad shallow search over unsafe actions, or model-owned bomb placement.
- Modifying `submission/agent.py` or promoting a candidate.
- Modifying `agent/hybrid_agent_online_robust/`.
- Generating a submission archive.
- Optimizing survival alone without evidence that it improves final rank conversion.

## Assumptions

- Current production behavior and its packaged model dependencies are available through `submission/agent.py` and sibling submission files.
- Existing production logs and replay files provide enough coverage for at least rank, terminal-step, and survival analysis; tactical opportunity metrics may require replay reconstruction.
- The competition engine's final-rank and timeout semantics are the source of truth.
- Existing compact benchmark infrastructure can be adapted for candidate validation without storing full episode histories.
- A separate implementation plan will choose exact low-risk and opportunity thresholds after the production conversion analysis is complete.

## Dependencies

- Current production agent package under `submission/`.
- Bomberland engine and competition evaluation utilities.
- Existing logs/replays under `logs/` and participant benchmark outputs.
- The production safety, danger-map, chain-reaction, escape, and safe-bomb checks.

## Deliverables

This specification defines the following future implementation artifacts:

- `docs/PRODUCTION_CONVERSION_ANALYSIS.md`
- `agent/hybrid_agent_conversion_candidate/`
- `docs/HYBRID_CONVERSION_CANDIDATE_REPORT.md`
- `logs/hybrid_conversion_candidate_benchmark.json`

Only this specification is created in the current task.
