# Implementation Plan: Conversion Optimization

**Branch**: `conversion-optimization` | **Date**: 2026-06-10 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/conversion-optimization/spec.md`

**Execution Status**: Planning only. No implementation, benchmark execution, production modification, or packaging is authorized by this plan.

## Summary

Investigate why the current production agent survives to late game but converts too few matches into rank 1, then implement an isolated, conservative conversion candidate. The work begins with replay/log analysis and measurable opportunity definitions. Candidate behavior is permitted only in deterministic low-risk states and may only rerank actions already accepted as safe by production. Emergency escape, production bomb safety, and exact fallback behavior remain authoritative.

The implementation sequence is deliberately gated:

1. Analyze production conversion outcomes and publish evidence.
2. Define reproducible low-risk and conversion-opportunity detectors.
3. Create an isolated candidate copied from current production.
4. Prove disabled parity over at least 6000 decisions.
5. Run a 100-episode smoke benchmark.
6. Run five independent 300-episode blocks only if smoke passes.
7. End with `PROMOTE_CANDIDATE` or `REJECT_PROMOTION`; do not modify production or create a submission archive.

## Technical Context

**Language/Version**: Python 3.x, matching the repository and organizer runtime

**Primary Dependencies**: Python standard library, NumPy, existing Bomberland engine/evaluation utilities, current production agent package; optional existing Torch inference dependency only if the production model path is reused unchanged

**Storage**: Repository files: compact JSON benchmark summaries, Markdown analysis/report files, and replay/log input files

**Testing**: `py_compile`, deterministic disabled-parity comparison, local engine benchmarks, compact multi-seed block validation

**Target Platform**: Organizer-controlled Bomberland Python runtime with a 100ms action limit

**Project Type**: Game-playing agent with offline analysis and benchmark CLI scripts

**Performance Goals**: Candidate action latency remains below 100ms; conversion layer uses a stricter 10ms maximum budget and falls back to production before budget risk

**Constraints**: No production modification, no protected-baseline modification, no pure RL/PPO, no unsafe bomb introduction, no emergency escape override, compact logs only

**Scale/Scope**: Existing production logs/replays, 6000+ parity decisions, 100 smoke episodes, and optionally 1500 final-validation episodes in five blocks

## Constitution Check

*GATE: Must pass before analysis and be re-checked before candidate implementation and before each validation stage.*

The repository constitution file is currently an unfilled template. For this feature, the user-provided project principles and the approved feature specification are binding.

| Gate | Plan Compliance |
|---|---|
| Production safety first | `submission/agent.py` is read-only throughout this feature. |
| Protected baseline | `agent/hybrid_agent_online_robust/` is read-only. |
| Isolated candidates only | Candidate lives only in `agent/hybrid_agent_conversion_candidate/`. |
| No pure RL/PPO research | No policy training, PPO tuning, or open-ended RL work is planned. |
| Mandatory safety gates | Parity, smoke, and conditional 5 x 300 validation are explicit phase gates. |
| No unsafe model override | Any model signal can only rerank production-approved safe actions. |
| Emergency escape authority | Conversion logic is bypassed whenever production is escaping or seeking safety. |
| Bomb safety authority | `PLACE_BOMB` is disabled for conversion by default and cannot be introduced independently. |
| No package before promotion | No submission archive or production update is in scope. |
| Compact logging | Benchmark output contains summaries and counters, not full frames or per-action rows. |
| Benchmark-driven verdict | Final report ends with `PROMOTE_CANDIDATE` or `REJECT_PROMOTION`. |

**Pre-analysis gate result**: PASS.

## Project Structure

### Documentation For This Feature

```text
specs/conversion-optimization/
├── spec.md
├── plan.md
└── tasks.md                         # Created later by /speckit.tasks only

docs/
├── PRODUCTION_CONVERSION_ANALYSIS.md
└── HYBRID_CONVERSION_CANDIDATE_REPORT.md
```

### Source And Benchmark Artifacts

```text
agent/
└── hybrid_agent_conversion_candidate/
    ├── agent.py
    └── ml/                          # Copied only if required by production package compatibility

scripts/participant/
├── analyze_production_conversion.py
└── benchmark_hybrid_conversion_candidate.py

logs/
├── hybrid_conversion_candidate_benchmark.json
└── hybrid_conversion_candidate_block{1..5}.json  # Only if smoke passes
```

**Structure Decision**: Use the repository's existing single-file agent package pattern and participant benchmark scripts. The production agent is copied into a new candidate folder; common production code is not refactored because that would risk parity and protected files.

## Phase 0: Production Conversion Analysis

### 0.1 Inventory Available Evidence

Read and classify existing sources without modifying them:

- Production benchmark JSON summaries under `logs/`.
- Replay JSON files generated by participant and evaluation scripts.
- Final-validation block summaries for `hybrid_model_optimized`.
- Leaderboard metrics supplied in the feature context.
- Existing reports describing rejected endgame, safety, RL, and heuristic candidates.
- Engine rank, timeout, bomb, and chain-reaction semantics.

Record for each source:

- Number of matches or agent appearances covered.
- Whether final ranks are available.
- Whether per-step observations/actions are available.
- Whether bomb owner/timer/radius are available.
- Whether data represents current production exactly.
- Any seed overlap or duplicated matches.

Do not combine incompatible samples silently. Report each dataset's coverage and use a clearly defined primary sample.

### 0.2 Define Metrics Before Computing Them

The analysis script and report must use documented definitions:

| Metric | Definition |
|---|---|
| Rank distribution | Counts and percentages for ranks 1, 2, 3, and 4 using engine final-rank semantics. Ties are reported separately where a strict ordinal rank cannot be assigned. |
| Rank-1 frequency | Unique first-place finishes divided by evaluated production appearances. |
| Average rank | Mean engine-assigned rank over production appearances. |
| Games over step 400 | Matches with terminal step greater than 400, reported as count and percentage. |
| Timeout finish | Match reaching configured max steps without a unique normal termination winner. |
| Alive players after step 350 | Distribution of alive-player count at step 350 and, where available, subsequent late-game checkpoints. |
| Bomb placement frequency | Valid production bomb placements per 100 alive action decisions and per match. |
| Enemy kill frequency | Enemy deaths attributable to a recent production-owned blast footprint, with simultaneous/chain events marked ambiguous when ownership is not unique. |
| Self-bomb deaths | Production deaths attributable to a recent production-owned bomb blast footprint. |
| Enemy-bomb deaths | Production deaths attributable to a recent non-production bomb blast footprint when no own-bomb attribution takes precedence. |
| Safe bomb opportunity declined | Production's existing bomb checks approve safe/useful placement, but production chooses non-bomb. Count only when checks can be reconstructed exactly. |
| Missed kill opportunity | A production-approved safe action could create immediate unavoidable blast pressure or materially reduce enemy safe escape cells, but production chooses another action. Detector must be conservative and reproducible. |
| Late-game pressure opportunity missed | At step >=350 or with <=2 opponents alive, production has multiple safe actions and one improves enemy-route control without reducing own safety, but production selects another. |

### 0.3 Opportunity Detector Design

Before coding candidate behavior, define offline-only detectors:

1. Reconstruct the production danger map including chain reactions.
2. Enumerate production-valid actions using the production scoring and safety helpers.
3. Compute own reachable safe area for each action.
4. Compute enemy reachable safe cells over a short deterministic horizon.
5. Mark an opportunity only when own safe area does not decrease and no candidate cell becomes dangerous before the agent can leave.
6. Separate bomb opportunities from movement pressure opportunities.
7. Treat ambiguous ownership, missing frames, and incomplete actions as unavailable rather than positive opportunities.

The first analysis iteration must favor precision over recall. False-positive opportunities would recreate the failed aggressive endgame direction.

### 0.4 Analysis Deliverable

Produce `docs/PRODUCTION_CONVERSION_ANALYSIS.md` containing:

- Data inventory and coverage.
- Metric definitions.
- Rank and late-game outcome tables.
- Bomb and death attribution tables.
- Missed-opportunity counts and rates.
- Representative replay references, not full replay dumps.
- A ranked list of likely conversion bottlenecks.
- Explicit rejected hypotheses.
- A go/no-go recommendation for candidate behavior.

**Analysis gate**: Candidate implementation may begin only if the report identifies at least one reproducible low-risk conversion opportunity with sufficient sample coverage. Otherwise stop with `REJECT_PROMOTION` and do not create behavioral changes.

## Phase 1: Candidate Design

### 1.1 Candidate Isolation

Create `agent/hybrid_agent_conversion_candidate/` by copying current production from `submission/`.

Rules:

- Never import production by path at runtime as the fallback mechanism; preserve a local exact copy so the candidate remains package-compatible.
- Copy sibling model/helper files only when required for exact current-production behavior.
- Do not edit `submission/agent.py`.
- Do not edit `agent/hybrid_agent_online_robust/`.
- Preserve production behavior before adding the conversion hook.

### 1.2 Feature Flags

Add the following environment-controlled flags:

| Flag | Default | Purpose |
|---|---:|---|
| `HYBRID_CONVERSION_ENABLE` | `false` | Master switch. Disabled mode must be exact production parity. |
| `HYBRID_CONVERSION_START_STEP` | `350` | Earliest step for time-based conversion consideration. Low-player states may still be analyzed only if the low-risk gate passes. |
| `HYBRID_CONVERSION_LOW_RISK_ONLY` | `true` | Prohibits conversion intervention outside deterministic low-risk states. |
| `HYBRID_CONVERSION_ALLOW_BOMB` | `false` | Prevents conversion logic from changing a non-bomb production action into bomb. |
| `HYBRID_CONVERSION_MAX_LATENCY_MS` | `10` | Hard conversion-layer time budget; budget risk returns production action. |

### 1.3 Low-Risk Gate

Conversion evaluation runs only when all required checks pass:

- Agent is alive.
- Production is not in emergency escape or path-to-safety behavior.
- Current cell is not in a current or near-future blast line.
- Production-selected destination is not in a current or future blast line.
- No recent own-bomb escape commitment is active.
- Reachable safe area meets a threshold selected from analysis data.
- Candidate actions do not reduce own reachable safe area below production.
- Candidate destinations are not dead ends or cells that become dangerous before a safe exit is reachable.
- Existing model load/inference state is healthy if a model signal is consulted.
- Conversion evaluation remains inside its latency budget.

Any failed or uncertain check increments a rejection counter and returns the production action exactly.

### 1.4 Conservative Conversion Actions

The conversion layer may only rerank production-approved safe actions. Initial implementation scope is movement-only.

Allowed evidence, subject to Phase 0 findings:

- Reduced distance to a vulnerable enemy without reducing own safe area.
- Occupying a safe route-control cell that reduces enemy reachable safe cells.
- Moving toward a production-approved safe bomb position for a future turn, without placing a bomb immediately.
- Selecting an item/box-control action only when it also improves late-game positioning and remains safety-neutral.

Disallowed behavior:

- Broad depth search or aggressive endgame search.
- Random chasing.
- Entering corridors/dead ends to pressure an enemy.
- Suicide trades or rank-speculative trades.
- Introducing `PLACE_BOMB` while `HYBRID_CONVERSION_ALLOW_BOMB=false`.
- Model-owned bomb placement.

If bomb experimentation is later enabled, the conversion layer may only retain or veto a bomb already chosen by production. It may not originate a bomb action during this feature's first validation cycle.

### 1.5 Required Counters

Add compact in-memory counters:

- `conversion_active_steps`
- `conversion_candidate_evaluated`
- `conversion_action_changed`
- `conversion_bomb_considered`
- `conversion_bomb_accepted`
- `conversion_bomb_rejected_safety`
- `conversion_fallback_to_production`
- `conversion_low_risk_gate_rejected`

Also maintain bounded reason counts for low-risk rejection and fallback causes. Do not write per-action logs by default.

### 1.6 Design Re-check

Before implementation proceeds to validation, re-check:

- Candidate remains isolated.
- Production and protected baseline have no diffs caused by this feature.
- Disabled code path returns production decision without extra timing-dependent behavior.
- Emergency escape executes before conversion logic.
- Bomb introduction is disabled.
- All uncertainty paths return production action.

**Post-design constitution result required**: PASS.

## Phase 2: Analysis And Benchmark Tooling

### 2.1 Production Analysis Script

Plan for `scripts/participant/analyze_production_conversion.py`:

- Discover only supported JSON log/replay formats.
- Deduplicate matches by stable identifiers such as seed, timestamp, and roster when available.
- Stream replay files rather than retaining all frames in memory.
- Aggregate metrics incrementally.
- Save only compact analysis JSON if needed; the required user-facing output is the Markdown analysis report.
- Include data-quality warnings and unavailable metric counts.

### 2.2 Candidate Benchmark Script

Create `scripts/participant/benchmark_hybrid_conversion_candidate.py` with modes for:

- Disabled parity.
- 100-episode smoke.
- One final-validation block at a time.
- Merge of five compact blocks.

The script compares:

- `agent/hybrid_agent_conversion_candidate/`
- Current `submission/agent.py`

Use balanced rosters and independent seed blocks. Cache loaded agents where safe, resetting per-episode state and counters without reloading model checkpoints every match.

### 2.3 Compact Log Schema

`logs/hybrid_conversion_candidate_benchmark.json` and any temporary block summaries contain only:

- Configuration and seed range.
- Episode and match counts.
- Rank distribution and rank-1 frequency.
- Wins/draws/losses and rates.
- Average rank and survival.
- Games over step 400 and timeout finishes.
- Self-bomb and enemy-bomb deaths.
- Timeout/error/invalid counts.
- Average, p95, and maximum action latency.
- Conversion counters and rejection-reason counts.
- Gate validity and verdict reasons.

Do not store frames, full replays, per-action runtime rows, full observations, or unbounded intervention histories.

## Phase 3: Validation Plan

### 3.1 Static Verification

Run before behavioral validation:

- Compile candidate and benchmark/analysis scripts with `python -m py_compile`.
- Verify no candidate imports are missing in a package-like load.
- Confirm no diff introduced by this feature in `submission/agent.py` or `agent/hybrid_agent_online_robust/`.
- Confirm no submission zip was generated.

### 3.2 Disabled Parity Gate

Configuration:

- `HYBRID_CONVERSION_ENABLE=false`.
- Compare candidate and current production on identical observations.
- At least 6000 alive-agent decisions.
- Isolate unrelated model latency gates so wall-clock variance cannot create false mismatches.

Pass condition:

- Exactly `0 / 6000` mismatches, or more than 6000 decisions with zero mismatches.

Failure action:

- Stop immediately.
- Fix parity before any enabled benchmark.
- Status remains `REJECT_PROMOTION` until rerun passes.

### 3.3 100-Episode Smoke Gate

Run candidate enabled against current production using balanced slots and multiple seeds.

Required metrics:

- Full rank distribution.
- Rank-1 frequency.
- Win/draw/loss.
- Average rank and survival.
- Step >400 and timeout counts.
- Self-bomb and enemy-bomb deaths.
- Runtime timeout/error/invalid counts.
- Average and p95 latency.
- All conversion counters.

Smoke rejection conditions:

- Candidate average rank worsens.
- Candidate loss rate increases.
- Candidate self-bomb deaths increase.
- Any candidate timeout, error, or invalid action occurs.
- Disabled parity is no longer zero.
- `conversion_active_steps` or `conversion_candidate_evaluated` is zero.
- Conversion logic activates but never changes any action, making performance comparison inconclusive.

Rank-1 frequency improvement is recorded at smoke scale but is not treated as conclusive unless sample size is sufficient. Final validation remains mandatory for promotion.

### 3.4 Final Validation Gate

Run only after smoke passes:

- Five independent seed blocks.
- 300 episodes per block.
- One process/command per block.
- One compact JSON summary per block.
- Merge only after all five blocks are valid.

Each block is invalid if:

- Candidate feature flag is not enabled.
- Conversion counters show only disabled behavior.
- Candidate has runtime errors or invalid actions.
- Block output is incomplete or contains fewer than 300 episodes.

### 3.5 Promotion Gate

Return `PROMOTE_CANDIDATE` only when all conditions pass over merged 5 x 300 results:

- All five blocks are valid.
- Disabled parity remains zero mismatches.
- Candidate average rank is lower than production average rank.
- Candidate rank-1 frequency is higher than production rank-1 frequency.
- Candidate loss rate is less than or equal to production loss rate.
- Candidate self-bomb deaths are less than or equal to production self-bomb deaths.
- Candidate timeout, error, and invalid-action counts are zero.
- Conversion counters prove the feature activated, evaluated candidates, and changed actions.

Otherwise return `REJECT_PROMOTION` and list each failed gate.

No production update or submission packaging follows automatically from a pass.

## Phase 4: Reporting

### Production Analysis Report

`docs/PRODUCTION_CONVERSION_ANALYSIS.md` will include:

- Data coverage.
- Required conversion metrics.
- Opportunity definitions.
- Conversion bottleneck findings.
- Candidate design implications.
- Limitations and rejected hypotheses.

### Candidate Report

`docs/HYBRID_CONVERSION_CANDIDATE_REPORT.md` will include:

- Candidate behavior contract and flags.
- Disabled parity result.
- Smoke result.
- Final block table only if smoke passed.
- Aggregate production-versus-candidate metrics.
- Conversion counters.
- Promotion gate evaluation.
- Final status: `PROMOTE_CANDIDATE` or `REJECT_PROMOTION`.

## Deliverables

| Deliverable | Purpose |
|---|---|
| `docs/PRODUCTION_CONVERSION_ANALYSIS.md` | Evidence-based diagnosis of production conversion failures. |
| `agent/hybrid_agent_conversion_candidate/` | Isolated candidate copied from current production with conservative conversion layer. |
| `scripts/participant/analyze_production_conversion.py` | Streaming compact production-log analysis. |
| `scripts/participant/benchmark_hybrid_conversion_candidate.py` | Parity, smoke, block validation, and merge harness. |
| `docs/HYBRID_CONVERSION_CANDIDATE_REPORT.md` | Candidate behavior, benchmark evidence, and verdict. |
| `logs/hybrid_conversion_candidate_benchmark.json` | Compact merged benchmark summary. |

## Risks And Mitigations

| Risk | Mitigation |
|---|---|
| False-positive kill or pressure opportunities recreate aggressive regressions | Use high-precision offline detectors; require unchanged own safe area and production-approved actions. |
| Rank-1 gains come from increased fourth-place outcomes | Enforce non-increasing loss rate and self-bomb deaths. |
| Timing differences break parity | Disable only the conversion layer while normalizing unrelated latency gates during parity comparison. |
| Model package imports differ between repo and organizer runtime | Copy production package structure and test package-like loading before benchmark. |
| Logs are incomplete or mixed across agent versions | Inventory coverage and separate samples; do not fabricate unavailable metrics. |
| Benchmark logs grow excessively | Stream analysis and persist compact summaries only. |
| Small smoke sample gives misleading win-rate result | Use smoke only as a safety gate; require 5 x 300 for promotion. |

## Complexity Tracking

No constitution violation is planned. The candidate duplicates current production into an isolated folder intentionally because exact fallback and protected production boundaries are more important than reducing code duplication during experimentation.

## Execution Hold

Do not run implementation until `/speckit.tasks` is reviewed.
