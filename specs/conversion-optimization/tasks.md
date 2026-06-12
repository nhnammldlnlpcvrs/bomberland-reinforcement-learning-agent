---

description: "Actionable tasks for conservative Bomberland conversion optimization"
---

# Tasks: Conversion Optimization

**Input**: Design documents from `/specs/conversion-optimization/`

**Prerequisites**: `spec.md`, `plan.md`

**Execution Rule**: Complete tasks in order unless marked `[P]`. Stop at every gate. Do not implement or validate beyond a failed gate.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: May run in parallel because it touches different files and has no unmet dependency.
- **[US1]**: Diagnose production conversion failures.
- **[US2]**: Build the isolated conservative conversion candidate.
- **[US3]**: Validate and decide without production risk.
- Every task includes an exact file path or verification target.

---

## Phase A: Spec And Plan Verification

**Purpose**: Confirm the feature is authorized, scoped, and safe before creating implementation artifacts.

- [ ] T001 Review `.specify/memory/constitution.md`, `specs/conversion-optimization/spec.md`, and `specs/conversion-optimization/plan.md`; record any conflict in `specs/conversion-optimization/tasks.md` before implementation begins. **Verify**: no task authorizes production modification, protected-baseline modification, PPO/RL work, aggressive endgame search, or submission packaging.
- [ ] T002 Verify current production identity by loading `submission/agent.py`, confirming it exposes class `Agent`, and recording its team identifier and packaged sibling dependencies in the implementation notes for `docs/PRODUCTION_CONVERSION_ANALYSIS.md`. **Verify**: the inspected path is exactly `submission/agent.py`.
- [ ] T003 Capture the pre-feature protected-file state with `git diff -- submission/agent.py agent/hybrid_agent_online_robust/` and `git status --short`. **Verify**: save or note the baseline state so later checks can distinguish pre-existing changes from feature changes.
- [ ] T004 Confirm the feature output paths do not already contain unrelated user work: `agent/hybrid_agent_conversion_candidate/`, `scripts/participant/analyze_production_conversion.py`, `scripts/participant/benchmark_hybrid_conversion_candidate.py`, `docs/PRODUCTION_CONVERSION_ANALYSIS.md`, and `docs/HYBRID_CONVERSION_CANDIDATE_REPORT.md`. **Verify**: any pre-existing file is reviewed and preserved rather than overwritten blindly.
- [ ] T005 Re-check the constitution gates from `specs/conversion-optimization/plan.md`. **Verify**: mark the setup checkpoint PASS before starting Phase B.

**Checkpoint A**: Production and protected baseline are identified and remain read-only; feature scope is approved for analysis only.

---

## Phase B: Production Analysis Tooling - User Story 1 (Priority: P1)

**Goal**: Produce a compact, reproducible diagnosis of why production survives late but converts too few matches into rank 1.

**Independent Test**: Run the analysis tool over available supported data and verify that the JSON and Markdown outputs contain all required metrics or explicit unavailable-data explanations.

### Evidence Inventory

- [ ] T006 [P] [US1] Inventory benchmark summary files under `logs/` and classify each by agent version, match count, available rank fields, terminal-step fields, death attribution fields, and seed coverage. **Verify**: inventory counts are available to the analysis script without loading full files into memory unnecessarily.
- [ ] T007 [P] [US1] Inventory replay files under `logs/` and participant output directories; classify supported schemas by presence of frames/history, actions, players, bombs, ranks, survival steps, and total steps. **Verify**: unsupported schemas are listed rather than silently ignored.
- [ ] T008 [P] [US1] Inspect engine rank, timeout, bomb radius, bomb ownership, and chain-reaction semantics in `engine/` and relevant evaluation runners. **Verify**: document the exact source functions used for metric definitions.
- [ ] T009 [US1] Define a deterministic deduplication key for production samples in `scripts/participant/analyze_production_conversion.py` design notes, using seed, roster, timestamp/file identity, or another defensible combination. **Verify**: duplicate samples cannot inflate aggregate counts unnoticed.

### Analysis Script Foundation

- [ ] T010 [US1] Create `scripts/participant/analyze_production_conversion.py` with CLI arguments for input roots, output JSON, output report, production labels, and optional sample limits. **Verify**: `python -m py_compile scripts/participant/analyze_production_conversion.py` passes and `--help` runs.
- [ ] T011 [US1] Implement streaming discovery and schema adapters in `scripts/participant/analyze_production_conversion.py` for supported compact benchmark summaries and replay JSON formats. **Verify**: the script processes one supported file at a time and does not retain all replay frames globally.
- [ ] T012 [US1] Implement data-quality accounting in `scripts/participant/analyze_production_conversion.py`: files discovered, files supported, files skipped, duplicate samples, production matches, production appearances, and metric-specific coverage. **Verify**: every skipped file has a compact reason count.

### Core Outcome Metrics

- [ ] T013 [P] [US1] Implement rank-distribution aggregation in `scripts/participant/analyze_production_conversion.py`, including rank 1/2/3/4 counts, percentages, ties, and average rank. **Verify**: a small fixture or known log produces totals equal to production appearances.
- [ ] T014 [P] [US1] Implement late-game aggregation: terminal step, games reaching step greater than 400, timeout finishes, and alive-player-count distribution at step 350. **Verify**: metrics include numerator, denominator, percentage, and unavailable coverage.
- [ ] T015 [P] [US1] Implement production bomb-frequency aggregation using valid placements per match and per 100 alive action decisions. **Verify**: duplicate bomb actions on an occupied bomb tile are not counted as new placements.
- [ ] T016 [P] [US1] Implement self-bomb, enemy-bomb, and ambiguous bomb-death attribution using bomb owner, blast footprint, timer horizon, and chain-reaction-aware timing. **Verify**: attribution categories are mutually exclusive and ambiguous cases are counted separately.
- [ ] T017 [P] [US1] Implement enemy-kill frequency attribution for production-owned bombs. **Verify**: simultaneous or multi-owner chain deaths are marked ambiguous unless ownership is defensible.

### Opportunity Metrics

- [ ] T018 [US1] Implement production-equivalent danger-map and chain-reaction reconstruction in `scripts/participant/analyze_production_conversion.py`, reusing structured production helpers where package-safe. **Verify**: reconstructed danger timing matches production on selected replay states.
- [ ] T019 [US1] Implement safe-action reconstruction for production replay states, including passability, arrival-time danger, reachable safe area, emergency escape, and recent-own-bomb commitment. **Verify**: unsafe actions are never labeled conversion opportunities.
- [ ] T020 [US1] Implement the conservative `safe bomb opportunity declined` detector: production bomb checks approve safe/useful placement, production chooses non-bomb, and all required fields are available. **Verify**: missing or ambiguous data increments unavailable coverage instead of opportunity count.
- [ ] T021 [US1] Implement the conservative `missed kill opportunity` detector using production-approved actions, enemy escape-area reduction, blast geometry, and unchanged own safe area. **Verify**: detector favors precision and emits compact reason counts for rejected opportunities.
- [ ] T022 [US1] Implement the `late-game pressure opportunity missed` detector for step >=350 or low alive-player count, limited to already-safe movement actions that reduce enemy reachable cells without reducing own safe area. **Verify**: no broad search, chasing, or unsafe bomb logic is used.

### Analysis Outputs

- [ ] T023 [US1] Emit compact aggregate data to `logs/production_conversion_analysis.json`; exclude frames, observations, full action histories, and per-action runtime rows. **Verify**: inspect file size and schema; all required metrics and coverage counts are present.
- [ ] T024 [US1] Generate `docs/PRODUCTION_CONVERSION_ANALYSIS.md` with data inventory, definitions, rank distribution, rank-1 frequency, average rank, late-game metrics, bomb frequency, kill frequency, opportunity metrics, death attribution, limitations, and representative replay references. **Verify**: every required metric is reported or explicitly marked unavailable with coverage reason.
- [ ] T025 [US1] Add a ranked conversion-bottleneck section and explicit rejected hypotheses to `docs/PRODUCTION_CONVERSION_ANALYSIS.md`. **Verify**: aggressive endgame search and pure RL/PPO remain rejected directions.
- [ ] T026 [US1] Add the analysis go/no-go decision to `docs/PRODUCTION_CONVERSION_ANALYSIS.md`. **Verify**: candidate implementation is authorized only if at least one reproducible low-risk opportunity has sufficient coverage; otherwise status is `REJECT_PROMOTION` and Tasks T027 onward stop.

**Checkpoint B - Analysis Gate**: `docs/PRODUCTION_CONVERSION_ANALYSIS.md` and `logs/production_conversion_analysis.json` are complete and compact. Continue only on a documented GO decision.

---

## Phase C: Candidate Setup - User Story 2 (Priority: P2)

**Goal**: Create an isolated exact-production candidate with conversion disabled by default.

**Independent Test**: Load the candidate package with conversion disabled and verify it imports, loads the same model dependencies, and is structurally independent from production.

- [ ] T027 [US2] Create `agent/hybrid_agent_conversion_candidate/` only after Checkpoint B passes. **Verify**: no files under `submission/` or `agent/hybrid_agent_online_robust/` are edited.
- [ ] T028 [US2] Copy current production `submission/agent.py` into `agent/hybrid_agent_conversion_candidate/agent.py`. **Verify**: before conversion changes, file behavior and required helper imports match production.
- [ ] T029 [US2] Copy only required production sibling dependencies from `submission/` into `agent/hybrid_agent_conversion_candidate/` to preserve package-compatible model loading. **Verify**: candidate loads from its own folder without importing submission-local files.
- [ ] T030 [US2] Add feature flags to `agent/hybrid_agent_conversion_candidate/agent.py`: `HYBRID_CONVERSION_ENABLE=false`, `HYBRID_CONVERSION_START_STEP=350`, `HYBRID_CONVERSION_LOW_RISK_ONLY=true`, `HYBRID_CONVERSION_ALLOW_BOMB=false`, and `HYBRID_CONVERSION_MAX_LATENCY_MS=10`. **Verify**: defaults match the plan exactly.
- [ ] T031 [US2] Add counters to `agent/hybrid_agent_conversion_candidate/agent.py`: `conversion_active_steps`, `conversion_candidate_evaluated`, `conversion_action_changed`, `conversion_bomb_considered`, `conversion_bomb_accepted`, `conversion_bomb_rejected_safety`, `conversion_fallback_to_production`, and `conversion_low_risk_gate_rejected`. **Verify**: counters are initialized per agent and require no filesystem logging.
- [ ] T032 [US2] Add bounded conversion rejection-reason counts and optional bounded intervention examples in `agent/hybrid_agent_conversion_candidate/agent.py`. **Verify**: no unbounded per-action log accumulation exists.
- [ ] T033 [US2] Compile and package-load the untouched candidate baseline. **Verify**: `python -m py_compile agent/hybrid_agent_conversion_candidate/agent.py` passes and all model/checkpoint imports resolve locally.
- [ ] T034 [US2] Re-run protected-file checks against the baseline captured in T003. **Verify**: this feature introduced no change to `submission/agent.py` or `agent/hybrid_agent_online_robust/`.

**Checkpoint C**: Candidate package is isolated, package-compatible, and conversion remains disabled by default.

---

## Phase D: Conservative Conversion Logic - User Story 2 (Priority: P2)

**Goal**: Add only high-precision, low-risk conversion reranking derived from the production analysis.

**Independent Test**: Feed representative low-risk and high-risk replay states to the candidate and verify that only eligible safe states can change action.

### Low-Risk Gate

- [ ] T035 [US2] Implement a conversion master bypass in `agent/hybrid_agent_conversion_candidate/agent.py` that returns the production action immediately when `HYBRID_CONVERSION_ENABLE=false`. **Verify**: no conversion scorer, counter timing dependency, or model call can change disabled behavior.
- [ ] T036 [US2] Implement the start-condition gate using `HYBRID_CONVERSION_START_STEP` and analysis-approved low-player conditions. **Verify**: missing step data returns production behavior.
- [ ] T037 [US2] Implement emergency-escape and path-to-safety bypass before conversion evaluation. **Verify**: representative emergency states return exactly the production action.
- [ ] T038 [US2] Implement blast-risk rejection for current cell, production destination, and candidate destinations using chain-reaction timing and arrival/leave timing. **Verify**: cells dangerous before safe departure are rejected.
- [ ] T039 [US2] Implement recent-own-bomb commitment rejection and minimum reachable-safe-area rejection using thresholds justified by `docs/PRODUCTION_CONVERSION_ANALYSIS.md`. **Verify**: rejected states increment `conversion_low_risk_gate_rejected` and return production.
- [ ] T040 [US2] Implement latency-budget checking against `HYBRID_CONVERSION_MAX_LATENCY_MS`. **Verify**: budget exhaustion increments fallback counters and returns production action.

### Safe Candidate Evaluation

- [ ] T041 [US2] Build the conversion candidate set from production-approved safe actions only. **Verify**: invalid, unsafe, blast-exposed, dead-end, and safe-area-reducing actions are absent.
- [ ] T042 [US2] Implement conservative movement-opportunity scoring using only analysis-approved signals such as enemy escape-area reduction, safe route control, and safety-neutral positioning. **Verify**: no aggressive chasing or broad endgame search is introduced.
- [ ] T043 [US2] Require a clear deterministic improvement margin before changing the production action. **Verify**: ties, weak evidence, missing data, or disagreement return production.
- [ ] T044 [US2] Enforce `HYBRID_CONVERSION_ALLOW_BOMB=false` so conversion cannot change a non-bomb production action into `PLACE_BOMB`. **Verify**: exhaustive action-path inspection shows no bomb introduction route.
- [ ] T045 [US2] If production already selected `PLACE_BOMB`, preserve production ownership of that decision and record `conversion_bomb_considered`; only analysis-approved conservative veto behavior may be considered if explicitly supported. **Verify**: conversion never strengthens or originates bomb aggression.
- [ ] T046 [US2] Wire conversion counters and bounded reason logging into accepted, rejected, and fallback paths. **Verify**: counters distinguish active states, evaluated candidates, changed actions, bomb outcomes, and low-risk rejections.
- [ ] T047 [US2] Add deterministic state-level checks or a small verification script for emergency bypass, unsafe-bomb prohibition, blast rejection, safe-area rejection, and uncertainty fallback. **Verify**: each check has at least one positive and one negative state case.
- [ ] T048 [US2] Compile and load the completed candidate package. **Verify**: no missing dependency, syntax error, or load-time model error occurs.
- [ ] T049 [US2] Re-run protected-file checks. **Verify**: `submission/agent.py` and `agent/hybrid_agent_online_robust/` remain unchanged by this feature.

**Checkpoint D**: Candidate logic is safety-gated, movement-first, latency-bounded, and exact-fallback on uncertainty.

---

## Phase E: Benchmark Harness - User Story 3 (Priority: P3)

**Goal**: Build compact staged validation that prevents unsafe or inconclusive candidates from reaching final validation.

**Independent Test**: Exercise parity, smoke, block, and merge modes with small episode counts and inspect compact output schemas.

- [ ] T050 [US3] Create `scripts/participant/benchmark_hybrid_conversion_candidate.py` with CLI modes for disabled parity, 100-episode smoke, one final block, and merge. **Verify**: `--help` documents each mode and `py_compile` passes.
- [ ] T051 [US3] Implement package-safe loading for `agent/hybrid_agent_conversion_candidate/` and current `submission/agent.py`. **Verify**: candidate model dependencies resolve from the candidate folder and production resolves from submission.
- [ ] T052 [US3] Implement per-episode agent-state reset and model-instance caching in `scripts/participant/benchmark_hybrid_conversion_candidate.py`. **Verify**: counters reset each episode while checkpoints are not repeatedly loaded.
- [ ] T053 [US3] Implement balanced two-candidate/two-production rosters with deterministic independent seeds. **Verify**: each side receives equal appearances and slot distribution over a block.
- [ ] T054 [US3] Implement final-rank, rank-distribution, win/draw/loss, average-rank, survival, timeout-finish, and step-greater-than-400 aggregation. **Verify**: summary totals equal all agent appearances.
- [ ] T055 [US3] Implement self-bomb, enemy-bomb, and ambiguous death attribution consistent with the analysis definitions. **Verify**: categories are mutually exclusive.
- [ ] T056 [US3] Implement runtime aggregation: timeout, error, invalid action, average latency, p95 latency, and maximum latency. **Verify**: no per-action runtime rows are persisted.
- [ ] T057 [US3] Aggregate all required conversion counters and rejection-reason counts. **Verify**: enabled runs can prove conversion activation and action changes.
- [ ] T058 [US3] Implement compact JSON serialization for smoke, per-block, and merged summaries. **Verify**: outputs contain no frames, full observations, full replays, or raw per-action rows.
- [ ] T059 [US3] Implement disabled parity comparison over identical observations while normalizing unrelated production model latency gates. **Verify**: timing variance cannot create false feature mismatches.
- [ ] T060 [US3] Implement smoke-gate evaluation and final-validation blocking. **Verify**: a synthetic or small-run gate failure prevents block commands from being recommended or automatically executed.
- [ ] T061 [US3] Implement final merge gate requiring five valid 300-episode block files. **Verify**: missing, disabled-only, incomplete, or invalid blocks force `REJECT_PROMOTION`.
- [ ] T062 [US3] Add final verdict logic for average rank, rank-1 frequency, loss rate, self-bomb deaths, runtime failures, parity, and counter activation. **Verify**: every failed condition appears as a verdict reason.

**Checkpoint E**: Harness is compact, resumable, balanced, and gate-driven.

---

## Phase F: Disabled Parity And Smoke Validation - User Story 3 (Priority: P3)

**Goal**: Prove exact disabled behavior and reject unsafe candidates before expensive validation.

- [ ] T063 [US3] Run static verification for candidate and scripts with `python -m py_compile`. **Verify**: all target files compile.
- [ ] T064 [US3] Run disabled parity for at least 6000 alive-agent decisions with `HYBRID_CONVERSION_ENABLE=false`. **Verify**: result is exactly zero mismatches and is stored in the compact benchmark output/report.
- [ ] T065 [US3] If T064 has any mismatch, stop validation, set report status to `REJECT_PROMOTION`, document examples, and do not run smoke or final validation. **Verify**: no later validation artifacts are created from the failed candidate.
- [ ] T066 [US3] Run the 100-episode enabled smoke benchmark against current production only after T064 passes. **Verify**: compact metrics include rank distribution, rank-1 frequency, average rank, loss, bomb deaths, runtime metrics, late-game metrics, and counters.
- [ ] T067 [US3] Evaluate smoke gates: candidate average rank must not worsen, loss rate must not increase, self-bomb deaths must not increase, timeout/error/invalid must remain zero, and conversion logic must activate and change actions. **Verify**: each condition has an explicit pass/fail value.
- [ ] T068 [US3] If smoke fails, write `REJECT_PROMOTION` to `docs/HYBRID_CONVERSION_CANDIDATE_REPORT.md`, explain every failed gate, and stop before final validation. **Verify**: no 300-episode block is run.

**Checkpoint F - Smoke Gate**: Continue only if parity is zero mismatch and every smoke safety gate passes.

---

## Phase G: Conditional Final Validation - User Story 3 (Priority: P3)

**Goal**: Establish multi-seed promotion evidence only after smoke passes.

- [ ] T069 [US3] Run final validation block 1 with 300 episodes and write a compact block summary under `logs/`. **Verify**: block is enabled, complete, and conversion counters are non-disabled.
- [ ] T070 [US3] Run final validation block 2 with 300 independent episodes and write a compact block summary. **Verify**: seed range differs from block 1 and block validity passes.
- [ ] T071 [US3] Run final validation block 3 with 300 independent episodes and write a compact block summary. **Verify**: block validity passes.
- [ ] T072 [US3] Run final validation block 4 with 300 independent episodes and write a compact block summary. **Verify**: block validity passes.
- [ ] T073 [US3] Run final validation block 5 with 300 independent episodes and write a compact block summary. **Verify**: block validity passes.
- [ ] T074 [US3] Merge the five block summaries into `logs/hybrid_conversion_candidate_benchmark.json`. **Verify**: total episode count is 1500, all five blocks are present, and no raw replay data is included.
- [ ] T075 [US3] Evaluate the final promotion gate: average rank improves, rank-1 frequency improves, loss rate does not increase, self-bomb deaths do not increase, timeout/error/invalid are zero, parity remains zero mismatch, and counters prove activation. **Verify**: output is exactly `PROMOTE_CANDIDATE` or `REJECT_PROMOTION` with reasons.

**Checkpoint G - Final Gate**: A pass recommends only the candidate status; it does not authorize production changes or packaging.

---

## Phase H: Reports And Final Safety Audit

**Purpose**: Finalize compact evidence and ensure protected state remains intact.

- [ ] T076 [US3] Generate or update `docs/HYBRID_CONVERSION_CANDIDATE_REPORT.md` with behavior contract, flags, parity, smoke, conditional block table, aggregate comparison, counters, gate reasons, and final status. **Verify**: report ends with exactly `PROMOTE_CANDIDATE` or `REJECT_PROMOTION`.
- [ ] T077 [US3] Verify `logs/hybrid_conversion_candidate_benchmark.json` contains only compact summaries and required metrics. **Verify**: inspect schema and file size; no frames or per-action rows exist.
- [ ] T078 Re-run `git diff -- submission/agent.py agent/hybrid_agent_online_robust/` against the baseline from T003. **Verify**: this feature introduced no protected-file changes.
- [ ] T079 Verify no new submission archive was generated under `dist/`, repository root, or feature folders. **Verify**: record the check in the final report notes.
- [ ] T080 Verify no PPO/RL training scripts, checkpoints, or aggressive endgame-search code were added or modified by this feature. **Verify**: final change list is limited to analysis, candidate, benchmark, spec/report, and compact log artifacts.
- [ ] T081 Print the final decision and production state. **Verify**: production remains current `submission/agent.py` regardless of candidate verdict until a separate explicit promotion task is issued.

---

## Explicit Non-Tasks

- [ ] NT001 Do not modify `submission/agent.py` during this feature.
- [ ] NT002 Do not modify any file under `agent/hybrid_agent_online_robust/`.
- [ ] NT003 Do not generate or update a submission zip.
- [ ] NT004 Do not train, fine-tune, or evaluate a new pure RL/PPO policy.
- [ ] NT005 Do not continue the rejected aggressive endgame search, broad shallow search, random chasing, or model-owned bomb placement direction.
- [ ] NT006 Do not promote automatically after a passing final validation; wait for a separate explicit promotion task.

---

## Dependencies And Execution Order

### Phase Dependencies

1. **Phase A** has no dependencies and blocks all implementation.
2. **Phase B** depends on Phase A and blocks candidate creation.
3. **Phase C** depends on a GO result from Checkpoint B.
4. **Phase D** depends on the isolated candidate from Phase C and analysis findings from Phase B.
5. **Phase E** depends on the completed candidate behavior contract.
6. **Phase F** depends on the benchmark harness and blocks final validation on any failure.
7. **Phase G** runs only if Checkpoint F passes.
8. **Phase H** follows the terminal validation state, whether rejected at smoke or decided after final validation.

### Parallel Opportunities

- T006, T007, and T008 may run in parallel.
- T013 through T017 may run in parallel after schema adapters are complete.
- Documentation drafting for metric definitions may proceed alongside analysis implementation, but computed values must wait for completed aggregation.
- Final validation blocks T069-T073 must run as separate processes; they may run sequentially for resource stability or in parallel only when machine capacity guarantees no latency distortion.

### Stop Conditions

- Stop after T026 if analysis finds no reproducible low-risk opportunity.
- Stop after T065 if disabled parity is not zero mismatch.
- Stop after T068 if smoke fails any gate.
- Never execute Phase G merely to gather more evidence for a smoke-failed candidate.

## Verification Matrix

| Requirement | Primary Tasks |
|---|---|
| Production conversion analysis | T006-T026 |
| Isolated candidate folder | T027-T034 |
| Required feature flags | T030 |
| Required counters | T031, T046, T057 |
| Low-risk-only conversion | T035-T043 |
| No emergency escape override | T037, T047 |
| No unsafe bomb introduction | T044-T045, T047 |
| Exact fallback on uncertainty | T035, T038-T043 |
| Disabled parity 0/6000 | T059, T064-T065 |
| 100-episode smoke | T066-T068 |
| Conditional 5 x 300 | T069-T075 |
| Compact logging | T023, T058, T074, T077 |
| Final decision gate | T075-T076 |
| Protected files unchanged | T003, T034, T049, T078 |

Tasks are ready for review. Do not implement until approved.
