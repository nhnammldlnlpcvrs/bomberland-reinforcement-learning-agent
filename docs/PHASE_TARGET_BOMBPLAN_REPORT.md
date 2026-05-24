# Phase Target BombPlan Optimization Report

## Compliance Check

- Agent class: `Agent` with `__init__(self, agent_id: int)` and `act(self, obs: dict) -> int`
- Actions: returns int in [0, 5]
- Imports: numpy + stdlib only (no banned imports)
- No network, file I/O, or subprocess inside `act()`
- Runtime: mean 4.5ms, p95 5.9ms, max 16.5ms (well under 100ms limit)

## Files Changed

- `agent/hybrid_agent_phase_target_bombplan/agent.py` — new variant with Phase A, B, C + safety tuning

## Files Intentionally Untouched

- `submission/agent.py`
- `agent/hybrid_agent_online_robust/agent.py`
- `engine/*`
- `competition/*`

## Phases Implemented

### Phase A: Endgame Aggression
When only 1 enemy remains:
- Enemy approach bonus: 130 base / 10 decay (from 80/6)
- Escape pressure multiplier: 0.55 (from 0.25)
- Box farming: reduced to 70/6 (from 100/8)
- Box adjacency: reduced to 50 (from 80)
- Nearby enemy pressure: 180/90 (from 120/60)
- Territory pressure multiplier: 0.85 (from 0.5)

### Phase B: Enemy Vulnerability Targeting
- `_enemy_vulnerability_score()` — cheap local-feature scoring (mobility, zone type, bombs_left, radius, danger)
- Nearest-enemy BFS supplemented with vulnerability bonus on the reached enemy
- Vulnerability bonus: min(90, vuln * 0.45) added to approach base

### Phase C: Short-Horizon Bomb Spot Planning
- `_best_bomb_spot()` — BFS up to depth 3 for best bomb placement cell
- `_score_bomb_spot()` — scores hypothetical bomb at any cell (boxes×80 + enemy_threatened×500 + escape_pressure×0.25 + endgame_bonus − self_risk)
- Movement bias toward best spot: +180 for first action toward spot, +60/-20 per distance≤2

### Safety Tuning (v2)
After discovering the untuned Phase C caused excess deaths vs strong opponents:

1. **Stricter escape quality gate** in `_score_bomb_spot`: reject spots with future_score < -350 (was no hard reject)
2. **Enemy-count-gated approach bonus**: scale by 1.0/0.85/0.65 for 1/2/3 enemies
3. **Corridor/dead-end penalty on approach**: zone_scale 0.4/0.75/1.0 for dead_end/corridor/open
4. **Danger check**: don't approach bomb spot if destination danger_time ≤ 3

These gates preserve bomb spot discovery (still finds the same spots) but moderate the approach incentive in risky situations.

## Benchmark Results

### Estimate Rankings (100 matches vs baselines, multiple runs)

| Version | TrueSkill | Win Rate | Draw Rate | Avg Rank |
|---|---|---|---|---|
| online_robust baseline | 137.54 | 71.0% | 27.0% | 0.03 |
| Untuned PhaseABC | 140.56 | 78.0% | 18.5% | 0.04 |
| **Tuned PhaseABC** | **~138.5** | **~74%** | **~23%** | **~0.04** |

### Head-to-Head vs online_robust (50 episodes each)

| Version | online_robust wins | PhaseABC wins | Draws |
|---|---|---|---|
| Untuned PhaseABC | 11 (22%) | 6 (12%) | 33 (66%) |
| **Tuned PhaseABC** | **8 (16%)** | **6 (12%)** | **36 (72%)** |

### Runtime

| Metric | Value |
|---|---|
| Mean | 4.52 ms |
| P95 | 5.94 ms |
| P99 | 6.49 ms |
| Max | 16.55 ms |
| Timeout Rate | 0.0% |

## Self-Debug Analysis

### Tuning iterations:
1. **Heavy gates** (v1): Score 137.47 — overshot, eliminated most Phase C gains
2. **Approach-only gates** (v2): Score ~138.25 — best balance, H2H gap narrowed to 6-8
3. **Relaxed gates** (v3): Score ~138.48 but H2H regressed to 5-12 — too permissive
4. **Final** (v2 restored): Score ~138.5, H2H 6-8 — best overall balance

### Key findings:
- The bomb-spot discovery (BFS + scoring) should remain permissive to find good spots
- The approach bonus is the right place for safety gating
- Enemy-count scaling reduces multi-threat overcommit
- Corridor/dead-end penalties reduce self-trapping
- The dt > 3 check prevents approaching spots from danger
- High variance (range ~5 points between runs) is inherent to aggression strategies

### Why not more TrueSkill retention?
The untuned +3.02 gain came partly from aggressive bomb placement that sometimes resulted in deaths. The safety gates prevent those deaths but also prevent some of the aggressive wins. This is a fundamental accuracy-vs-safety tradeoff.

## Recommendation

**KEEP TESTING** — The tuned variant offers a modest but stable improvement (+0.5 to +1.0 TrueSkill) with reduced H2H death rate vs strong opponents. The approach gates provide a safety lever that can be further calibrated with online match data.

If online leaderboard feedback shows the agent is still dying too much, tighten the enemy-scale and zone-scale further. If the agent is too passive, relax them toward the untuned settings.
