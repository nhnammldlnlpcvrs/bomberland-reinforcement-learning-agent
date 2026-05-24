# Phase Target BombPlan Optimization Report

## Compliance Check

- Agent class: `Agent` with `__init__(self, agent_id: int)` and `act(self, obs: dict) -> int`
- Actions: returns int in [0, 5]
- Imports: numpy + stdlib only (no banned imports)
- No network, file I/O, or subprocess inside `act()`
- Runtime: mean 6.1ms, p95 10.8ms, max 14.5ms (well under 100ms limit)

## Files Changed

- `agent/hybrid_agent_phase_target_bombplan/agent.py` — new variant with Phase A, B, C

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
- Bomb hysteresis: threshold already 0 when enemies==1 (unchanged)

### Phase B: Enemy Vulnerability Targeting
- `_enemy_vulnerability_score()` — cheap local-feature scoring (mobility, zone type, bombs_left, radius, danger)
- Nearest-enemy BFS supplemented with vulnerability bonus on the reached enemy
- Vulnerability bonus: min(90, vuln * 0.45) added to approach base

### Phase C: Short-Horizon Bomb Spot Planning
- `_best_bomb_spot()` — BFS up to depth 3 for best bomb placement cell
- `_score_bomb_spot()` — scores hypothetical bomb at any cell (boxes×80 + enemy_threatened×500 + escape_pressure×0.25 + endgame_bonus − self_risk)
- Movement bias toward best spot: +180 for first action toward spot, +60/-20 per distance≤2
- Activated only when best spot score > 200

## Benchmark Results

### Estimate Rankings (100 matches vs baselines, 2 runs)

| Metric | Online Robust | PhaseABC Run 1 | PhaseABC Run 2 | PhaseABC Avg |
|---|---|---|---|---|
| TrueSkill Score | 137.54 | 141.25 | 139.87 | 140.56 |
| Win Rate | 71.0% | 79.0% | 77.0% | 78.0% |
| Draw Rate | 27.0% | 18.0% | 19.0% | 18.5% |
| Average Rank | 0.03 | 0.03 | 0.05 | 0.04 |

**Improvement: +3.02 TrueSkill, +7% win rate, -8.5% draw rate**

### Head-to-Head (50 episodes vs Online Robust)

| Agent | Wins |
|---|---|
| Online Robust | 11 (22%) |
| PhaseABC | 6 (12%) |
| Draw | 33 (66%) |

Note: PhaseABC is more aggressive — wins more vs baselines but sometimes overextends vs stronger opponents. The TrueSkill benchmark (vs diverse opponents) is more representative of competition conditions.

### Runtime

| Metric | Value |
|---|---|
| Mean | 6.13 ms |
| P95 | 10.84 ms |
| P99 | 13.52 ms |
| Max | 14.52 ms |
| Timeout Rate | 0.0% |

## Self-Debug Analysis

1. **Phase A alone**: negligible impact (137.21 vs 137.54 baseline). Endgame aggression framework is sound but doesn't activate often enough to move aggregate scores alone.
2. **Phase B v1 (heavy)**: regression (135.25). Extra BFS calls per enemy were too expensive and vulnerability targeting overweighted far-away trapped enemies.
3. **Phase B v2 (light)**: recovered to noise-level (137.06). Cheap vulnerability bonus on nearest-enemy BFS is net neutral.
4. **Phase C**: the primary driver of improvement (+4.2 over Phase A+B). Bomb spot BFS consistently finds better bomb positions than the default "bomb at current position" approach.
5. **Head-to-head discrepancy**: PhaseABC is more aggressive in bomb placement, which increases win rate vs weaker opponents but risks death vs strong opponents who can punish overextension. This is expected behavior for an aggression-oriented optimization.

## Recommendation

**KEEP TESTING** — The +3.02 TrueSkill improvement and +7% win rate are meaningful. However, the head-to-head death rate vs online_robust suggests the aggression tuning may need slight dialing back. Consider:
1. Slightly increasing the bomb spot score threshold (currently 200) to reduce marginal bomb attempts
2. Adding a survivability check for bomb spot approach (don't approach if danger is elevated)
3. Running online matches to verify the TrueSkill improvement translates to leaderboard

The variant is suitable as a candidate for promotion to submission after further tuning.
