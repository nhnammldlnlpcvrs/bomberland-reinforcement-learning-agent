# Online Robust Counterfactual Analysis

## Scope

This analysis is intentionally non-invasive:

- No production agent logic was changed.
- `agent/hybrid_agent_online_robust/agent.py` was used only as a reference for heuristic components.
- No engine, evaluator, competition, submission, replay, dataset, or checkpoint files were modified.
- The goal is to find repeated missed-conversion patterns, not to design a new agent.

The new PDF guide was present as `docs/AI Challenge GDGoC HCMUS - Huong dan.pdf`, but this environment did not have a PDF text extractor installed. The text-readable source used for rules was `docs/COMPETITION_GUIDE.md`, which covers the same game mechanics needed for this analysis.

## Useful Guide Ideas

The useful lightweight ideas from the guide are:

1. Turn order matters: movement resolves before bomb placement, then timers decrease, explosions resolve, agents are removed, and items spawn. Counterfactual action analysis must evaluate the state before the next action is applied.
2. Bomb timer and blast geometry are enough for a cheap tactical search: bombs start at timer 7, blasts stop at walls and boxes, and explosions pass through agents.
3. Bomb chain reactions matter for safety. Reusing the online_robust danger map is preferable to a separate simplified danger model.
4. Items are long-term value. Enemy moves toward safe items are a plausible worst-case response in draws.
5. Multiple agents can share a tile. Enemy proximity should be treated as tactical pressure, not as an occupancy block.

## Rejected Guide Ideas

The following ideas were rejected for this phase:

1. Full minimax or MCTS: too expensive and too likely to change the policy direction before we have enough evidence.
2. PPO, DQN, or neural action replacement: previous experiments showed these are not reliable for the current sparse, multi-agent setting.
3. Global aggression tuning: previous less_draw and tactical trigger attempts increased instability or did not transfer reliably.
4. Rewriting safety logic: the remaining losses are important, but weakening safety would likely increase variance.
5. Optimizing only local score: online-vs-local history shows local score is not a sufficient acceptance signal.

## Counterfactual Method

For selected draw/loss replay frames, I evaluated all six candidate actions:

- `STOP`
- `LEFT`
- `RIGHT`
- `UP`
- `DOWN`
- `PLACE_BOMB`

The analysis used online_robust-style components:

- immediate validity and `danger_time <= 1` rejection
- bomb escape validation through `can_escape_after_bomb`
- reachable safe space
- future survivability
- controlled expansion value
- item distance
- enemy escape pressure for hypothetical bombs
- box destruction value
- corridor/dead-end classification

Enemy response assumptions were deliberately simple:

- enemy moves away from our bomb threat
- enemy moves toward an item if safe
- enemy places a bomb if we are already low-mobility or trapped

This is not an exact reproduction of the agent's internal score. It is a replay audit tool to identify repeated failure patterns.

## Aggregate Replay Evidence

Command used:

```powershell
python -m scripts.participant.analyze_matches --log_dir logs/json --team_name HybridAgent --top 20
```

Summary from 572 JSON files:

| Metric | Value |
|---|---:|
| Analyzed matches | 482 |
| Wins | 148 |
| Draws | 283 |
| Losses | 51 |
| Average steps | 433.8 |
| Own-bomb deaths | 37 |
| Enemy-bomb deaths | 11 |
| Unknown deaths | 3 |
| Deaths in corridor/dead-end | 51 |
| Draws near 500 steps | 282 |
| Draws with no bomb placed | 5 |
| Draws with stuck-loop signal | 5 |

This points to two dominant issues:

- draws usually reach the 500-step cap
- losses are mostly late bomb/corridor failures, not random early suicides

## Counterfactual Pass

I ran an ad hoc local counterfactual pass over recent HybridAgent draw/loss logs. No script was committed.

All HybridAgent recent draw/loss subset:

| Metric | Value |
|---|---:|
| Selected draw/loss logs | 39 |
| Sampled decision frames | 312 |
| Own-bomb late escape pattern hits | 225 |
| Corridor disadvantage under enemy pressure hits | 12 |

Exact pool subset with `HybridAgent`, `TacticalRuleAgent`, `SmarterRuleAgent`, `GeniusRuleAgent`:

| Metric | Value |
|---|---:|
| Selected draw/loss logs | 3 |
| Sampled decision frames | 24 |
| Own-bomb late escape pattern hits | 12 |
| Corridor disadvantage under enemy pressure hits | 4 |

Recent draw-only late-game action distribution over 40 near-500 draws:

| Action | Count |
|---|---:|
| STOP | 923 |
| LEFT | 597 |
| RIGHT | 584 |
| UP | 482 |
| DOWN | 493 |
| PLACE_BOMB | 81 |

Late draw rates:

- STOP rate: 29.2%
- PLACE_BOMB rate: 2.6%
- draw files with no bomb frames by us in the late sample: 1

This does not prove bomb spam is missing. It does show that the late draw state is conservative and bomb-sparse.

## Concrete Evidence

### Evidence 1: Own Bomb Escape Collapse Near Corner

File: `logs/json/match_20260523_192547_190390_none.json`

Aggregate analyzer:

- outcome: loss
- total steps: 500
- death step: 91
- death position: `(4, 1)`
- death source: own bomb
- pre-death zone: corridor/dead-end
- bomb nearby before death: yes

Counterfactual details:

| Step | Position | Chosen | Counterfactual best | Danger | Key state |
|---:|---|---|---|---:|---|
| 85 | `(2, 1)` | `DOWN` | `DOWN` | 6 | own bomb at `(2, 1)` timer 6 |
| 87 | `(3, 2)` | `UP` | `DOWN` | 9999 | best move had safe reach 17 vs chosen path toward lower-space corridor |
| 88 | `(3, 1)` | `STOP` | `RIGHT` | 3 | chosen cell dead-end, safe_reach 2 |
| 90 | `(4, 1)` | `STOP` | no valid safe alternative in simple model | 1 | death imminent |

Interpretation:

The error is not simply "placed bomb while unsafe." The escape path initially looked valid, but the agent drifted into a low-space corridor/dead-end after the bomb was active. The current safety layer checks immediate danger well, but the replay suggests escape-margin decay after bomb placement can still collapse.

### Evidence 2: Late Own Bomb Creates Dead-End Stall

File: `logs/json/match_20260524_160810_672740_none.json`

Aggregate analyzer:

- outcome: loss
- total steps: 500
- death step: 487
- death position: `(5, 6)`
- death source: own bomb
- pre-death zone: corridor/dead-end
- bomb nearby before death: yes

Counterfactual details:

| Step | Position | Chosen | Counterfactual best | Danger | Key state |
|---:|---|---|---|---:|---|
| 481 | `(5, 7)` | `UP` | `RIGHT` | 6 | own bomb at `(5, 7)` timer 6 |
| 482 | `(5, 6)` | `STOP` | `STOP` | 5 | cell dead-end, safe_reach 1 |
| 483 | `(5, 6)` | `STOP` | `STOP` | 4 | enemy at `(5, 4)`, dead-end nearby |
| 486 | `(5, 6)` | `STOP` | invalid by danger rule | 1 | no escape left |

Interpretation:

This is a repeated shape: bomb is legal at placement time, then the practical escape cell is a dead-end with only one safe reachable cell. The agent can pass `can_escape_after_bomb` and still later lose because the escape route is narrow and opponent bombs or movement block the remaining margin.

### Evidence 3: Combat Bomb in Corridor Against Nearby Enemy

File: `logs/json/match_20260524_202058_284886_none.json`

Counterfactual details:

| Step | Position | Chosen | Counterfactual best | Danger | Key state |
|---:|---|---|---|---:|---|
| 102 | `(11, 2)` | `PLACE_BOMB` | `PLACE_BOMB` | 6 | pressure 530, direct threat true, zone corridor |
| 103 | `(11, 2)` | `UP` | `UP` | 5 | moves to `(11, 1)`, dead-end |
| 105 | `(10, 1)` | `RIGHT` | `STOP` | 3 | both options score poorly, safe_reach 2 |
| 107 | `(11, 1)` | `STOP` | invalid by danger rule | 1 | death path locked |

Interpretation:

The bomb has real tactical pressure, so rejecting it globally would reduce conversion. The bottleneck is not "bombs are bad"; it is that some valid pressure bombs route our escape through dead-end geometry. This is why broad aggression changes are risky.

### Evidence 4: Draws Are Usually Not Pure Stuck Loops

Recent 40 draw logs near the 500-step cap showed:

- late STOP rate: 29.2%
- late PLACE_BOMB rate: 2.6%
- only 1 draw file had no bomb frames by us in the sampled late window
- several draw examples had high STOP counts, but aggregate stuck-loop signal from `analyze_matches.py` was only 5 out of 283 draws

Examples with heavy late STOP or loop signal:

| File | Last-80 STOP count | Last-20 unique positions | Bomb frames by us |
|---|---:|---:|---:|
| `match_20260524_192010_766312_none.json` | 79 | 1 | 41 |
| `match_20260524_192047_667800_none.json` | 79 | 1 | 33 |
| `match_20260524_192042_477697_none.json` | 47 | 1 | 87 |
| `match_20260524_192030_860658_none.json` | 22 | 6 | 0 |

Interpretation:

There is evidence of passive draw states, but not a clean repeated stuck-loop bug. Many draws include bombs, movement, and survival. The more credible issue is missed conversion under conservative thresholding, but the current evidence does not isolate a safe one-line policy change.

## Failure Pattern Ranking

### 1. Own-bomb escape margin collapses in corridor/dead-end

Evidence count:

- 37 of 51 losses attributed to own bomb
- 51 of 51 losses had pre-death corridor/dead-end signal
- 225 counterfactual sampled frames in recent losses matched own-bomb late escape pattern

This is the strongest bottleneck. It is also dangerous to fix aggressively because many successful kills rely on the same bomb pressure system.

### 2. Passive 500-step draw conversion gap

Evidence count:

- 282 of 283 draws reached near 500 steps
- late draw PLACE_BOMB rate was only 2.6% in the recent 40-draw sample
- late STOP rate was 29.2%

This is a real bottleneck, but the evidence is weaker for a code change because low bomb rate may be correct when safe meaningful bombs do not exist. Previous anti-draw/aggression variants already regressed online-style robustness.

### 3. Corridor disadvantage under enemy pressure

Evidence count:

- 12 counterfactual sampled frames in recent draw/loss subset
- 4 hits in exact local pool subset
- repeated examples show nearby enemies plus low-mobility escape routes after bomb placement

This is related to the top pattern. It suggests the issue is local geometry under enemy response, not global aggression.

## Missed-Conversion Evidence

The counterfactual pass did not find enough repeated high-confidence "agent should have bombed here and safely won" cases.

There were individual frames where `PLACE_BOMB` scored highly in the counterfactual model, for example:

- `match_20260524_160821_556323_none.json`, step 62: at `(3, 5)`, counterfactual `PLACE_BOMB` had pressure 1140 and was valid in the simplified model, while chosen action was `RIGHT`.
- `match_20260524_202042_300830_none.json`, step 95: at `(3, 7)`, counterfactual `PLACE_BOMB` had pressure 1600 and direct threat true, while chosen action was `UP`.

However, both are in loss contexts with enemy proximity and later corridor risk. They are not enough to justify lowering bomb thresholds or adding aggression globally.

## Evidence for Safe Code Change

Current evidence does not support a safe agent change yet.

What is supported:

- Keep using replay/counterfactual analysis.
- If future logs repeat the same pattern with more exact action-level proof, investigate a narrow post-bomb escape-margin audit.
- Any future change should be local and conservative: detect "escape route ends in dead-end with safe_reach <= 1 while enemy/bomb pressure exists" before committing to a pressure bomb.

What is not supported:

- global bomb pressure increase
- global STOP penalty
- global chase increase
- lowering `PLACE_BOMB` threshold broadly
- weakening `can_escape_after_bomb`
- replacing heuristic decisions with neural prior

## Recommendation

Keep `agent/hybrid_agent_online_robust` unchanged.

The strongest repeated bottleneck is own-bomb/corridor escape-margin collapse, but previous aggressive variants already showed that broad conversion tuning can damage stability. The current logs show a real leaderboard gap, but not a safe, repeated, isolated code change with enough evidence.

If no stronger repeated pattern appears in future online-style logs, online_robust is likely near local optimum; leaderboard gap is opponent/meta dependent.
