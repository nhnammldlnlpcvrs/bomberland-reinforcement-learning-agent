# Failure Analysis Playbook

## 1. Common Failure Modes

- Self bomb
- No escape after bomb
- Stuck loop
- Spawn camping
- Corridor trap
- Passive draw
- Over-aggressive chase
- Item greed
- Bad bomb timing

## 2. Symptoms and Metrics

- Average steps high
- Average rank bad
- Draw high
- Win low
- Death step distribution clustered early or after bomb placement

## 3. Fix Matrix

| Symptom | Likely Cause | Fix |
|---|---|---|
| self-bomb deaths | escape validation too weak | strengthen bomb escape and future survivability after bomb |
| draw near 500 | passive movement or low conversion | controlled expansion, better enemy pressure |
| stuck-loop signal | local repeat positions | loop breaker and expansion value |
| early deaths | unsafe move scoring | enforce danger map invariant |
| avg rank worse | over-aggression | reduce chase and bomb pressure |
| item greed | reward imbalance | cap item score and prefer safe territory |
| corridor deaths | low mobility choice | corridor/dead-end penalty |

