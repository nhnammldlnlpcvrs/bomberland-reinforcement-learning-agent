# Imitation Hyperparameter Sweep Report

This report ranks tiny imitation policies by behavior quality, not accuracy alone.

## All Runs

| Run | Dataset | CWP | Bomb Boost | STOP Weight | Val Acc | Top-2 | Entropy | STOP Pred | Bomb Pred | Move Div | Max Move Dir | Score | Warnings |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 19 | balanced | 1.0 | 1.0 | 1.0 | 11.7% | 37.2% | 0.986 | 0.7% | 53.5% | 50% | 28.6% | 0.119 | BOMB>12 |
| 10 | original | 1.0 | 1.0 | 0.8 | 19.6% | 35.2% | 0.999 | 0.0% | 0.0% | 75% | 52.8% | -0.046 | BOMB<2, UP>45 |
| 6 | original | 0.6 | 1.0 | 0.8 | 28.6% | 51.8% | 0.991 | 52.6% | 0.0% | 75% | 43.8% | -0.053 | STOP>35, BOMB<2 |
| 4 | original | 0.3 | 1.0 | 0.8 | 23.0% | 48.4% | 0.943 | 0.0% | 0.0% | 25% | 100.0% | -0.077 | BOMB<2, RIGHT>45 |
| 12 | balanced | 0.0 | 1.0 | 0.8 | 23.2% | 47.4% | 0.972 | 11.5% | 0.0% | 25% | 88.5% | -0.081 | BOMB<2, RIGHT>45 |
| 22 | wins_only | 0.0 | 1.0 | 0.8 | 30.3% | 46.9% | 0.976 | 25.4% | 0.0% | 50% | 73.3% | -0.085 | BOMB<2, RIGHT>45 |
| 28 | wins_only | 0.8 | 1.0 | 0.8 | 19.6% | 34.0% | 0.986 | 0.0% | 0.0% | 75% | 55.5% | -0.094 | BOMB<2, RIGHT>45 |
| 7 | original | 0.8 | 1.0 | 1.0 | 31.8% | 49.4% | 0.999 | 100.0% | 0.0% | 0% | 0.0% | -0.106 | STOP>35, BOMB<2 |
| 13 | balanced | 0.3 | 1.0 | 1.0 | 31.1% | 48.2% | 0.983 | 100.0% | 0.0% | 0% | 0.0% | -0.122 | STOP>35, BOMB<2 |
| 3 | original | 0.3 | 1.0 | 1.0 | 25.9% | 46.7% | 0.956 | 91.0% | 0.0% | 25% | 9.0% | -0.142 | STOP>35, BOMB<2 |
| 11 | balanced | 0.0 | 1.0 | 1.0 | 26.9% | 46.2% | 0.973 | 100.0% | 0.0% | 0% | 0.0% | -0.143 | STOP>35, BOMB<2 |
| 9 | original | 1.0 | 1.0 | 1.0 | 20.0% | 40.6% | 0.999 | 0.0% | 0.0% | 50% | 87.0% | -0.144 | BOMB<2, LEFT>45 |
| 1 | original | 0.0 | 1.0 | 1.0 | 26.7% | 46.5% | 0.948 | 83.1% | 0.0% | 25% | 16.9% | -0.146 | STOP>35, BOMB<2 |
| 24 | wins_only | 0.3 | 1.0 | 0.8 | 21.5% | 40.3% | 0.993 | 0.0% | 0.0% | 25% | 100.0% | -0.148 | BOMB<2, RIGHT>45 |
| 16 | balanced | 0.6 | 1.0 | 0.8 | 21.8% | 40.1% | 0.988 | 0.0% | 0.0% | 50% | 95.4% | -0.151 | BOMB<2, RIGHT>45 |
| 2 | original | 0.0 | 1.0 | 0.8 | 26.9% | 45.5% | 0.951 | 57.0% | 0.0% | 25% | 43.0% | -0.155 | STOP>35, BOMB<2 |
| 18 | balanced | 0.8 | 1.0 | 0.8 | 17.4% | 39.4% | 0.998 | 0.0% | 0.0% | 100% | 69.7% | -0.157 | BOMB<2, DOWN>45 |
| 30 | wins_only | 1.0 | 1.0 | 0.8 | 14.4% | 29.6% | 0.999 | 0.0% | 22.7% | 25% | 77.3% | -0.158 | BOMB>12, UP>45 |
| 29 | wins_only | 1.0 | 1.0 | 1.0 | 22.7% | 39.4% | 0.987 | 0.0% | 0.0% | 50% | 98.5% | -0.159 | BOMB<2, RIGHT>45 |
| 26 | wins_only | 0.6 | 1.0 | 0.8 | 21.3% | 38.6% | 0.996 | 0.0% | 0.0% | 25% | 100.0% | -0.165 | BOMB<2, RIGHT>45 |
| 15 | balanced | 0.6 | 1.0 | 1.0 | 27.1% | 41.8% | 0.998 | 84.1% | 0.0% | 50% | 11.2% | -0.182 | STOP>35, BOMB<2 |
| 8 | original | 0.8 | 1.0 | 0.8 | 20.3% | 36.7% | 0.998 | 0.0% | 0.0% | 25% | 100.0% | -0.184 | BOMB<2, RIGHT>45 |
| 14 | balanced | 0.3 | 1.0 | 0.8 | 17.8% | 37.2% | 0.958 | 0.0% | 0.0% | 50% | 82.4% | -0.187 | BOMB<2, RIGHT>45 |
| 5 | original | 0.6 | 1.0 | 1.0 | 26.4% | 42.1% | 0.961 | 97.6% | 0.0% | 25% | 1.7% | -0.187 | STOP>35, BOMB<2 |
| 20 | balanced | 1.0 | 1.0 | 0.8 | 15.9% | 31.1% | 0.986 | 0.0% | 0.2% | 50% | 62.6% | -0.188 | BOMB<2, UP>45 |
| 21 | wins_only | 0.0 | 1.0 | 1.0 | 23.0% | 41.6% | 0.976 | 93.6% | 0.0% | 50% | 3.7% | -0.189 | STOP>35, BOMB<2 |
| 27 | wins_only | 0.8 | 1.0 | 1.0 | 22.7% | 38.9% | 0.995 | 65.3% | 0.0% | 75% | 13.7% | -0.212 | STOP>35, BOMB<2 |
| 23 | wins_only | 0.3 | 1.0 | 1.0 | 21.0% | 38.1% | 0.980 | 85.6% | 0.0% | 50% | 8.8% | -0.223 | STOP>35, BOMB<2 |
| 17 | balanced | 0.8 | 1.0 | 1.0 | 14.9% | 32.3% | 0.999 | 0.0% | 0.0% | 75% | 70.7% | -0.227 | BOMB<2, DOWN>45 |
| 25 | wins_only | 0.6 | 1.0 | 1.0 | 24.0% | 40.1% | 0.975 | 35.9% | 0.0% | 75% | 59.9% | -0.252 | STOP>35, BOMB<2, RIGHT>45 |

## Top 5 Recommended Configs

| Rank | Run | Dataset | CWP | Bomb Boost | STOP Weight | Top-2 | Entropy | STOP Pred | Bomb Pred | Score | Why |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 19 | balanced | 1.0 | 1.0 | 1.0 | 37.2% | 0.986 | 0.7% | 53.5% | 0.119 | tradeoff: BOMB>12 |
| 2 | 10 | original | 1.0 | 1.0 | 0.8 | 35.2% | 0.999 | 0.0% | 0.0% | -0.046 | tradeoff: BOMB<2, UP>45 |
| 3 | 6 | original | 0.6 | 1.0 | 0.8 | 51.8% | 0.991 | 52.6% | 0.0% | -0.053 | tradeoff: STOP>35, BOMB<2 |
| 4 | 4 | original | 0.3 | 1.0 | 0.8 | 48.4% | 0.943 | 0.0% | 0.0% | -0.077 | tradeoff: BOMB<2, RIGHT>45 |
| 5 | 12 | balanced | 0.0 | 1.0 | 0.8 | 47.4% | 0.972 | 11.5% | 0.0% | -0.081 | tradeoff: BOMB<2, RIGHT>45 |

## Selected Config

Best run: `19` on `balanced`.

The selection score rewards top-2 accuracy and entropy, then penalizes STOP-heavy, no-bomb, bomb-heavy, and single-direction movement collapse. This keeps the chosen policy closer to a usable action-ranker for a future safety-filtered hybrid system.

No warning-free policy was found in this sweep. Treat the selected config as the least-bad research candidate, not a deployable policy.
