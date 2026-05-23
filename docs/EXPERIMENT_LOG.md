# Experiment Log

Values marked `approx` are approximate and should be replaced with measured results when available.

| Version | Commit | Base | Local Score | Online Score | Win Rate | Draw Rate | Avg Rank | Avg Steps | Decision | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| online-best v1 | approx | early hybrid conservative | approx 133 local mean | approx 114.1 | approx 35% online | n/a | approx 0.95 online | approx 408 online | champion base | Better online than local-aggressive versions. |
| less_draw | 370a0ae / 17f7fcb approx | hybrid + late aggression | approx 135+ local single runs | approx 112.6 | approx 22% online | n/a | approx 1.16 online | approx 469 online | reject as champion | Local-strong but online weaker and longer games. |
| online_robust controlled expansion | 3509a8c | online-best v1 | approx 137.58 compare mean | pending | pending | approx 32.77% local compare | approx 0.04 local compare | approx 392 replay sample | submit trial only | Conservative expansion, loop breaker, territory value. |
| dataset pipeline initialized | pending | online_robust heuristic replays | n/a | n/a | n/a | n/a | n/a | n/a | tooling | Replay logs to imitation `.npz` dataset for future supervised policy learning. |
