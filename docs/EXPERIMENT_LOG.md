# Experiment Log

Values marked `approx` are approximate and should be replaced with measured results when available.

| Version | Commit | Base | Local Score | Online Score | Win Rate | Draw Rate | Avg Rank | Avg Steps | Decision | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| online-best v1 | approx | early hybrid conservative | approx 133 local mean | approx 114.1 | approx 35% online | n/a | approx 0.95 online | approx 408 online | champion base | Better online than local-aggressive versions. |
| less_draw | 370a0ae / 17f7fcb approx | hybrid + late aggression | approx 135+ local single runs | approx 112.6 | approx 22% online | n/a | approx 1.16 online | approx 469 online | reject as champion | Local-strong but online weaker and longer games. |
| online_robust controlled expansion | 3509a8c | online-best v1 | approx 137.58 compare mean | pending | pending | approx 32.77% local compare | approx 0.04 local compare | approx 392 replay sample | submit trial only | Conservative expansion, loop breaker, territory value. |
| dataset pipeline initialized | pending | online_robust heuristic replays | n/a | n/a | n/a | n/a | n/a | n/a | tooling | Replay logs to imitation `.npz` dataset for future supervised policy learning. |
| tiny imitation baseline initialized | pending | replay imitation dataset | pending | n/a | n/a | n/a | n/a | n/a | research | Tiny CNN behavior-cloning baseline for dataset quality and feature learnability checks. |
| curated dataset pipeline | pending | imitation replay dataset | pending | n/a | n/a | n/a | n/a | n/a | research | Dataset curation for STOP reduction, bomb representation, endgame weighting, and episode caps. |
| policy bias analysis initialized | pending | tiny imitation checkpoint | pending | n/a | n/a | n/a | n/a | n/a | research | Policy entropy, prediction distribution, movement diversity, and STOP/PLACE_BOMB confusion analysis. |
| imitation hyperparameter sweep | pending | curated imitation datasets | pending | n/a | n/a | n/a | n/a | n/a | research | Sweeps class weighting and action-bias knobs using behavior-aware selection score. |
| neural-prior action ranking | pending | safe-action masked replay dataset | pending | n/a | n/a | n/a | n/a | n/a | research | Trains a CNN prior to rank heuristic-safe actions without deploying it into production. |
| symmetry augmentation | pending | imitation/action-ranking datasets | pending | n/a | n/a | n/a | n/a | n/a | research | Adds hflip, vflip, and rot180 augmentation with action remapping to reduce directional collapse. |
| behavior-aware action ranker | pending | symmetry-augmented safe-action ranking | n/a | n/a | n/a | n/a | n/a | n/a | research | Smoke: top-2 safe 82.2%, entropy 0.787, STOP 21.7%, bomb 6.7%, max direction 31.2%; still offline only. |
| shadow ranker evaluation | b2f5c95 | behavior-aware checkpoint + online_robust heuristic | n/a | n/a | n/a | n/a | n/a | n/a | research | **Critical finding:** model predicts BOMB 100% raw / 88% masked in live play vs 6.2% on static eval. Neural prior NOT ready for hybrid reranking. Script: `scripts/participant/shadow_ranker_eval.py`. |
| bomb-collapse root cause diagnosis | pending | behavior-aware checkpoint + augmented ranking dataset | n/a | n/a | n/a | n/a | n/a | n/a | research | **Root cause confirmed:** unconstrained BOMB logit. Training mask only includes BOMB when heuristic bombed (7.5% of samples), so BOMB logit drifts to +5.0 above other actions. Lightweight mask exposes BOMB 67% of time (8.9x vs training), revealing the inflated logit. NOT a feature encoding or schema bug. Script: `ml/diagnose_feature_shift.py`. |
