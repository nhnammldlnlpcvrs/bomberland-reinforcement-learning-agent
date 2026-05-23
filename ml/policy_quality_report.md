# Policy Quality Report

This report tracks imitation-policy behavior beyond accuracy. The model is still research-only and must not be deployed without deterministic action masks and the production safety filter.

## Dataset Comparisons

The initial replay dataset is useful for verifying the pipeline, but it is draw-heavy and under-represents `PLACE_BOMB`. Curated datasets should preserve safe movement while reducing passive survival bias.

Recommended comparisons:

| Dataset | Purpose | Expected Risk |
|---|---|---|
| original | baseline learnability check | passive STOP bias and weak bomb learning |
| wins-only | higher quality teacher signal | may overfit winning states |
| balanced | reduce STOP dominance and boost meaningful bomb examples | duplicated bomb samples can overcorrect |

Smoke curation results from the current replay dataset:

| Dataset | Samples | STOP | PLACE_BOMB | Win | Draw | Avg Step |
|---|---:|---:|---:|---:|---:|---:|
| original | 51,800 | 26.5% | 1.6% | 52.4% | 47.3% | 219.5 |
| balanced curated | 37,613 | 25.1% | 6.6% | 58.0% | 41.5% | 208.7 |
| wins-only curated | 21,982 | 22.4% | 6.6% | 100.0% | 0.0% | 177.8 |

## Prediction Bias

Accuracy alone is misleading because a passive model can score well by predicting common survival actions. Track prediction distribution, confusion matrix, entropy, and movement diversity.

Policy bias warnings to watch:

- STOP predicted too often.
- `PLACE_BOMB` nearly absent.
- One movement direction dominates.
- Entropy is too low.
- Prediction distribution collapses to one action.

## Bomb Usage

The first dataset showed `PLACE_BOMB` around 1-2% of samples. That is too sparse for a supervised CNN to learn bomb timing. The curation tool targets a conservative 5-10% bomb ratio without forcing bomb spam.

## STOP Dominance

STOP is valid and sometimes necessary, so it should not be removed. The curation tool keeps STOP samples but caps excessive STOP representation to reduce passive survival cloning.

## Entropy Metrics

Policy entropy helps detect collapse even when accuracy improves. A useful early model should keep enough entropy to represent multiple plausible safe actions, especially movement choices.

Smoke policy-bias findings:

| Training Input | Val Acc | Top-2 Acc | Pred STOP | Pred PLACE_BOMB | Entropy | Main Issue |
|---|---:|---:|---:|---:|---:|---|
| original subset | 10.8% | 26.9% | 1.0% | 40.0% | 1.790 | vertical movement collapse and bomb overprediction |
| wins-only curated subset | 22.0% | 40.3% | 9.6% | 38.0% | 1.743 | best diversity so far, but bomb prediction is too high |
| balanced curated subset | 17.1% | 42.1% | 6.6% | 49.9% | 1.768 | bomb overprediction and horizontal collapse |
| balanced conservative weights | 26.7% | 45.2% | 50.1% | 0.0% | 1.737 | passive/no-bomb collapse |

The current tiny CNN is learnable enough to expose bias, but not yet a useful policy. The main control knob is class weighting: full inverse-frequency weighting overcorrects toward bombs, while reduced bomb weighting can erase bomb predictions.

## Recommended Dataset Strategy

Use the balanced curated dataset for the next imitation experiments, then compare against the original dataset with `ml/analyze_policy_bias.py`. Prefer the dataset that improves bomb representation and movement diversity without producing a low-entropy or bomb-spam policy.

Next recommended training run:

- Use the wins-only curated dataset as the starting point.
- Keep `PLACE_BOMB` in the 5-10% dataset range.
- Sweep `class_weight_power` between 0.6 and 0.9.
- Keep `bomb_boost_weight` near 0.8-1.0.
- Track entropy and prediction distribution before considering larger models.
