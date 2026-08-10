# Model Card: F1 Championship Forecasting

## Summary

This project forecasts a Formula 1 driver's final championship position and
reporting tier from historical season information. It is an educational,
research-oriented forecasting project, not a production decision system or a
betting recommendation.

The current saved regression model is a predeclared Random Forest with explicit
pre-season cold-start flags. In the leak-free rolling-origin evaluation, the
naive previous-season final-order baseline currently outperforms the
machine-learning models, so the Random Forest is not described as superior to
that baseline.

## Intended use

- Compare ranking and regression methods on historical F1 seasons.
- Explore how prior driver and constructor performance relates to the next
  season's championship outcome.
- Reproduce the evaluation with the pinned Python dependencies and downloaded
  raw data.

## Out-of-scope use

- Live race-by-race prediction.
- Betting, financial, employment, or safety-critical decisions.
- Claims about future seasons whose entrant or constructor context differs
  materially from the historical data.

## Data

- Source: [Formula 1 Race Data Kaggle dataset](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data)
- Layout: Ergast-compatible CSV tables.
- Local snapshot: project-recorded pull date 2025-01-29, covering seasons through
  2024.
- Raw CSVs are downloaded locally and are not committed to Git.
- `data/raw/data_manifest.json` records SHA-256 and byte size for every raw
  table. The release manifest copies those identifiers so a result can be tied
  to exact input bytes even when the original archive digest is unavailable.

## Prediction target and features

Each row represents a driver entering a season. The target is that driver's
final championship position and derived tier. Predictors contain prior-season
driver statistics and the prior final championship position and points of the
constructor they enter with. Four pre-season-safe cold-start indicators identify
rookies, drivers returning after a gap, missing driver history, and missing
constructor history.

Same-season race outcomes are not predictors. The current season is used only
to identify entrants and construct the final evaluation target. For historical
rows, the season-opening constructor is the first constructor observed in the
race data; a production forecaster should replace this with a pre-season entry
list.

## Evaluation protocol

1. For each test season from 2015–2024, training uses all seasons earlier than
   that test season.
2. Every test season is held out in full; no rows from the test season appear in
   training, and no random row split is used.
3. Regression reports per-season RMSE, R², and Spearman rank correlation before
   aggregating their means and standard deviations.
4. Tier classification reports per-season accuracy, macro F1, and per-class F1
   because the classes are imbalanced.
5. The naive previous-season final-order and previous-season average-finish
   methods are included as transparent baselines.

## Reported results

Walk-forward means across 2015–2024:

| Model / baseline | Test seasons | Mean RMSE | RMSE SD | Mean R² | R² SD | Mean Spearman | Spearman SD | Mean Δ vs naive | Δ SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive: previous-season final order | 10 | 3.758 | 0.747 | 0.641 | 0.161 | 0.821 | 0.081 | 0.000 | 0.000 |
| Random Forest (history only) | 10 | 6.657 | 2.818 | -0.277 | 1.042 | 0.808 | 0.061 | -0.012 | 0.065 |
| Random Forest + cold-start flags | 10 | 6.531 | 2.704 | -0.217 | 0.998 | 0.807 | 0.051 | -0.014 | 0.055 |
| Gradient Boosting | 10 | 6.938 | 2.630 | -0.346 | 1.028 | 0.784 | 0.058 | -0.036 | 0.084 |
| Baseline: previous avg finish | 10 | 4.564 | 0.875 | 0.462 | 0.273 | 0.783 | 0.085 | -0.038 | 0.031 |
| Ridge | 10 | 9.517 | 2.178 | -1.359 | 1.235 | 0.777 | 0.057 | -0.043 | 0.066 |

These values are regenerated with:

```bash
python scripts/evaluate.py
```

The naive baseline ranks each test-season entrant by prior-season championship
points, assigning zero to drivers without prior history. The delta columns are
computed against that same-season baseline before aggregation.

Paired season-level bootstrap intervals show that the history-only Random
Forest's mean Spearman difference versus the naive baseline is -0.012 with a
95% interval of [-0.051, 0.026]. It wins five seasons and loses five on
Spearman, but loses nine of ten seasons on RMSE. The cold-start model improves
mean RMSE by 0.126 positions relative to the history-only forest while slightly
reducing mean Spearman. These mixed results are treated as an ablation, not a
claim of general improvement.

Tier classification walk-forward metrics. The mean macro F1 headline is
**0.441**, but it should not be read as uniform usefulness across tiers. Podium
F1 **0.174** and Top 5 F1 **0.207** are close to unusable for individual driver
classification. This is a 200-tree Random Forest with `max_depth=8`, trained on
`FEATURE_COLUMNS` from `src/model.py`; the weak results likely reflect class
imbalance across the six tiers and/or overlapping feature distributions between
adjacent tiers. Champion F1 **0.800** and Backmarker F1 **0.681** are the tiers
where the classifier is genuinely useful for individual driver classification.

| Model | Test seasons | Mean accuracy | Accuracy SD | Mean macro F1 | Macro F1 SD |
|---|---:|---:|---:|---:|---:|
| Random Forest | 10 | 0.509 | 0.072 | 0.441 | 0.073 |

| Tier | Test seasons | Mean F1 | F1 SD |
|---|---:|---:|---:|
| Champion | 10 | 0.800 | 0.322 |
| Podium | 10 | 0.174 | 0.283 |
| Top 5 | 10 | 0.207 | 0.274 |
| Top 10 | 10 | 0.511 | 0.143 |
| Midfield | 10 | 0.271 | 0.193 |
| Backmarker | 10 | 0.681 | 0.120 |

## Uncertainty estimates

The user-facing forecast fits 100 season-level bootstrap Random Forests. Each
bootstrap samples historical seasons with replacement, never mixing a forecast
season into training. For each driver, the prediction artifact reports the
empirical frequency of finishing champion, in the top three, and in the top five
across those bootstrap rankings, together with a position standard deviation and
P05–P95 interval.

These values quantify uncertainty from the historical training sample. They are
not calibrated probabilities, prediction intervals with guaranteed coverage, or
betting odds; the model does not claim that a 62% bootstrap frequency equals a
62% real-world chance.

Historical calibration confirms this limitation. Across 223 driver-seasons, the
bootstrap P05–P95 interval covers the actual position only 32.3% of the time,
with mean width 4.92 positions. Rolling conformal intervals use only prior
out-of-fold residuals and reach 96.4% coverage, but their mean width is 19.68
positions. The top-three and champion Brier scores are 0.068 and 0.019,
respectively. The 80–100% top-three bin is overconfident: mean prediction 95.4%
versus 77.3% observed.

## Where this model breaks

The walk-forward Random Forest errors are also segmented post-hoc by driver and
season context. Returning drivers with no immediately prior history have the
largest average error (MAE 12.443 across 11 observations), followed by rookies
(MAE 8.571 across 33 observations). Established returning drivers are materially
more predictable (MAE 3.198 across 179 observations). Mid-season constructor
swaps have MAE 5.416 across only seven observations, so that comparison is
directional rather than conclusive.

The 2022 regulation-change case study has RMSE 5.209, MAE 4.117, and Spearman
0.843. It is not the worst test season in this sample; 2015 has the largest MAE
at 7.697. This prevents the analysis from turning a plausible regulation-change
story into an unsupported claim. See the generated error-analysis CSVs for the
complete per-driver and per-season evidence.

## Limitations and risks

- The dataset is a historical snapshot and may contain source corrections,
  missing values, or inconsistent historical reporting.
- New drivers and team changes have limited historical information; missing
  prior-season values are currently imputed with zero for model inputs.
- Constructor identity is based on the first observed current-season race for
  historical reconstruction, which is an approximation of pre-season context.
- Bootstrap position intervals are empirical and are not calibrated uncertainty
  intervals with guaranteed coverage; their observed 32.3% coverage is far
  below the nominal 90%.
- Conformal intervals restore historical coverage but are too wide to provide
  precise driver-level position forecasts.
- The evaluation covers historical seasons and should not be treated as proof
  of performance on a future regulation era.

## Reproducibility

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/evaluate.py --rebuild-features
python scripts/model_audit.py
python main.py --year 2023 --visualise
python -m pytest -q --cov
```

The commands write regression, paired baseline, uncertainty calibration,
permutation importance, tier, and error-analysis artifacts under `results/`.
Generated working artifacts remain ignored; a validated reviewer-facing example
is published under `docs/`.
