# Model Card: F1 Championship Forecasting

## Summary

This project forecasts a Formula 1 driver's final championship position and
reporting tier from historical season information. It is an educational,
research-oriented forecasting project, not a production decision system or a
betting recommendation.

The current saved regression model is a predeclared Random Forest used for the
user-facing forecast artifact. In the leak-free rolling-origin evaluation, the
naive previous-season final-order baseline currently outperforms the machine-learning
models, so the Random Forest should not be described as superior to that
baseline.

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
- Local snapshot: pulled 2025-01-29, covering seasons through 2024.
- Raw CSVs are downloaded locally and are not committed to Git.

## Prediction target and features

Each row represents a driver entering a season. The target is that driver's
final championship position and derived tier. Predictors contain prior-season
driver statistics and the prior final championship position and points of the
constructor they enter with.

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
| Random Forest | 10 | 6.657 | 2.818 | -0.277 | 1.042 | 0.808 | 0.061 | -0.012 | 0.065 |
| Gradient Boosting | 10 | 6.888 | 2.747 | -0.350 | 1.060 | 0.788 | 0.057 | -0.033 | 0.068 |
| Baseline: previous avg finish | 10 | 4.564 | 0.875 | 0.462 | 0.273 | 0.783 | 0.085 | -0.038 | 0.031 |
| Ridge | 10 | 11.921 | 2.317 | -2.648 | 1.729 | 0.686 | 0.102 | -0.134 | 0.125 |

These values are regenerated with:

```bash
python scripts/evaluate.py
```

The naive baseline ranks each test-season entrant by prior-season championship
points, assigning zero to drivers without prior history. The delta columns are
computed against that same-season baseline before aggregation.

Tier classification walk-forward metrics:

| Model | Test seasons | Mean accuracy | Accuracy SD | Mean macro F1 | Macro F1 SD |
|---|---:|---:|---:|---:|---:|
| Random Forest | 10 | 0.516 | 0.094 | 0.415 | 0.096 |

| Tier | Test seasons | Mean F1 | F1 SD |
|---|---:|---:|---:|
| Champion | 10 | 0.700 | 0.292 |
| Podium | 10 | 0.174 | 0.283 |
| Top 5 | 10 | 0.117 | 0.249 |
| Top 10 | 10 | 0.542 | 0.126 |
| Midfield | 10 | 0.273 | 0.214 |
| Backmarker | 10 | 0.684 | 0.138 |

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

## Where this model breaks

The walk-forward Random Forest errors are also segmented post-hoc by driver and
season context. Returning drivers with no immediately prior history have the
largest average error (MAE 14.396 across 11 observations), followed by rookies
(MAE 8.241 across 33 observations). Established returning drivers are materially
more predictable (MAE 3.216 across 179 observations). Mid-season constructor
swaps have MAE 4.823 across only seven observations, so that comparison is
directional rather than conclusive.

The 2022 regulation-change case study has RMSE 5.277, MAE 4.123, and Spearman
0.848. It is not the worst test season in this sample; 2015 has the largest MAE
at 7.503. This prevents the analysis from turning a plausible regulation-change
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
  intervals with guaranteed coverage.
- The evaluation covers historical seasons and should not be treated as proof
  of performance on a future regulation era.

## Reproducibility

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/evaluate.py --rebuild-features
python main.py --year 2023 --visualise
python -m pytest -q
```

The command writes regression and tier summary CSVs plus per-season detail files
under `results/`; these generated artifacts are intentionally ignored by Git.
