# Model Card: F1 Championship Forecasting

## Summary

This project forecasts a Formula 1 driver's final championship position and
reporting tier from historical season information. It is an educational,
research-oriented forecasting project, not a production decision system or a
betting recommendation.

The current saved regression model is selected by Spearman rank correlation on
the fixed 2020–2024 holdout. In the broader rolling-origin evaluation, the
previous-season points-rank baseline currently outperforms the machine-learning
models, so the model should not be described as superior to that baseline.

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

1. A fixed time split trains on 1950–2019 and evaluates on 2020–2024.
2. Rolling-origin evaluation retrains before each test season from 2015–2024,
   using only seasons earlier than the test season.
3. Regression cross-validation groups rows by season.
4. Tier classification uses grouped, stratified cross-validation and reports
   per-class F1 because the classes are imbalanced.
5. The previous-season points-rank and previous-season average-finish methods
   are included as transparent baselines.

## Reported results

Rolling-origin means across 2015–2024:

| Model / baseline | Mean RMSE | RMSE SD | Mean Spearman | Spearman SD |
|---|---:|---:|---:|---:|
| Baseline: previous points rank | 3.758 | 0.747 | 0.821 | 0.081 |
| Random Forest | 6.657 | 2.818 | 0.808 | 0.061 |
| Gradient Boosting | 6.888 | 2.747 | 0.788 | 0.057 |
| Baseline: previous avg finish | 4.564 | 0.875 | 0.783 | 0.085 |
| Ridge | 11.921 | 2.317 | 0.686 | 0.102 |

These values are regenerated with:

```bash
python scripts/evaluate.py
```

## Limitations and risks

- The dataset is a historical snapshot and may contain source corrections,
  missing values, or inconsistent historical reporting.
- New drivers and team changes have limited historical information; missing
  prior-season values are currently imputed with zero for model inputs.
- Constructor identity is based on the first observed current-season race for
  historical reconstruction, which is an approximation of pre-season context.
- The model does not provide calibrated uncertainty intervals.
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

The command writes a summary CSV and per-season detail CSV under `results/`;
these generated artifacts are intentionally ignored by Git.
