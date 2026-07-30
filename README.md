# f1-championship-prediction
# Predicting F1 Championship Final Standings 🏎️📊

A data science project where I use historical Formula 1 race data to **forecast the final Drivers’ Championship standings** for a season.

The goal is to explore how well machine learning models can approximate the final rankings using data like race results, qualifying performance, team strength, and driver consistency.

## Project Overview

This project focuses on:

- Collecting and cleaning **historical F1 data** (drivers, constructors, races, results)
- Engineering features that represent prior-season driver performance, constructor strength, qualifying pace, and reliability
- Training machine learning models to:
  - Predict **final championship position**
  - Predict **final position / tier** (e.g. champion, podium contender, midfield, backmarker)
- Evaluating how well we can **forecast a season's final standings** without using that season's race outcomes as predictors.

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/icecold009/f1-championship-prediction.git
cd f1-championship-prediction
pip install -r requirements.txt

# 2. Download the raw CSV inputs (kept out of Git)
python scripts/download_data.py

# 3. Run processing, training, prediction, visualisation, and HTML report
python main.py --year 2023 --report

# 4. Re-run the auditable rolling-origin evaluation independently
python scripts/evaluate.py
```

Predictions are saved to `results/2023_predictions.csv`; the chart and user-facing report are saved to
`results/predicted_vs_actual_2023.png` and `results/f1_prediction_report_2023.html`. Processed data
and result files are regenerated locally and ignored by Git. The evaluation command writes summary
and per-season detail CSVs under `results/`. See [MODEL_CARD.md](MODEL_CARD.md) for intended use,
limitations, and the evaluation protocol.

For a validated release bundle with provenance metadata, run
`python scripts/build_release.py --download --year 2023`. See [RELEASE.md](RELEASE.md) for the
local checklist and the manual GitHub Actions artifact workflow.

## Tests

Run the unit tests locally with:

```bash
python -m pytest -q
```

The same test command runs in GitHub Actions for Python 3.12 and 3.13.

## Problem Framing

There is one primary prediction task and one derived reporting task:

1. **Regression task**  
   Predict the **final championship position** for each driver. The numeric
   output is a predicted position, not predicted championship points.

2. **Ranking / classification task**  
   Predict a driver’s **final standing (or tier)**, for example:
   - World Champion  
   - Top 3  
   - Top 5 / Top 10  
   - Midfield  
   - Backmarker  

The final output is a **predicted ordered list of drivers**, which can be compared to the actual final standings.

## Data

The dataset includes (per season and per driver):

- Driver information (name, team, nationality, experience)  
- Constructor / team information  
- Race-by-race results:
  - Finishing position  
  - Points scored  
  - Grid position  
  - DNFs / DNS  
- Historical season-level aggregates:
  - Total races started  
  - Average finish position  
  - Average grid position  
  - Points per race  
  - Podiums, wins, poles, fastest laps  

The raw CSVs are downloaded from the public [Formula 1 Race Data Kaggle dataset](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data), which uses the Ergast-compatible table layout expected by this project. The local snapshot used for the reported results was pulled on **2025-01-29**, contains seasons through **2024**, and is refreshed with `python scripts/download_data.py`. Ergast’s public API was retired after the 2024 season; the linked Kaggle dataset preserves the compatible table structure while providing an auditable source for this snapshot.

The project uses this dataset as its raw-data source; the local `data/raw/` files are generated inputs and are intentionally not tracked.

## Features & Approach

Key feature engineering ideas. All predictors are available before the target
season's championship outcome is known:

### Performance metrics

- Prior-season average finish position
- Prior-season average grid position
- Prior-season grid-to-finish delta (racecraft)
- Prior-season win / podium / points-scoring rates

### Reliability

- Prior-season number and percentage of DNFs
- Prior-season races started

### Team strength

- Prior constructor championship points
- Prior constructor championship position

The current season's race results are not used as predictors. The season-opening
constructor is used to join its prior final championship strength, while the
final driver standings are retained only as evaluation targets.

## Model Results

Every reported metric uses the same leak-free walk-forward protocol. For each
test season **N+1**, the model is trained on all available seasons through **N**;
all rows from the test season remain together and never appear in training.
The reported values are the mean and standard deviation across ten chronological
test seasons, 2015–2024. RMSE measures position error, R² measures explained
position variance, and Spearman ρ measures agreement in predicted order.

The primary naïve baseline is **previous-season final order**: rank the current
season's entrants by their prior-season championship points, assigning drivers
without prior history zero points. It uses no fitted model and represents the
simple pre-season guess that last season's order will persist. The delta column
is each row's Spearman ρ minus that naïve baseline on the same test season.

| Model / baseline | Test seasons | Mean RMSE | RMSE SD | Mean R² | R² SD | Mean Spearman ρ | Spearman SD | Mean Δ vs naïve | Δ SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive: previous-season final order | 10 | 3.758 | 0.747 | 0.641 | 0.161 | 0.821 | 0.081 | 0.000 | 0.000 |
| Random Forest | 10 | 6.657 | 2.818 | -0.277 | 1.042 | 0.808 | 0.061 | -0.012 | 0.065 |
| Gradient Boosting | 10 | 6.888 | 2.747 | -0.350 | 1.060 | 0.788 | 0.057 | -0.033 | 0.068 |
| Baseline: previous avg finish | 10 | 4.564 | 0.875 | 0.462 | 0.273 | 0.783 | 0.085 | -0.038 | 0.031 |
| Ridge | 10 | 11.921 | 2.317 | -2.648 | 1.729 | 0.686 | 0.102 | -0.134 | 0.125 |

The naïve previous-season-order baseline remains stronger than every machine
learning regressor in this walk-forward evaluation. Random Forest is retained as
the predeclared operational model for the generated forecast artifact; it is not
described as the best model based on these reported test seasons.

### Tier classification

The tier classifier is also retrained before each test season. Accuracy is the
fraction of correctly classified drivers, while macro F1 gives each tier equal
weight. The previous ambiguous “CV Accuracy” value has been removed; these are
walk-forward test-season metrics only.

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

The complete per-season values are regenerated by `python scripts/evaluate.py`;
the training cutoff for every row is retained in
`results/rolling_origin_summary_details.csv` and
`results/tier_rolling_origin_summary_details.csv`.

### Prediction uncertainty

The user-facing forecast also fits **100 season-level bootstrap Random Forests**
using only seasons before the forecast year. For every driver,
`results/2023_predictions.csv` and the HTML report include:

- `Champion Probability`, `Top 3 Probability`, and `Top 5 Probability`: the
  fraction of bootstrap rankings placing that driver in each outcome set.
- `Bootstrap Position P05` and `Bootstrap Position P95`: the central 90%
  interval of predicted positions.
- `Bootstrap Position SD`: the spread of predicted positions.

These are empirical model-uncertainty estimates, not calibrated betting odds or
guaranteed real-world probabilities. The full per-driver distribution summary is
shown in the generated report; the number of bootstrap runs is recorded in the
prediction CSV.

For the regenerated 2023 artifact, the leading point-forecast rows are:

| Driver | Point rank | Champion | Top 3 | Position P05–P95 |
|---|---:|---:|---:|---:|
| Max Verstappen | 1 | 99% | 100% | 1.88–3.73 |
| Carlos Sainz | 2 | 0% | 44% | 3.92–7.39 |
| Charles Leclerc | 3 | 1% | 65% | 3.57–7.01 |
| Sergio Pérez | 4 | 0% | 64% | 3.98–6.41 |

The CSV and HTML report include the same uncertainty fields for every driver.

---

## Where this model breaks

This section uses the same leak-free Random Forest walk-forward predictions and
adds post-hoc diagnostic labels. A positive signed error means the model placed
a driver lower than their actual final position. The labels are not predictors.

### Driver and season type

| Diagnostic group | Observations | RMSE | MAE | Mean signed error |
|---|---:|---:|---:|---:|
| Returning | 179 | 4.760 | 3.216 | 0.619 |
| Rookie | 33 | 10.825 | 8.241 | 5.186 |
| Returning after gap | 11 | 17.798 | 14.396 | 11.937 |
| No mid-season swap | 216 | 7.195 | 4.501 | 1.849 |
| Mid-season swap | 7 | 5.676 | 4.823 | 1.989 |
| Other test seasons | 201 | 7.328 | 4.554 | 1.851 |
| 2022 regulation-change season | 22 | 5.277 | 4.123 | 1.875 |

The clearest failure mode is missing history: returning drivers after a gap have
**14.396 MAE**, while ordinary returning drivers have **3.216 MAE**. Rookies
also have much larger error (**8.241 MAE**) than established returners. Mid-season
swaps are slightly worse than non-swaps, but the sample is only seven driver-
seasons, so that result is not conclusive.

### Worst chronological test seasons

| Test season | Train through | RMSE | MAE | Spearman ρ |
|---:|---:|---:|---:|---:|
| 2015 | 2014 | 11.257 | 7.503 | 0.728 |
| 2016 | 2015 | 10.470 | 6.028 | 0.867 |
| 2019 | 2018 | 8.562 | 6.006 | 0.760 |
| 2020 | 2019 | 5.061 | 4.245 | 0.755 |
| 2022 | 2021 | 5.277 | 4.123 | 0.848 |

The 2022 regulation-change case is a useful stress test, but it is not the
worst season by MAE in this sample. That is the honest conclusion: the model
shows a positive **1.875** mean signed error in 2022, but the largest failures
are concentrated in earlier seasons and in drivers with weak or missing history.

The underlying per-driver errors, season summaries, and group summaries are
regenerated by `python scripts/error_analysis.py` into
`results/error_analysis_driver.csv`,
`results/error_analysis_season_summary.csv`, and
`results/error_analysis_group_summary.csv`.

---

## Feature Importance (Predeclared Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | prev_season_races_started | 0.508173 |
| 2 | prev_team_final_position | 0.133599 |
| 3 | prev_season_avg_grid_pos | 0.073407 |
| 4 | prev_season_avg_finish_pos | 0.055272 |
| 5 | prev_season_quali_to_race_delta | 0.055053 |
| 6 | prev_season_points_sum | 0.052092 |
| 7 | prev_team_final_points | 0.052020 |
| 8 | prev_season_std_finish_pos | 0.032530 |
| 9 | prev_season_points_per_race | 0.018175 |
| 10 | prev_season_dnf_rate | 0.012942 |

## Tech Stack

- **Language:** Python 3.x  
- **Data:** CSV files (`pandas` DataFrames)  
- **Core libraries:**
  - `pandas`, `numpy`  
  - `scikit-learn`  
  - `matplotlib`, `seaborn` (visualization)  
