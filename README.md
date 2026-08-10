# F1 Championship Forecasting

[![CI](https://github.com/icecold009/f1-championship-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/icecold009/f1-championship-prediction/actions/workflows/ci.yml)
[![Python 3.12–3.13](https://img.shields.io/badge/python-3.12%E2%80%933.13-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A leakage-safe study of whether historical driver and constructor performance can
forecast the final Formula 1 Drivers’ Championship order.

## Executive finding

The strongest result is a negative one: a naïve “repeat last season’s order”
baseline remains stronger overall than the fitted models. Across ten untouched
future seasons (2016–2025), the baseline reaches mean Spearman ρ **0.821**,
compared with **0.807** for the history-only Random Forest and **0.811** after
adding cold-start flags. The history-only forest wins four seasons and loses
six, but its paired 95% interval for mean Spearman improvement spans
**−0.057 to +0.029**. On RMSE, it loses eight of ten seasons.

That makes this project an exercise in trustworthy forecasting rather than an
algorithm leaderboard: every season is held out intact, uncertainty is
backtested, failure modes are segmented, and the exact raw snapshot is
identified by checksums.

**[Open the validated example report](docs/index.html)**
The report contains the forecast, baseline comparison, uncertainty calibration,
error analysis, and held-out permutation importance.

## Project Overview

This project focuses on:

- Collecting and cleaning **historical F1 data** (drivers, constructors, races, results)
- Engineering features that represent prior-season driver performance, constructor strength, grid/race pace, and reliability
- Training machine learning models to:
  - Predict **final championship position**
  - Predict **final position / tier** (e.g. champion, podium contender, midfield, backmarker)
- Evaluating how well we can **forecast a season's final standings** without using that season's race outcomes as predictors.

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/icecold009/f1-championship-prediction.git
cd f1-championship-prediction
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. Download the raw CSV inputs (kept out of Git)
python scripts/download_data.py

# 3. Run processing, training, prediction, visualisation, and HTML report
python main.py --year 2023 --report

# 4. Re-run the auditable rolling-origin evaluation independently
python scripts/evaluate.py

# 5. Run the slower calibration and interpretation audit
python scripts/model_audit.py
```

Predictions are saved to `results/2023_predictions.csv`; the chart and user-facing report are saved to
`results/predicted_vs_actual_2023.png` and `results/f1_prediction_report_2023.html`. Processed data
and result files are regenerated locally and ignored by Git. The evaluation command writes summary
and per-season detail CSVs under `results/`. See [MODEL_CARD.md](MODEL_CARD.md) for intended use,
limitations, and the evaluation protocol.

For a quick local release bundle, run
`python scripts/build_release.py --download --year 2023` and validate it with
`python scripts/check_release.py --quick --year 2023`. For a validated bundle
including the historical calibration and interpretation audit, add
`--full-audit` and omit `--quick`. See [RELEASE.md](RELEASE.md) for the local
checklist and the manual GitHub Actions artifact workflow.

## Tests

Run the unit tests locally with:

```bash
pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q --cov --cov-report=term-missing
```

The same lint, format, test, and coverage gates run in GitHub Actions for
Python 3.12 and 3.13.

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
  - DNFs, lapped finishes, and DNS
- Historical season-level aggregates:
  - Total races started  
  - Average finish position  
  - Average grid position  
  - Points per race  
  - Podiums, wins, poles, fastest laps  

The raw CSVs are downloaded from the public [Formula 1 Race Data Kaggle dataset](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data), which uses the Ergast-compatible table layout expected by this project. The current snapshot's pull time, archive SHA-256, table hashes, and byte sizes are recorded in `data/raw/data_manifest.json`; `results/release_manifest.json` carries those identifiers into every validated build. The downloader also rejects materially truncated tables before replacing the existing snapshot. A report must not be described as reproducible from an archive when its manifest records unknown provenance.

Ergast’s public API was retired after the 2024 season; the linked Kaggle dataset
preserves the compatible table structure. Running
`python scripts/download_data.py` fetches a new snapshot, validates required
columns and minimum content counts, and records the archive digest and response
metadata when available.

The project uses this dataset as its raw-data source; the local `data/raw/` files are generated inputs and are intentionally not tracked.
If the source includes an in-progress latest season, its entrants remain visible
with unknown final targets and are excluded from walk-forward scoring until all
scheduled races have result rows.

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
- Prior-season sprint points, included in the points-based baseline

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
test seasons, 2016–2025. RMSE measures position error, R² measures explained
position variance, and Spearman ρ measures agreement in predicted order.

The primary naïve baseline is **previous-season final order**: rank the current
season's entrants by their prior-season race and sprint championship points,
assigning drivers without prior history zero points. It uses no fitted model and
represents the simple pre-season guess that last season's order will persist. The delta column
is each row's Spearman ρ minus that naïve baseline on the same test season.

| Model / baseline | Test seasons | Mean RMSE | RMSE SD | Mean R² | R² SD | Mean Spearman ρ | Spearman SD | Mean Δ vs naïve | Δ SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive: previous-season final order | 10 | 3.728 | 0.744 | 0.641 | 0.168 | 0.821 | 0.084 | 0.000 | 0.000 |
| Random Forest (history only) | 10 | 6.189 | 2.301 | -0.071 | 0.823 | 0.807 | 0.058 | -0.014 | 0.074 |
| Random Forest + cold-start flags | 10 | 5.982 | 1.969 | 0.026 | 0.668 | 0.811 | 0.045 | -0.010 | 0.065 |
| Gradient Boosting | 10 | 6.302 | 2.087 | -0.089 | 0.754 | 0.757 | 0.095 | -0.064 | 0.113 |
| Baseline: previous avg finish | 10 | 4.543 | 0.706 | 0.466 | 0.247 | 0.788 | 0.073 | -0.033 | 0.025 |
| Ridge | 10 | 9.081 | 1.572 | -1.135 | 1.018 | 0.734 | 0.090 | -0.087 | 0.061 |

The naïve previous-season-order baseline remains stronger than every machine
learning regressor in this walk-forward evaluation. The cold-start flags reduce
Random Forest RMSE by **0.207 positions** and improve mean rank
correlation by **0.004** in this refresh. That mixed ablation result is retained rather than presenting the
new features as a blanket improvement.

### Paired evidence against the naïve baseline

The confidence intervals below resample whole test seasons, preserving the
paired comparison. A Spearman win means the method has higher rank correlation
than the naïve baseline in that season; an RMSE win means lower error.

| Method | Mean Spearman Δ | Paired 95% CI | Spearman W–L | Mean RMSE Δ | RMSE W–L |
|---|---:|---:|---:|---:|---:|
| Random Forest (history only) | -0.014 | [-0.057, 0.029] | 4–6 | +2.460 | 2–8 |
| Random Forest + cold-start flags | -0.010 | [-0.047, 0.028] | 4–6 | +2.254 | 2–8 |
| Gradient Boosting | -0.064 | [-0.132, -0.000] | 2–8 | +2.574 | 1–9 |
| Previous average finish | -0.033 | [-0.047, -0.019] | 1–9 | +0.815 | 0–10 |
| Ridge | -0.087 | [-0.123, -0.051] | 0–10 | +5.353 | 0–10 |

![Per-season model performance versus the naïve baseline](docs/model_vs_naive_by_season.png)

### Tier classification

The tier classifier is also retrained before each test season. Accuracy is the
fraction of correctly classified drivers, while macro F1 gives each tier equal
weight. The previous ambiguous “CV Accuracy” value has been removed; these are
walk-forward test-season metrics only. The mean macro F1 headline is **0.414**,
but it should not be read as uniform usefulness across tiers. Podium F1 **0.183**
and Top 5 F1 **0.217** are close to unusable for individual driver
classification. This is a 200-tree Random Forest with `max_depth=8`, trained on
`FEATURE_COLUMNS` from `src/model.py`; the weak results likely reflect class
imbalance across the six tiers and/or overlapping feature distributions between
adjacent tiers. Champion F1 **0.700** and Backmarker F1 **0.685** are the tiers
where the classifier is genuinely useful for individual driver classification.

| Model | Test seasons | Mean accuracy | Accuracy SD | Mean macro F1 | Macro F1 SD |
|---|---:|---:|---:|---:|---:|
| Random Forest | 10 | 0.498 | 0.083 | 0.414 | 0.078 |

| Tier | Test seasons | Mean F1 | F1 SD |
|---|---:|---:|---:|
| Champion | 10 | 0.700 | 0.399 |
| Podium | 10 | 0.183 | 0.299 |
| Top 5 | 10 | 0.217 | 0.284 |
| Top 10 | 10 | 0.518 | 0.159 |
| Midfield | 10 | 0.184 | 0.181 |
| Backmarker | 10 | 0.685 | 0.104 |

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

#### Historical calibration audit

The uncertainty method is also backtested on the same ten untouched seasons.
The result is cautionary:

| Measure | Result |
|---|---:|
| Driver-season observations | 222 |
| Bootstrap P05–P95 coverage | 0.342 |
| Bootstrap mean interval width | 4.896 positions |
| Rolling conformal coverage | 0.977 |
| Rolling conformal mean width | 19.899 positions |
| Top-3 Brier score | 0.070 |
| Champion Brier score | 0.021 |

The bootstrap intervals are much too narrow to be interpreted as 90% prediction
intervals. Rolling conformal intervals recover coverage using only residuals
available before each test season, but their width makes them weakly informative.
The highest top-three probability bin is also overconfident: mean predicted
probability **0.967** versus an observed top-three rate of **0.789** across 19
driver-seasons. These findings are why the report labels bootstrap outputs as
model-sensitivity frequencies rather than calibrated real-world odds.

For the regenerated 2023 artifact, the leading point-forecast rows are:

| Driver | Point rank | Champion | Top 3 | Position P05–P95 |
|---|---:|---:|---:|---:|
| Max Verstappen | 1 | 97% | 100% | 1.85–3.63 |
| Sergio Pérez | 2 | 0% | 49% | 4.19–6.54 |
| Carlos Sainz | 3 | 0% | 57% | 4.00–7.00 |
| Charles Leclerc | 4 | 1% | 65% | 3.67–6.96 |

The CSV and HTML report include the same uncertainty fields for every driver.

---

## Where this model breaks

This section uses the same leak-free Random Forest walk-forward predictions and
adds post-hoc diagnostic labels. A positive signed error means the model placed
a driver lower than their actual final position. The labels are not predictors.

### Driver and season type

| Diagnostic group | Observations | RMSE | MAE | Mean signed error |
|---|---:|---:|---:|---:|
| Returning | 180 | 4.343 | 3.138 | 0.760 |
| Rookie | 31 | 9.219 | 7.362 | 4.129 |
| Returning after gap | 11 | 15.666 | 12.562 | 10.501 |
| No mid-season swap | 213 | 6.291 | 4.182 | 1.751 |
| Mid-season swap | 9 | 5.774 | 4.497 | 0.804 |
| Other test seasons | 200 | 6.348 | 4.177 | 1.690 |
| 2022 regulation-change season | 22 | 5.519 | 4.355 | 1.916 |

The clearest failure mode is missing history: returning drivers after a gap have
**12.562 MAE**, while ordinary returning drivers have **3.138 MAE**. Rookies
also have much larger error (**7.362 MAE**) than established returners. Mid-season
swaps are slightly worse than non-swaps, but the sample is only nine driver-
seasons, so that result is not conclusive.

### Worst chronological test seasons

| Test season | Train through | RMSE | MAE | Spearman ρ |
|---:|---:|---:|---:|---:|
| 2019 | 2018 | 8.125 | 6.036 | 0.744 |
| 2016 | 2015 | 9.847 | 5.772 | 0.849 |
| 2022 | 2021 | 5.519 | 4.355 | 0.832 |
| 2021 | 2020 | 8.050 | 4.195 | 0.874 |
| 2020 | 2019 | 4.967 | 4.161 | 0.768 |

The 2022 regulation-change case is a useful stress test, but it is not the
worst season by MAE in this sample. That is the honest conclusion: the model
shows a positive **1.916** mean signed error in 2022, but the largest failures
are concentrated in earlier seasons and in drivers with weak or missing history.

The underlying per-driver errors, season summaries, and group summaries are
regenerated by `python scripts/error_analysis.py` into
`results/error_analysis_driver.csv`,
`results/error_analysis_season_summary.csv`, and
`results/error_analysis_group_summary.csv`.

---

## Held-out permutation importance

Training-set impurity importance has been replaced with permutation importance
measured separately on every untouched future season. Values are the mean
increase in held-out RMSE after shuffling one feature; larger positive values
indicate more useful out-of-season information.

| Rank | Feature | Mean RMSE increase | Season SD | Positive seasons |
|---|---|---:|---:|---:|
| 1 | prev_season_points_sum | 4.272 | 1.144 | 10/10 |
| 2 | prev_season_races_started | 3.366 | 1.442 | 10/10 |
| 3 | prev_team_final_position | 2.322 | 1.678 | 10/10 |
| 4 | prev_team_final_points | 1.378 | 1.367 | 9/10 |
| 5 | prev_season_avg_finish_pos | 0.738 | 0.574 | 10/10 |
| 6 | prev_season_avg_grid_pos | 0.663 | 0.317 | 10/10 |
| 7 | missing_constructor_history | 0.459 | 0.781 | 6/10 |
| 8 | prev_season_podium_rate | 0.223 | 0.076 | 10/10 |
| 9 | prev_season_win_rate | 0.053 | 0.030 | 9/10 |
| 10 | returning_after_gap | 0.038 | 0.079 | 6/10 |

## Next steps

- Replace the season-opening-constructor approximation with an explicit pre-season entry list, as `MODEL_CARD.md` notes that the historical reconstruction currently takes the first constructor observed in race data.
- Investigate leakage-safe prior-season aggregates from the qualifying and pit-stop tables that `create_features()` currently loads and discards with `del qualifying, pit_stops`.
- Add per-season confusion matrices and class-support counts for all six tiers before changing the classifier, because Podium F1 **0.183** and Top 5 F1 **0.217** are close to unusable and do not show whether imbalance or adjacent-tier overlap dominates.
- Treat rookies and returning-after-gap drivers as a separate cold-start evaluation slice, because missing prior-season values are imputed with zero and those groups have the largest observed MAE (**7.362** and **12.562**).
- Keep the previous-season-order baseline as a model-selection gate for future changes, because it still beats every fitted regressor overall and the Random Forest loses nine of ten seasons on RMSE.

## Tech Stack

- **Language:** Python 3.12–3.13
- **Data:** CSV files (`pandas` DataFrames)  
- **Core libraries:**
  - `pandas`, `numpy`  
  - `scikit-learn`  
  - `matplotlib`, `seaborn` (visualization)  
- **Quality gates:** `pytest`, branch coverage, Ruff lint/format, GitHub Actions
