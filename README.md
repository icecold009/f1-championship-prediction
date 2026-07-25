# f1-championship-prediction
# Predicting F1 Championship Final Standings 🏎️📊

A data science project where I use historical Formula 1 race data to **forecast the final Drivers’ Championship standings** for a season.

The goal is to explore how well machine learning models can approximate the final rankings using data like race results, qualifying performance, team strength, and driver consistency.

## Project Overview

This project focuses on:

- Collecting and cleaning **historical F1 data** (drivers, constructors, races, results)
- Engineering features that represent prior-season driver performance, constructor strength, qualifying pace, and reliability
- Training machine learning models to:
  - Predict **final points**
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

# 3. Run processing, training, prediction, and visualisation
python main.py --year 2023 --visualise
```

Predictions are saved to `results/2023_predictions.csv`; the optional chart is saved to
`results/predicted_vs_actual_2023.png`. Processed data and result files are regenerated locally
and ignored by Git.

## Tests

Run the unit tests locally with:

```bash
python -m pytest -q
```

The same test command runs in GitHub Actions for Python 3.12 and 3.13.

## Problem Framing

There are two main prediction tasks:

1. **Regression task**  
   Predict the **final championship points** for each driver.

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

Models trained on seasons 1950–2019 and evaluated on the latest five seasons (2020–2024).
Each forecast row uses the driver's prior-season statistics and the prior final standings of the constructor they enter with. Same-season race results are used only to identify entrants and construct the final target, not as model inputs. Cross-validation groups rows by season to avoid leakage across seasons.
Ranking quality measured with Spearman correlation (higher = better predicted order).

### Rolling-origin backtest

Each regressor was retrained before each test season from **2015–2024**, using
only seasons earlier than that test season. Values below are the mean and
standard deviation across the ten chronological test seasons.

| Model / baseline | Mean RMSE | RMSE SD | Mean Spearman | Spearman SD |
|---|---:|---:|---:|---:|
| Baseline: previous points rank | 3.758 | 0.747 | 0.821 | 0.081 |
| Random Forest | 6.657 | 2.818 | 0.808 | 0.061 |
| Gradient Boosting | 6.888 | 2.747 | 0.788 | 0.057 |
| Baseline: previous avg finish | 4.564 | 0.875 | 0.783 | 0.085 |
| Ridge | 11.921 | 2.317 | 0.686 | 0.102 |

In this evaluation snapshot, the previous-season points-rank baseline outperforms
the ML regressors. That is a useful result: future model changes must beat this
reference before they can be described as adding predictive value.

The fixed 2020–2024 holdout below is retained as the headline comparison;
the rolling-origin results show how stable performance is across multiple
forecast cutoffs rather than relying on one test window.

| Model | CV RMSE | R² | Spearman ρ |
|---|---|---|---|
| Ridge Regression | 17.410 | -1.789 | 0.699 |
| Random Forest Regressor | 16.828 | 0.236 | 0.832 |
| Gradient Boosting Regressor | 17.404 | 0.152 | 0.776 |

**Best model:** Random Forest — selected by highest Spearman ρ (0.832).

**Tier Classifier (Random Forest):** Stratified Grouped CV Accuracy 0.707 | Test Accuracy 0.518 | Test Macro F1 0.461

| Tier | Test F1 |
|---|---:|
| Champion | 0.667 |
| Podium | 0.471 |
| Top 5 | 0.125 |
| Top 10 | 0.515 |
| Midfield | 0.256 |
| Backmarker | 0.730 |

---

## Feature Importance (Best Model)

| Rank | Feature | Importance |
|---|---|---|
| 1 | prev_season_races_started | 0.504747 |
| 2 | prev_team_final_position | 0.132477 |
| 3 | prev_season_avg_grid_pos | 0.074579 |
| 4 | prev_season_avg_finish_pos | 0.056302 |
| 5 | prev_season_quali_to_race_delta | 0.056148 |
| 6 | prev_team_final_points | 0.052832 |
| 7 | prev_season_points_sum | 0.050951 |
| 8 | prev_season_std_finish_pos | 0.032967 |
| 9 | prev_season_points_per_race | 0.018516 |
| 10 | prev_season_dnf_rate | 0.013810 |

## Tech Stack

- **Language:** Python 3.x  
- **Data:** CSV files (`pandas` DataFrames)  
- **Core libraries:**
  - `pandas`, `numpy`  
  - `scikit-learn`  
  - `matplotlib`, `seaborn` (visualization)  
