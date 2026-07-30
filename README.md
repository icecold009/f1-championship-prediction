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

---

## Feature Importance (Predeclared Random Forest)

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
