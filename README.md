# f1-championship-prediction
# Predicting F1 Championship Final Standings 🏎️📊

A data science project where I use historical Formula 1 race data to **predict the final Drivers’ Championship standings** for a season.

The goal is to explore how well machine learning models can approximate the final rankings using data like race results, qualifying performance, team strength, and driver consistency.

## Project Overview

This project focuses on:

- Collecting and cleaning **historical F1 data** (drivers, constructors, races, results)
- Engineering features that represent:
  - Driver performance across races  
  - Team/constructor strength  
  - Qualifying vs race pace  
  - Reliability (DNFs, DNS, etc.)
- Training machine learning models to:
  - Predict **final points**
  - Predict **final position / tier** (e.g. champion, podium contender, midfield, backmarker)
- Evaluating how well we can **reconstruct the final standings** of a season using only historic season data up to that point.

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
- Season-level aggregates:
  - Total races started  
  - Average finish position  
  - Average grid position  
  - Points per race  
  - Podiums, wins, poles, fastest laps  

The raw CSVs are downloaded from the public [Formula 1 Race Data Kaggle dataset](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data), which uses the Ergast-compatible table layout expected by this project. Run `python scripts/download_data.py` to refresh them.

Typical source categories:

- Public F1 datasets (CSV)  
- Ergast API exports  
- Manually cleaned CSV files in the local `data/raw/` folder

## Features & Approach

Key feature engineering ideas:

### Performance metrics

- Average finish position  
- Average grid position  
- Delta between grid and finish (racecraft)  
- Win / podium / points-scoring rate  

### Reliability

- Number and percentage of DNFs  
- Races started vs races in season  

### Team strength

- Total constructor points  
- Team average finish position  
- Team average qualifying position  

### Experience

- Seasons in F1  
- Total career points (up to that year)  

## Model Results

Models trained on seasons 1950–2019 and evaluated on the latest five seasons (2020–2024).
Cross-validation groups rows by season to avoid leakage across seasons.
Ranking quality measured with Spearman correlation (higher = better predicted order).

| Model | CV RMSE | R² | Spearman ρ |
|---|---|---|---|
| Ridge Regression | 10.511 | -0.391 | 0.777 |
| Random Forest Regressor | 9.174 | 0.798 | 0.973 |
| Gradient Boosting Regressor | 9.348 | 0.773 | 0.976 |

**Best model:** Gradient Boosting — selected by highest Spearman ρ (0.976).

**Tier Classifier (Random Forest):** Grouped CV Accuracy 0.857 | Test Accuracy 0.728 | Test Macro F1 0.681

| Tier | Test F1 |
|---|---:|
| Champion | 0.800 |
| Podium | 0.632 |
| Top 5 | 0.381 |
| Top 10 | 0.691 |
| Midfield | 0.680 |
| Backmarker | 0.904 |

---

## Feature Importance (Best Model)

| Rank | Feature | Importance |
|---|---|---|
| 1 | points_sum | 0.362626 |
| 2 | avg_finish_pos | 0.177251 |
| 3 | team_final_position | 0.163500 |
| 4 | points_per_race | 0.145328 |
| 5 | races_started | 0.035875 |
| 6 | prev_season_points | 0.023919 |
| 7 | std_finish_pos | 0.020127 |
| 8 | quali_to_race_delta | 0.018182 |
| 9 | prev_season_avg_pos | 0.017307 |
| 10 | team_final_points | 0.016320 |

## Tech Stack

- **Language:** Python 3.x  
- **Data:** CSV files (`pandas` DataFrames)  
- **Core libraries:**
  - `pandas`, `numpy`  
  - `scikit-learn`  
  - `matplotlib`, `seaborn` (visualization)  
  - `xgboost` (optional, if used)  
