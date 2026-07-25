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

# 2. Process data
python SRC/data_processing.py

# 3. Train models
python SRC/model.py

# 4. Generate predictions
python SRC/predict.py
```

Predictions are saved to `Results/predictions.csv`.

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

Typical sources:

- Public F1 datasets (CSV)  
- Ergast API exports  
- Manually cleaned CSV files in the `data/` folder  

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

All models trained on 80% of seasons, tested on 20% holdout. 
Ranking quality measured with Spearman correlation (higher = better predicted order).

| Model | CV RMSE | R² | Spearman ρ |
|---|---|---|---|
| Ridge Regression | 10.395 | 0.769 | 0.898 |
| Random Forest Regressor | 8.789 | 0.847 | 0.951 |
| Gradient Boosting Regressor | 8.975 | 0.854 | 0.949 |

**Best model:** Random Forest — selected by highest Spearman ρ (0.951).

**Tier Classifier (Random Forest):** CV Accuracy 0.851 | Test Accuracy 0.849

---

## Feature Importance (Best Model)

| Rank | Feature | Importance |
|---|---|---|
| 1 | points_sum | 0.266095 |
| 2 | points_per_race | 0.247010 |
| 3 | avg_finish_pos | 0.173354 |
| 4 | team_final_position | 0.164472 |
| 5 | std_finish_pos | 0.026965 |
| 6 | avg_grid_pos | 0.026899 |
| 7 | prev_season_points | 0.023409 |
| 8 | prev_season_avg_pos | 0.020578 |
| 9 | quali_to_race_delta | 0.019210 |
| 10 | races_started | 0.015752 |

## Tech Stack

- **Language:** Python 3.x  
- **Data:** CSV files (`pandas` DataFrames)  
- **Core libraries:**
  - `pandas`, `numpy`  
  - `scikit-learn`  
  - `matplotlib`, `seaborn` (visualization)  
  - `xgboost` (optional, if used)  
