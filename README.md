# f1-championship-prediction
# Predicting F1 Championship Final Standings 🏎️📊

A data science project where I use historical Formula 1 race data to **predict the final Drivers’ Championship standings** for a season.

The goal is to explore how well machine learning models can approximate the final rankings using data like race results, qualifying performance, team strength, and driver consistency.

---

## 📌 Project Overview

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

---

## 🧠 Problem Framing

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

---

## 📂 Data

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

---

## 🧱 Features & Approach

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

---

## 🧮 Models Used

This project experiments with multiple models (using scikit-learn):

- **Regression**
  - Linear Regression / Ridge Regression  
  - Random Forest Regressor  
  - Gradient Boosting / XGBoost (optional)  

- **Classification (tiers / buckets)**
  - Logistic Regression  
  - Random Forest Classifier  

---

## 🛠️ Tech Stack

- **Language:** Python 3.x  
- **Data:** CSV files (`pandas` DataFrames)  
- **Core libraries:**
  - `pandas`, `numpy`  
  - `scikit-learn`  
  - `matplotlib`, `seaborn` (visualization)  
  - `xgboost` (optional, if used)  
