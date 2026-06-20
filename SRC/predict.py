import os
import pickle
import pandas as pd
from scipy.stats import spearmanr

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR    = os.path.join(BASE_DIR, "Data", "processed")
RAW_DIR     = os.path.join(BASE_DIR, "Data", "raw data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "avg_finish_pos",
    "std_finish_pos",
    "races_started",
    "points_sum",
    "avg_grid_pos",
    "win_rate",
    "podium_rate",
    "dnf_rate",
    "points_per_race",
    "quali_to_race_delta",
    "sprint_points_sum",
    "team_final_points",
    "team_final_position",
    "prev_season_points",
    "prev_season_avg_pos",
    "prev_season_win_rate",
]

def load_models():
    reg_path = os.path.join(MODEL_DIR, "championship_model.pkl")
    clf_path = os.path.join(MODEL_DIR, "tier_classifier.pkl")
    with open(reg_path, "rb") as f:
        reg_model = pickle.load(f)
    with open(clf_path, "rb") as f:
        clf_model = pickle.load(f)
    return reg_model, clf_model

def predict_championship(year: int):
    # ── Load models and features ───────────────────────────────────────────
    reg_model, clf_model = load_models()

    features_path = os.path.join(PROC_DIR, "features.csv")
    df = pd.read_csv(features_path)
    season_df = df[df["year"] == year].copy()

    if season_df.empty:
        print(f"No data found for year {year}. Available years: {sorted(df['year'].unique())}")
        return

    # ── Load driver and constructor names for readable output ──────────────
    drivers      = pd.read_csv(os.path.join(RAW_DIR, "drivers.csv"))
    constructors = pd.read_csv(os.path.join(RAW_DIR, "constructors.csv"))

    driver_names = drivers[["driverId", "forename", "surname"]].copy()
    driver_names["driver_name"] = driver_names["forename"] + " " + driver_names["surname"]

    season_df = season_df.merge(
        driver_names[["driverId", "driver_name"]], on="driverId", how="left"
    )
    season_df = season_df.merge(
        constructors[["constructorId", "name"]].rename(columns={"name": "team_name"}),
        on="constructorId", how="left"
    )

    # ── Predict ───────────────────────────────────────────────────────────
    X = season_df[FEATURE_COLUMNS].fillna(0)
    season_df["predicted_position"] = reg_model.predict(X)
    season_df["predicted_tier"]     = clf_model.predict(X)

    # Rank by predicted position (lowest number = champion)
    season_df = season_df.sort_values("predicted_position").reset_index(drop=True)
    season_df["predicted_rank"] = season_df.index + 1

    # ── Build output table ────────────────────────────────────────────────
    output = season_df[[
        "predicted_rank",
        "driver_name",
        "team_name",
        "predicted_tier",
        "predicted_position",
        "champ_position",
        "champ_points",
    ]].copy()

    output.columns = [
        "Predicted Rank",
        "Driver",
        "Team",
        "Predicted Tier",
        "Predicted Score",
        "Actual Position",
        "Actual Points",
    ]
    output["Predicted Score"] = output["Predicted Score"].round(2)

    # ── Spearman vs actual (only if actual data available) ────────────────
    has_actual = output["Actual Position"].notna().sum() > 3
    if has_actual:
        valid = output.dropna(subset=["Actual Position"])
        corr_val, _ = spearmanr(valid["Actual Position"], valid["Predicted Score"])
        spearman = round(float(corr_val), 3)  # type: ignore
        print(f"\n── {year} Championship Prediction ─────────────────────────────")
        print(f"  Spearman rank correlation vs actual: {spearman}")
    else:
        print(f"\n── {year} Championship Prediction (no actual results yet) ──────")

    print(f"\n{output.to_string(index=False)}")

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, f"{year}_predictions.csv")
    output.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")
    return output

if __name__ == "__main__":
    # Change year to any season in your dataset
    predict_championship(2023)