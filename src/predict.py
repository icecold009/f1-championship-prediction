import logging
import os
import pickle

import pandas as pd
from scipy.stats import spearmanr
from typing import Any

try:
    from src.model import bootstrap_position_predictions
except ModuleNotFoundError:  # pragma: no cover - CLI path adds src directly
    from model import bootstrap_position_predictions

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BOOTSTRAP_RUNS = 100
BOOTSTRAP_ESTIMATORS = 200
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "prev_season_races_started",
    "prev_season_avg_finish_pos",
    "prev_season_std_finish_pos",
    "prev_season_points_sum",
    "prev_season_avg_grid_pos",
    "prev_season_win_rate",
    "prev_season_podium_rate",
    "prev_season_dnf_rate",
    "prev_season_points_per_race",
    "prev_season_quali_to_race_delta",
    "prev_season_sprint_points_sum",
    "prev_team_final_points",
    "prev_team_final_position",
]

def load_models() -> tuple[Any, Any]:
    """Load the saved regression and tier-classification model artifacts."""
    reg_path = os.path.join(MODEL_DIR, "championship_model.pkl")
    clf_path = os.path.join(MODEL_DIR, "tier_classifier.pkl")
    try:
        with open(reg_path, "rb") as f:
            reg_model = pickle.load(f)
        with open(clf_path, "rb") as f:
            clf_model = pickle.load(f)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Saved models were not found. Run `python src/model.py` first."
        ) from exc
    except (EOFError, OSError, pickle.UnpicklingError) as exc:
        raise RuntimeError(
            "Saved model artifacts could not be loaded. Run `python src/model.py` again."
        ) from exc
    return reg_model, clf_model

def predict_championship(year: int) -> pd.DataFrame | None:
    """Predict and save the ordered championship standings for one season."""
    # ── Load models and features ───────────────────────────────────────────
    reg_model, clf_model = load_models()

    features_path = os.path.join(PROC_DIR, "features.csv")
    df = pd.read_csv(features_path)
    season_df = df[df["year"] == year].copy()

    if season_df.empty:
        available_years = [int(value) for value in sorted(df["year"].unique())]
        logger.info("No data found for year %s. Available years: %s", year, available_years)
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

    # ── Predict with point model and season-level bootstrap uncertainty ───
    X = season_df[FEATURE_COLUMNS].fillna(0)
    season_df["predicted_position"] = reg_model.predict(X)
    season_df["predicted_tier"]     = clf_model.predict(X)

    train_df = df[df["year"] < year].dropna(subset=["champ_position"])
    uncertainty = bootstrap_position_predictions(
        train_df,
        season_df,
        n_bootstrap=BOOTSTRAP_RUNS,
        n_estimators=BOOTSTRAP_ESTIMATORS,
        random_state=42,
    )
    for column in uncertainty.columns:
        season_df[column] = uncertainty[column].to_numpy()

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
        "bootstrap_runs",
        "bootstrap_position_mean",
        "bootstrap_position_sd",
        "bootstrap_position_p05",
        "bootstrap_position_p95",
        "champion_probability",
        "top_3_probability",
        "top_5_probability",
    ]].copy()

    output.columns = [
        "Predicted Rank",
        "Driver",
        "Team",
        "Predicted Tier",
        "Predicted Position",
        "Actual Position",
        "Actual Points",
        "Bootstrap Runs",
        "Bootstrap Position Mean",
        "Bootstrap Position SD",
        "Bootstrap Position P05",
        "Bootstrap Position P95",
        "Champion Probability",
        "Top 3 Probability",
        "Top 5 Probability",
    ]
    output["Predicted Position"] = output["Predicted Position"].round(2)
    for column in (
        "Bootstrap Position Mean",
        "Bootstrap Position SD",
        "Bootstrap Position P05",
        "Bootstrap Position P95",
        "Champion Probability",
        "Top 3 Probability",
        "Top 5 Probability",
    ):
        output[column] = output[column].round(4)

    # ── Spearman vs actual (only if actual data available) ────────────────
    has_actual = output["Actual Position"].notna().sum() > 3
    if has_actual:
        valid = output.dropna(subset=["Actual Position"])
        corr_val, _ = spearmanr(valid["Actual Position"], valid["Predicted Position"])
        spearman = round(float(corr_val), 3)  # type: ignore
        logger.info("\n── %s Championship Prediction ─────────────────────────────", year)
        logger.info("  Spearman rank correlation vs actual: %s", spearman)
    else:
        logger.info("\n── %s Championship Prediction (no actual results yet) ──────", year)

    logger.info("\n%s", output.to_string(index=False))

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, f"{year}_predictions.csv")
    output.to_csv(out_path, index=False)
    logger.info("\n  Saved → %s", out_path)
    return output

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Change year to any season in your dataset
    predict_championship(2023)
