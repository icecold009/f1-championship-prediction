"""Generate leak-free per-season and per-driver error analysis."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from data_processing import load_raw_data
from model import FEATURE_COLUMNS, _rolling_cutoffs, assign_tier, get_spearman

logger = logging.getLogger(__name__)
DEFAULT_FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results"
REGULATION_CHANGE_YEARS = {2022}


def annotate_driver_context(
    features: pd.DataFrame,
    results: pd.DataFrame,
    races: pd.DataFrame,
) -> pd.DataFrame:
    """Add post-hoc driver and season categories without changing predictors."""
    annotated = features.copy()
    first_year = annotated.groupby("driverId")["year"].transform("min")
    no_prior_history = annotated["prev_season_races_started"].isna()
    annotated["driver_type"] = np.select(
        [annotated["year"].eq(first_year), no_prior_history],
        ["Rookie", "Returning after gap"],
        default="Returning",
    )

    constructor_counts = (
        results.merge(races[["raceId", "year"]], on="raceId", how="left")
        .groupby(["year", "driverId"])["constructorId"]
        .nunique()
        .reset_index(name="constructor_count")
    )
    annotated = annotated.merge(
        constructor_counts, on=["year", "driverId"], how="left"
    )
    annotated["constructor_count"] = annotated["constructor_count"].fillna(0).astype(int)
    annotated["mid_season_swap"] = annotated["constructor_count"] > 1
    annotated["regulation_change_year"] = annotated["year"].isin(
        REGULATION_CHANGE_YEARS
    )
    return annotated


def generate_error_rows(
    features: pd.DataFrame,
    test_seasons: int = 10,
    min_train_seasons: int = 20,
    n_estimators: int = 200,
) -> pd.DataFrame:
    """Generate out-of-season Random Forest predictions for every test driver."""
    rows = []
    for test_year, train_df, test_df, train_years in _rolling_cutoffs(
        features, test_seasons, min_train_seasons
    ):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42,
        )
        model.fit(
            train_df[FEATURE_COLUMNS].fillna(0),
            train_df["champ_position"],
        )
        predictions = model.predict(test_df[FEATURE_COLUMNS].fillna(0))
        output = test_df.copy()
        output["test_year"] = test_year
        output["train_end_year"] = train_years[-1]
        output["predicted_position"] = predictions
        output["absolute_error"] = (
            output["predicted_position"] - output["champ_position"]
        ).abs()
        output["signed_error"] = (
            output["predicted_position"] - output["champ_position"]
        )
        output["actual_tier"] = output["champ_position"].apply(assign_tier)
        rows.append(output)

    if not rows:
        raise ValueError("No walk-forward error rows were produced.")

    errors = pd.concat(rows, ignore_index=True)
    errors["predicted_rank"] = errors.groupby("test_year")[
        "predicted_position"
    ].rank(method="first", ascending=True)
    return errors


def summarize_seasons(errors: pd.DataFrame) -> pd.DataFrame:
    """Summarize error size and direction for each chronological test season."""
    rows = []
    for test_year, group in errors.groupby("test_year", sort=True):
        rows.append(
            {
                "test_year": int(test_year),
                "train_end_year": int(group["train_end_year"].iloc[0]),
                "observations": len(group),
                "rmse": np.sqrt(np.mean(group["signed_error"] ** 2)),
                "mae": group["absolute_error"].mean(),
                "median_absolute_error": group["absolute_error"].median(),
                "mean_signed_error": group["signed_error"].mean(),
                "spearman": get_spearman(
                    group["champ_position"], group["predicted_position"]
                ),
                "rookie_mae": group.loc[
                    group["driver_type"] == "Rookie", "absolute_error"
                ].mean(),
                "returning_mae": group.loc[
                    group["driver_type"] == "Returning", "absolute_error"
                ].mean(),
                "swap_mae": group.loc[
                    group["mid_season_swap"], "absolute_error"
                ].mean(),
                "regulation_change_year": bool(
                    group["regulation_change_year"].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows).round(3)


def summarize_groups(errors: pd.DataFrame) -> pd.DataFrame:
    """Summarize error size across driver and season categories."""
    groups = [
        ("Driver type", "driver_type", lambda value: str(value)),
        (
            "Constructor changes",
            "mid_season_swap",
            lambda value: "Mid-season swap" if value else "No mid-season swap",
        ),
        (
            "Regulation context",
            "regulation_change_year",
            lambda value: "2022 regulation-change season"
            if value
            else "Other test seasons",
        ),
    ]
    rows = []
    for dimension, column, label in groups:
        for value, group in errors.groupby(column, dropna=False):
            rows.append(
                {
                    "dimension": dimension,
                    "group": label(value),
                    "observations": len(group),
                    "seasons": group["test_year"].nunique(),
                    "rmse": np.sqrt(np.mean(group["signed_error"] ** 2)),
                    "mae": group["absolute_error"].mean(),
                    "median_absolute_error": group["absolute_error"].median(),
                    "mean_signed_error": group["signed_error"].mean(),
                }
            )
    return pd.DataFrame(rows).round(3)


def run_error_analysis(
    features_path: Path = DEFAULT_FEATURES_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    test_seasons: int = 10,
    min_train_seasons: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and save per-driver, per-season, and grouped error artifacts."""
    features = pd.read_csv(features_path)
    try:
        raw = load_raw_data()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Raw data is missing. Run `python scripts/download_data.py` first."
        ) from exc
    races, results = raw[0], raw[1]
    annotated = annotate_driver_context(features, results, races)
    errors = generate_error_rows(
        annotated,
        test_seasons=test_seasons,
        min_train_seasons=min_train_seasons,
    )
    drivers = raw[2][["driverId", "forename", "surname"]].copy()
    drivers["driver"] = drivers["forename"] + " " + drivers["surname"]
    errors = errors.merge(drivers[["driverId", "driver"]], on="driverId", how="left")
    errors = errors[
        [
            "test_year",
            "train_end_year",
            "driverId",
            "driver",
            "champ_position",
            "predicted_position",
            "predicted_rank",
            "absolute_error",
            "signed_error",
            "actual_tier",
            "driver_type",
            "constructor_count",
            "mid_season_swap",
            "regulation_change_year",
        ]
    ].sort_values(["test_year", "predicted_rank"])
    season_summary = summarize_seasons(errors)
    group_summary = summarize_groups(errors)

    output_dir.mkdir(parents=True, exist_ok=True)
    errors.to_csv(output_dir / "error_analysis_driver.csv", index=False)
    season_summary.to_csv(output_dir / "error_analysis_season_summary.csv", index=False)
    group_summary.to_csv(output_dir / "error_analysis_group_summary.csv", index=False)
    logger.info("Saved driver errors -> %s", output_dir / "error_analysis_driver.csv")
    logger.info("Saved season summary -> %s", output_dir / "error_analysis_season_summary.csv")
    logger.info("Saved group summary -> %s", output_dir / "error_analysis_group_summary.csv")
    return errors, season_summary, group_summary


def main() -> int:
    """Run the error-analysis command."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=DEFAULT_FEATURES_PATH
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--test-seasons", type=int, default=10)
    parser.add_argument("--min-train-seasons", type=int, default=20)
    args = parser.parse_args()
    try:
        _, season_summary, group_summary = run_error_analysis(
            features_path=args.features,
            output_dir=args.output_dir,
            test_seasons=args.test_seasons,
            min_train_seasons=args.min_train_seasons,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Error analysis failed: %s", exc)
        return 1
    logger.info("\n%s", season_summary.to_string(index=False))
    logger.info("\n%s", group_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
