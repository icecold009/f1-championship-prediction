"""Validate that a generated report contains the required release artifacts."""

import argparse
import logging
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_FILES = (
    "circuits.csv",
    "constructor_results.csv",
    "constructor_standings.csv",
    "constructors.csv",
    "driver_standings.csv",
    "drivers.csv",
    "lap_times.csv",
    "pit_stops.csv",
    "qualifying.csv",
    "races.csv",
    "results.csv",
    "seasons.csv",
    "sprint_results.csv",
    "status.csv",
)
logger = logging.getLogger(__name__)


def validate_release(base_dir: Path = BASE_DIR, year: int = 2023) -> list[str]:
    """Return actionable errors for a release directory, or an empty list."""
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    errors: list[str] = []

    missing_raw = [name for name in RAW_FILES if not (raw_dir / name).exists()]
    if missing_raw:
        errors.append(f"Missing raw data files: {', '.join(missing_raw)}")

    features_path = processed_dir / "features.csv"
    if not features_path.exists():
        errors.append("Missing processed features: data/processed/features.csv")
    else:
        features = pd.read_csv(features_path, nrows=1)
        required_features = {"year", "driverId", "champ_position"}
        missing_features = sorted(required_features - set(features.columns))
        if missing_features:
            errors.append(f"Processed features missing columns: {', '.join(missing_features)}")

    for filename in ("championship_model.pkl", "tier_classifier.pkl"):
        if not (models_dir / filename).exists():
            errors.append(f"Missing model artifact: models/{filename}")

    prediction_path = results_dir / f"{year}_predictions.csv"
    if not prediction_path.exists():
        errors.append(f"Missing prediction artifact: results/{year}_predictions.csv")
    else:
        predictions = pd.read_csv(prediction_path, nrows=1)
        required_prediction_columns = {"Predicted Rank", "Driver", "Predicted Score"}
        missing_prediction_columns = sorted(
            required_prediction_columns - set(predictions.columns)
        )
        if missing_prediction_columns:
            errors.append(
                "Prediction artifact missing columns: "
                + ", ".join(missing_prediction_columns)
            )

    for filename in (
        f"predicted_vs_actual_{year}.png",
        f"f1_prediction_report_{year}.html",
        "rolling_origin_summary.csv",
        "release_manifest.json",
    ):
        if not (results_dir / filename).exists():
            errors.append(f"Missing release artifact: results/{filename}")

    return errors


def main() -> int:
    """Validate the default local release and return a shell-friendly status."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2023, help="Prediction season to validate")
    args = parser.parse_args()

    errors = validate_release(year=args.year)
    if errors:
        logger.error("Release check failed:")
        for error in errors:
            logger.error("- %s", error)
        return 1

    logger.info("Release check passed for %s.", args.year)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
