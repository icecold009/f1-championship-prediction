"""Backward-compatible facade for the staged championship data pipeline."""

import logging
import os
from pathlib import Path

import pandas as pd

try:
    from . import data_pipeline
except ImportError:  # Script entrypoints import this facade from the src path.
    import data_pipeline

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

# Keep the existing private helper names available to current callers/tests.
NON_START_STATUS_NAMES = data_pipeline.NON_START_STATUS_NAMES
_status_ids = data_pipeline._status_ids
_completed_season_years = data_pipeline._completed_season_years
_complete_driver_standings = data_pipeline._complete_driver_standings


def _path(filename: str) -> str:
    return os.path.join(RAW_DIR, filename)


def load_raw_data() -> tuple[pd.DataFrame, ...]:
    """Load source CSVs in the original positional order."""
    return data_pipeline.load_raw_tables(RAW_DIR).as_legacy_tuple()


def create_features(
    races: pd.DataFrame,
    results: pd.DataFrame,
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    driver_standings: pd.DataFrame,
    constructor_standings: pd.DataFrame,
    qualifying: pd.DataFrame,
    status: pd.DataFrame,
    sprint_results: pd.DataFrame,
    pit_stops: pd.DataFrame,
) -> pd.DataFrame:
    """Build features through typed pipeline stages and write the legacy CSV."""
    output_path = Path(PROC_DIR) / "features.csv"
    return data_pipeline.create_features(
        races,
        results,
        drivers,
        constructors,
        driver_standings,
        constructor_standings,
        qualifying,
        status,
        sprint_results,
        pit_stops,
        output_path=output_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = load_raw_data()
    features = create_features(*data)
    logger.info("%s", features.head())
    logger.info("Features shape: %s", features.shape)
    logger.info("Columns: %s", list(features.columns))
