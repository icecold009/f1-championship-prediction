"""Run the reproducible rolling-origin evaluation protocol."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from data_processing import create_features, load_raw_data
from model import evaluate_rolling_origin

logger = logging.getLogger(__name__)
DEFAULT_FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "results" / "rolling_origin_summary.csv"


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-season evaluation rows into a comparable model summary."""
    summary = (
        results.groupby("model")
        .agg(
            test_seasons=("test_year", "nunique"),
            mean_rmse=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            mean_spearman=("spearman", "mean"),
            spearman_sd=("spearman", "std"),
        )
        .reset_index()
        .sort_values("mean_spearman", ascending=False)
    )
    return summary.round(3)


def run_evaluation(
    features_path: Path = DEFAULT_FEATURES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    test_seasons: int = 10,
    min_train_seasons: int = 20,
    rebuild_features: bool = False,
) -> pd.DataFrame:
    """Run rolling-origin evaluation and save summary and detail CSVs."""
    if rebuild_features or not features_path.exists():
        logger.info("Building processed features from data/raw")
        try:
            create_features(*load_raw_data())
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Raw data is missing. Run `python scripts/download_data.py` first."
            ) from exc

    features = pd.read_csv(features_path)
    details = evaluate_rolling_origin(
        features,
        test_seasons=test_seasons,
        min_train_seasons=min_train_seasons,
    )
    if details.empty:
        raise ValueError("No rolling-origin evaluation rows were produced.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    details_path = output_path.with_name(f"{output_path.stem}_details{output_path.suffix}")
    summary = summarize_results(details)
    summary.to_csv(output_path, index=False)
    details.to_csv(details_path, index=False)

    logger.info("Saved summary -> %s", output_path)
    logger.info("Saved per-season details -> %s", details_path)
    logger.info("\n%s", summary.to_string(index=False))
    return summary


def main() -> int:
    """Parse CLI options and run the evaluation command."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Run leakage-safe rolling-origin F1 model evaluation.",
        epilog=(
            "Examples:\n"
            "  python scripts/evaluate.py\n"
            "  python scripts/evaluate.py --test-seasons 5\n"
            "  python scripts/evaluate.py --rebuild-features"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Processed feature CSV (default: data/processed/features.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Summary CSV path (default: results/rolling_origin_summary.csv)",
    )
    parser.add_argument(
        "--test-seasons",
        type=int,
        default=10,
        help="Number of latest seasons to test (default: 10)",
    )
    parser.add_argument(
        "--min-train-seasons",
        type=int,
        default=20,
        help="Minimum historical seasons before a test cutoff (default: 20)",
    )
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Rebuild data/processed/features.csv from data/raw first",
    )
    args = parser.parse_args()

    try:
        run_evaluation(
            features_path=args.features,
            output_path=args.output,
            test_seasons=args.test_seasons,
            min_train_seasons=args.min_train_seasons,
            rebuild_features=args.rebuild_features,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Evaluation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
