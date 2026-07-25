import argparse
import logging
import os
import sys

# Add src to path so we can import directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_processing import load_raw_data, create_features
from model import train_model
from predict import predict_championship

logger = logging.getLogger(__name__)


def main() -> None:
    """Run processing, model training, prediction, and optional visualisation."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="F1 Championship Prediction Pipeline"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="Season year to predict (default: 2023)"
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip data processing if features.csv already exists"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training if models already exist"
    )
    parser.add_argument(
        "--visualise",
        action="store_true",
        help="Generate the predicted-vs-actual chart after prediction"
    )
    args = parser.parse_args()

    # ── Step 1: Data Processing ────────────────────────────────────────────
    if args.skip_processing:
        logger.info("⏭️  Skipping data processing (--skip-processing flag set)")
    else:
        logger.info("%s", "=" * 60)
        logger.info("STEP 1: Data Processing")
        logger.info("%s", "=" * 60)
        data = load_raw_data()
        create_features(*data)

    # ── Step 2: Model Training ─────────────────────────────────────────────
    if args.skip_training:
        logger.info("\n⏭️  Skipping model training (--skip-training flag set)")
    else:
        logger.info("\n%s", "=" * 60)
        logger.info("STEP 2: Model Training")
        logger.info("%s", "=" * 60)
        train_model()

    # ── Step 3: Prediction ─────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("STEP 3: Predicting %s Championship", args.year)
    logger.info("%s", "=" * 60)
    predict_championship(args.year)

    if args.visualise:
        logger.info("\n%s", "=" * 60)
        logger.info("STEP 4: Visualising %s Championship", args.year)
        logger.info("%s", "=" * 60)
        from visualise import create_visualisation
        create_visualisation(args.year)

    logger.info("\n✅  Pipeline complete!")

if __name__ == "__main__":
    main()
