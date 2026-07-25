import argparse
import sys
import os

# Add src to path so we can import directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_processing import load_raw_data, create_features
from model import train_model
from predict import predict_championship

def main():
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
        print("⏭️  Skipping data processing (--skip-processing flag set)")
    else:
        print("=" * 60)
        print("STEP 1: Data Processing")
        print("=" * 60)
        data = load_raw_data()
        create_features(*data)

    # ── Step 2: Model Training ─────────────────────────────────────────────
    if args.skip_training:
        print("\n⏭️  Skipping model training (--skip-training flag set)")
    else:
        print("\n" + "=" * 60)
        print("STEP 2: Model Training")
        print("=" * 60)
        train_model()

    # ── Step 3: Prediction ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 3: Predicting {args.year} Championship")
    print("=" * 60)
    predict_championship(args.year)

    if args.visualise:
        print("\n" + "=" * 60)
        print(f"STEP 4: Visualising {args.year} Championship")
        print("=" * 60)
        from visualise import create_visualisation
        create_visualisation(args.year)

    print("\n✅  Pipeline complete!")

if __name__ == "__main__":
    main()
