"""Run the reproducible rolling-origin evaluation protocol."""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

import evaluation as _evaluation
from data_processing import create_features, load_raw_data
from evaluation import (
    NAIVE_BASELINE_NAME,
    evaluate_rolling_origin_with_failures,
    evaluate_tier_rolling_origin,
    expand_tier_confusion,
    summarize_paired_comparisons,
    summarize_results,
    summarize_tier_classes,
    summarize_tier_results,
    validate_baseline_gate,
)

logger = logging.getLogger(__name__)
DEFAULT_FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "results" / "rolling_origin_summary.csv"
TIER_LABELS = _evaluation.TIER_LABELS
evaluate_rolling_origin = _evaluation.evaluate_rolling_origin
_paired_mean_interval = _evaluation._paired_mean_interval


def create_baseline_comparison_chart(
    details: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot each model's paired per-season Spearman delta versus the baseline."""
    plot_data = details.loc[details["model"] != NAIVE_BASELINE_NAME].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    for model_name, group in plot_data.groupby("model", sort=False):
        ax.plot(
            group["test_year"],
            group["spearman_delta_vs_naive"],
            marker="o",
            linewidth=1.8,
            label=model_name,
        )
    ax.axhline(0, color="#172033", linewidth=1, linestyle="--")
    ax.set(
        title="Per-season ranking performance versus naïve final-order baseline",
        xlabel="Held-out test season",
        ylabel="Spearman Δ (model − naïve baseline)",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def run_evaluation(
    features_path: Path = DEFAULT_FEATURES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    test_seasons: int = 10,
    min_train_seasons: int = 20,
    rebuild_features: bool = False,
    require_baseline_gate: bool = False,
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
    evaluation_run = evaluate_rolling_origin_with_failures(
        features,
        test_seasons=test_seasons,
        min_train_seasons=min_train_seasons,
    )
    details = evaluation_run.details
    failures = evaluation_run.failures
    if details.empty:
        raise ValueError("No rolling-origin evaluation rows were produced.")

    tier_details = evaluate_tier_rolling_origin(
        features,
        test_seasons=test_seasons,
        min_train_seasons=min_train_seasons,
    )
    if tier_details.empty:
        raise ValueError("No tier rolling-origin evaluation rows were produced.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    details_path = output_path.with_name(
        f"{output_path.stem}_details{output_path.suffix}"
    )
    tier_output_path = output_path.with_name("tier_rolling_origin_summary.csv")
    tier_details_path = output_path.with_name("tier_rolling_origin_summary_details.csv")
    tier_class_output_path = output_path.with_name(
        "tier_rolling_origin_class_summary.csv"
    )
    paired_output_path = output_path.with_name("model_vs_naive_summary.csv")
    paired_chart_path = output_path.with_name("model_vs_naive_by_season.png")
    failures_path = output_path.with_name(f"{output_path.stem}_failures.csv")
    summary = summarize_results(details)
    paired_summary = summarize_paired_comparisons(details)
    baseline_gate = validate_baseline_gate(summary)
    tier_summary = summarize_tier_results(tier_details)
    tier_class_summary = summarize_tier_classes(tier_details)
    summary.to_csv(output_path, index=False)
    details.to_csv(details_path, index=False)
    tier_summary.to_csv(tier_output_path, index=False)
    tier_details.to_csv(tier_details_path, index=False)
    tier_class_summary.to_csv(tier_class_output_path, index=False)
    expand_tier_confusion(tier_details).to_csv(
        output_path.with_name("tier_rolling_origin_confusion.csv"), index=False
    )
    paired_summary.to_csv(paired_output_path, index=False)
    failures.to_csv(failures_path, index=False)
    create_baseline_comparison_chart(details, paired_chart_path)

    logger.info("Saved summary -> %s", output_path)
    logger.info("Saved per-season details -> %s", details_path)
    logger.info("\n%s", summary.to_string(index=False))
    logger.info("Saved tier summary -> %s", tier_output_path)
    logger.info("Saved tier per-season details -> %s", tier_details_path)
    logger.info("Saved tier class summary -> %s", tier_class_output_path)
    logger.info("Saved paired baseline comparison -> %s", paired_output_path)
    logger.info("Saved evaluation failures -> %s", failures_path)
    logger.info("Saved paired baseline chart -> %s", paired_chart_path)
    logger.info("\n%s", tier_summary.to_string(index=False))
    logger.info("\n%s", tier_class_summary.to_string(index=False))
    logger.info("\n%s", paired_summary.to_string(index=False))
    logger.info("Baseline gate: %s", json.dumps(baseline_gate, sort_keys=True))
    if require_baseline_gate and not baseline_gate["passed"]:
        raise ValueError(
            "Model-selection baseline gate failed: no candidate meets the naive baseline"
        )
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
    parser.add_argument(
        "--require-baseline-gate",
        action="store_true",
        help="Fail if every evaluated candidate does not meet the previous-season baseline.",
    )
    args = parser.parse_args()

    try:
        run_evaluation(
            features_path=args.features,
            output_path=args.output,
            test_seasons=args.test_seasons,
            min_train_seasons=args.min_train_seasons,
            rebuild_features=args.rebuild_features,
            require_baseline_gate=args.require_baseline_gate,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Evaluation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
