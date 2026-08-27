"""Run the reproducible rolling-origin evaluation protocol."""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from data_processing import create_features, load_raw_data
from model import (
    NAIVE_BASELINE_NAME,
    TIER_LABELS,
    evaluate_rolling_origin,
    evaluate_tier_rolling_origin,
)

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
            mean_r2=("r2", "mean"),
            r2_sd=("r2", "std"),
            mean_spearman=("spearman", "mean"),
            spearman_sd=("spearman", "std"),
            mean_spearman_delta_vs_naive=("spearman_delta_vs_naive", "mean"),
            spearman_delta_vs_naive_sd=("spearman_delta_vs_naive", "std"),
        )
        .reset_index()
        .sort_values("mean_spearman", ascending=False)
    )
    return summary.round(3)


def summarize_tier_results(details: pd.DataFrame) -> pd.DataFrame:
    """Aggregate walk-forward tier accuracy and macro-F1 metrics."""
    summary = (
        details.groupby("model")
        .agg(
            test_seasons=("test_year", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
        )
        .reset_index()
    )
    return summary.round(3)


def summarize_tier_classes(details: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-tier F1 values across the same chronological test seasons."""
    rows = []
    for tier in TIER_LABELS:
        column = f"f1_{tier.lower().replace(' ', '_')}"
        rows.append(
            {
                "tier": tier,
                "test_seasons": details["test_year"].nunique(),
                "mean_f1": details[column].mean(),
                "f1_sd": details[column].std(),
            }
        )
    return pd.DataFrame(rows).round(3)


def _paired_mean_interval(
    values: np.ndarray,
    confidence: float = 0.95,
    bootstrap_runs: int = 10_000,
    random_state: int = 42,
) -> tuple[float, float]:
    """Return a deterministic season-level bootstrap interval for a mean."""
    rng = np.random.default_rng(random_state)
    draws = rng.choice(values, size=(bootstrap_runs, len(values)), replace=True)
    means = draws.mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def summarize_paired_comparisons(details: pd.DataFrame) -> pd.DataFrame:
    """Compare each method with the naïve baseline on identical test seasons."""
    baseline = (
        details.loc[details["model"] == NAIVE_BASELINE_NAME]
        .set_index("test_year")[["rmse", "spearman"]]
        .rename(columns={"rmse": "baseline_rmse", "spearman": "baseline_spearman"})
    )
    rows = []
    for model_name, model_rows in details.groupby("model", sort=False):
        if model_name == NAIVE_BASELINE_NAME:
            continue
        paired = model_rows.set_index("test_year")[["rmse", "spearman"]].join(
            baseline, how="inner"
        )
        spearman_delta = (paired["spearman"] - paired["baseline_spearman"]).to_numpy()
        rmse_delta = (paired["rmse"] - paired["baseline_rmse"]).to_numpy()
        spearman_low, spearman_high = _paired_mean_interval(spearman_delta)
        rmse_low, rmse_high = _paired_mean_interval(rmse_delta)
        rows.append(
            {
                "model": model_name,
                "test_seasons": len(paired),
                "mean_spearman_delta_vs_naive": spearman_delta.mean(),
                "spearman_delta_ci95_low": spearman_low,
                "spearman_delta_ci95_high": spearman_high,
                "spearman_wins": int((spearman_delta > 0).sum()),
                "spearman_ties": int(np.isclose(spearman_delta, 0).sum()),
                "spearman_losses": int((spearman_delta < 0).sum()),
                "mean_rmse_delta_vs_naive": rmse_delta.mean(),
                "rmse_delta_ci95_low": rmse_low,
                "rmse_delta_ci95_high": rmse_high,
                "rmse_wins": int((rmse_delta < 0).sum()),
                "rmse_ties": int(np.isclose(rmse_delta, 0).sum()),
                "rmse_losses": int((rmse_delta > 0).sum()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("mean_spearman_delta_vs_naive", ascending=False)
        .round(3)
    )


def validate_baseline_gate(
    summary: pd.DataFrame,
    *,
    tolerance: float = 0.0,
) -> dict[str, object]:
    """Return an explicit model-selection decision against the naive baseline."""

    baseline_rows = summary.loc[summary["model"] == NAIVE_BASELINE_NAME]
    if baseline_rows.empty:
        raise ValueError("The previous-season total-points baseline is missing")
    baseline = float(baseline_rows.iloc[0]["mean_spearman"])
    candidates = summary.loc[summary["model"] != NAIVE_BASELINE_NAME].copy()
    candidates["passes_baseline_gate"] = candidates["mean_spearman"] >= baseline - tolerance
    return {
        "baseline_model": NAIVE_BASELINE_NAME,
        "baseline_mean_spearman": baseline,
        "tolerance": tolerance,
        "candidate_models": candidates[["model", "mean_spearman", "passes_baseline_gate"]].to_dict("records"),
        "passed": bool(candidates.empty or candidates["passes_baseline_gate"].all()),
    }


def expand_tier_confusion(details: pd.DataFrame) -> pd.DataFrame:
    """Expand per-season confusion/support JSON into an auditable long table."""

    labels = TIER_LABELS
    rows: list[dict[str, object]] = []
    for _, record in details.iterrows():
        matrix = json.loads(record["confusion_matrix_json"])
        actual_support = json.loads(record["actual_support_json"])
        predicted_support = json.loads(record["predicted_support_json"])
        for actual_index, actual_label in enumerate(labels):
            for predicted_index, predicted_label in enumerate(labels):
                rows.append(
                    {
                        "test_year": int(record["test_year"]),
                        "model": record["model"],
                        "actual_tier": actual_label,
                        "predicted_tier": predicted_label,
                        "count": int(matrix[actual_index][predicted_index]),
                        "actual_support": int(actual_support.get(actual_label, 0)),
                        "predicted_support": int(predicted_support.get(predicted_label, 0)),
                    }
                )
    return pd.DataFrame(rows)


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
    details = evaluate_rolling_origin(
        features,
        test_seasons=test_seasons,
        min_train_seasons=min_train_seasons,
    )
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
    create_baseline_comparison_chart(details, paired_chart_path)

    logger.info("Saved summary -> %s", output_path)
    logger.info("Saved per-season details -> %s", details_path)
    logger.info("\n%s", summary.to_string(index=False))
    logger.info("Saved tier summary -> %s", tier_output_path)
    logger.info("Saved tier per-season details -> %s", tier_details_path)
    logger.info("Saved tier class summary -> %s", tier_class_output_path)
    logger.info("Saved paired baseline comparison -> %s", paired_output_path)
    logger.info("Saved paired baseline chart -> %s", paired_chart_path)
    logger.info("\n%s", tier_summary.to_string(index=False))
    logger.info("\n%s", tier_class_summary.to_string(index=False))
    logger.info("\n%s", paired_summary.to_string(index=False))
    logger.info("Baseline gate: %s", json.dumps(baseline_gate, sort_keys=True))
    if require_baseline_gate and not baseline_gate["passed"]:
        raise ValueError("Model-selection baseline gate failed: no candidate meets the naive baseline")
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
