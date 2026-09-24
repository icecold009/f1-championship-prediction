"""Pure chronological evaluation, fold records, and metric aggregation."""

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
)

try:
    from .model_registry import (
        FEATURE_COLUMNS,
        create_regression_candidates,
        create_tier_classifier,
    )
except ImportError:  # Script entrypoints import this module from the src path.
    from model_registry import (
        FEATURE_COLUMNS,
        create_regression_candidates,
        create_tier_classifier,
    )

TIER_LABELS = ["Champion", "Podium", "Top 5", "Top 10", "Midfield", "Backmarker"]
ROLLING_ORIGIN_TEST_SEASONS = 10
NAIVE_BASELINE_NAME = "Naive: previous-season final order"


@dataclass(frozen=True)
class EvaluationFold:
    test_year: int
    train_end_year: int
    train: pd.DataFrame
    test: pd.DataFrame
    train_years: tuple[int, ...]

    def __iter__(self):
        """Retain the legacy tuple iteration used by audit scripts."""
        yield self.test_year
        yield self.train
        yield self.test
        yield list(self.train_years)


@dataclass(frozen=True)
class EvaluationRun:
    details: pd.DataFrame
    failures: pd.DataFrame


EVALUATION_FAILURE_COLUMNS = [
    "test_year",
    "train_end_year",
    "model",
    "error_type",
    "error_message",
]
PAIRED_COMPARISON_COLUMNS = [
    "model",
    "test_seasons",
    "mean_spearman_delta_vs_naive",
    "spearman_delta_ci95_low",
    "spearman_delta_ci95_high",
    "spearman_wins",
    "spearman_ties",
    "spearman_losses",
    "mean_rmse_delta_vs_naive",
    "rmse_delta_ci95_low",
    "rmse_delta_ci95_high",
    "rmse_wins",
    "rmse_ties",
    "rmse_losses",
]


def rolling_origin_folds(
    df: pd.DataFrame,
    test_seasons: int = ROLLING_ORIGIN_TEST_SEASONS,
    min_train_seasons: int = 20,
) -> list[EvaluationFold]:
    """Create immutable chronological fold records with whole seasons intact."""
    evaluation_df = df.dropna(subset=["champ_position"]).copy()
    seasons = sorted(int(year) for year in evaluation_df["year"].unique())
    first_test_index = max(min_train_seasons, len(seasons) - test_seasons)
    folds = []
    for test_year in seasons[first_test_index:]:
        train_df = evaluation_df[evaluation_df["year"] < test_year]
        test_df = evaluation_df[evaluation_df["year"] == test_year]
        train_years = tuple(sorted(int(year) for year in train_df["year"].unique()))
        if len(train_years) < min_train_seasons or test_df.empty:
            continue
        folds.append(
            EvaluationFold(
                test_year=test_year,
                train_end_year=train_years[-1],
                train=train_df,
                test=test_df,
                train_years=train_years,
            )
        )
    return folds


def _rolling_cutoffs(
    df: pd.DataFrame,
    test_seasons: int,
    min_train_seasons: int,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame, list[int]]]:
    """Compatibility tuple view of the explicit rolling-fold records."""
    return [
        tuple(fold)
        for fold in rolling_origin_folds(df, test_seasons, min_train_seasons)
    ]


def assign_tier(pos: float | int | None) -> str:
    """Map a championship position to its reporting tier."""
    if pd.isna(pos):
        return "Unknown"
    p = int(pos)
    # These bands mirror the project’s reporting categories: champion, podium, top 5, top 10, midfield, backmarker.
    if p == 1:
        return "Champion"
    elif p <= 3:
        return "Podium"
    elif p <= 5:
        return "Top 5"
    elif p <= 10:
        return "Top 10"
    elif p <= 15:
        return "Midfield"
    else:
        return "Backmarker"


def get_spearman(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Return the Spearman rank correlation for true and predicted values."""
    corr_val, _ = spearmanr(y_true, y_pred)
    return float(corr_val)  # type: ignore


def previous_season_final_order(test_df: pd.DataFrame) -> pd.Series:
    """Predict current order by ranking entrants on prior-season total points.

    This is a deliberately simple pre-season baseline. Drivers without a
    prior-season record receive zero points, sprint points are included, and
    ties use source-row order.
    """
    previous_points = test_df["prev_season_points_sum"].fillna(0)
    if "prev_season_sprint_points_sum" in test_df:
        previous_points = previous_points + test_df[
            "prev_season_sprint_points_sum"
        ].fillna(0)
    return previous_points.rank(method="first", ascending=False)


def evaluate_tier_rolling_origin(
    df: pd.DataFrame,
    test_seasons: int = ROLLING_ORIGIN_TEST_SEASONS,
    min_train_seasons: int = 20,
) -> pd.DataFrame:
    """Evaluate tier classification on successive future seasons.

    Each row is scored only after a classifier trained on strictly earlier
    seasons. The six class F1 values are retained for transparent reporting.
    """
    rows: list[dict[str, float | int | str]] = []

    for fold in rolling_origin_folds(df, test_seasons, min_train_seasons):
        test_year = fold.test_year
        train_df = fold.train
        test_df = fold.test
        train_years = fold.train_years
        classifier = create_tier_classifier(random_state=42)
        X_train = train_df[FEATURE_COLUMNS].fillna(0)
        y_train = train_df["champ_position"].apply(assign_tier)
        X_test = test_df[FEATURE_COLUMNS].fillna(0)
        y_test = test_df["champ_position"].apply(assign_tier)

        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        class_f1 = f1_score(
            y_test,
            predictions,
            labels=TIER_LABELS,
            average=None,
            zero_division=0,
        )
        row: dict[str, float | int | str] = {
            "test_year": test_year,
            "train_end_year": train_years[-1],
            "model": "Random Forest",
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(
                f1_score(
                    y_test,
                    predictions,
                    labels=TIER_LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),
            "confusion_matrix_json": json.dumps(
                confusion_matrix(y_test, predictions, labels=TIER_LABELS).tolist()
            ),
            "actual_support_json": json.dumps(
                y_test.value_counts()
                .reindex(TIER_LABELS, fill_value=0)
                .astype(int)
                .to_dict()
            ),
            "predicted_support_json": json.dumps(
                pd.Series(predictions)
                .value_counts()
                .reindex(TIER_LABELS, fill_value=0)
                .astype(int)
                .to_dict()
            ),
        }
        row.update(
            {
                f"f1_{tier.lower().replace(' ', '_')}": float(score)
                for tier, score in zip(TIER_LABELS, class_f1, strict=False)
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


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
        pd.DataFrame(rows, columns=PAIRED_COMPARISON_COLUMNS)
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
    candidates["passes_baseline_gate"] = (
        candidates["mean_spearman"] >= baseline - tolerance
    )
    return {
        "baseline_model": NAIVE_BASELINE_NAME,
        "baseline_mean_spearman": baseline,
        "tolerance": tolerance,
        "candidate_models": candidates[
            ["model", "mean_spearman", "passes_baseline_gate"]
        ].to_dict("records"),
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
                        "predicted_support": int(
                            predicted_support.get(predicted_label, 0)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_rolling_origin_with_failures(
    df: pd.DataFrame,
    test_seasons: int = ROLLING_ORIGIN_TEST_SEASONS,
    min_train_seasons: int = 20,
    candidate_factory: Callable[[], dict[str, tuple[Any, list[str]]]] | None = None,
) -> EvaluationRun:
    """Retain successful baseline/model rows and report isolated fold failures."""
    details = []
    failures = []
    candidate_factory = (
        create_regression_candidates if candidate_factory is None else candidate_factory
    )
    for fold in rolling_origin_folds(df, test_seasons, min_train_seasons):
        y_train = fold.train["champ_position"]
        y_test = fold.test["champ_position"]
        naive_predictions = previous_season_final_order(fold.test)
        naive_spearman = get_spearman(y_test, naive_predictions)
        baseline_predictions = {
            NAIVE_BASELINE_NAME: naive_predictions,
            "Baseline: previous avg finish": fold.test[
                "prev_season_avg_finish_pos"
            ].fillna(y_train.median()),
        }
        for name, predictions in baseline_predictions.items():
            spearman = get_spearman(y_test, predictions)
            details.append(
                {
                    "test_year": fold.test_year,
                    "train_end_year": fold.train_end_year,
                    "model": name,
                    "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                    "r2": float(r2_score(y_test, predictions)),
                    "spearman": spearman,
                    "spearman_delta_vs_naive": spearman - naive_spearman,
                }
            )
        for name, (model, feature_columns) in candidate_factory().items():
            try:
                X_train = fold.train[feature_columns].fillna(0)
                X_test = fold.test[feature_columns].fillna(0)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                spearman = get_spearman(y_test, predictions)
                details.append(
                    {
                        "test_year": fold.test_year,
                        "train_end_year": fold.train_end_year,
                        "model": name,
                        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                        "r2": float(r2_score(y_test, predictions)),
                        "spearman": spearman,
                        "spearman_delta_vs_naive": spearman - naive_spearman,
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "test_year": fold.test_year,
                        "train_end_year": fold.train_end_year,
                        "model": name,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                continue
    return EvaluationRun(
        details=pd.DataFrame(details),
        failures=pd.DataFrame(failures, columns=EVALUATION_FAILURE_COLUMNS),
    )


def evaluate_rolling_origin(
    df: pd.DataFrame,
    test_seasons: int = ROLLING_ORIGIN_TEST_SEASONS,
    min_train_seasons: int = 20,
) -> pd.DataFrame:
    """Compatibility DataFrame API for rolling-origin regression metrics."""
    run = evaluate_rolling_origin_with_failures(df, test_seasons, min_train_seasons)
    run.details.attrs["failures"] = run.failures
    return run.details
