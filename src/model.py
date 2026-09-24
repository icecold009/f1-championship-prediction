import logging
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

try:
    from . import evaluation as _evaluation
    from .data_pipeline import split_features
    from .evaluation import (
        assign_tier,
        evaluate_rolling_origin,
        evaluate_tier_rolling_origin,
    )
    from .model_registry import (
        COLD_START_FEATURE_COLUMNS,  # noqa: F401 - re-exported compatibility names.
        FEATURE_COLUMNS,
        HISTORY_FEATURE_COLUMNS,  # noqa: F401 - re-exported compatibility names.
        create_bootstrap_regressor,
        create_regression_candidates,
        create_tier_classifier,
        fit_model,
        save_model_artifacts,
    )
except ImportError:  # Script entrypoints import src modules from the src path.
    import evaluation as _evaluation
    from data_pipeline import split_features
    from evaluation import (
        assign_tier,
        evaluate_rolling_origin,
        evaluate_tier_rolling_origin,
    )
    from model_registry import (
        COLD_START_FEATURE_COLUMNS,  # noqa: F401 - re-exported compatibility names.
        FEATURE_COLUMNS,
        HISTORY_FEATURE_COLUMNS,  # noqa: F401 - re-exported compatibility names.
        create_bootstrap_regressor,
        create_regression_candidates,
        create_tier_classifier,
        fit_model,
        save_model_artifacts,
    )

TIER_LABELS = _evaluation.TIER_LABELS
ROLLING_ORIGIN_TEST_SEASONS = _evaluation.ROLLING_ORIGIN_TEST_SEASONS
NAIVE_BASELINE_NAME = _evaluation.NAIVE_BASELINE_NAME
_rolling_cutoffs = _evaluation._rolling_cutoffs
get_spearman = _evaluation.get_spearman
previous_season_final_order = _evaluation.previous_season_final_order
evaluate_rolling_origin_with_failures = (
    _evaluation.evaluate_rolling_origin_with_failures
)

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def _regression_candidates() -> dict[str, tuple[object, list[str]]]:
    """Create fresh estimators and their predeclared feature sets."""
    return create_regression_candidates()


def bootstrap_position_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_bootstrap: int = 100,
    n_estimators: int = 200,
    random_state: int = 42,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Estimate position uncertainty with season-level bootstrap models.

    Whole historical seasons are sampled with replacement for each fit. This
    preserves within-season rows and rejects any train/test season overlap.
    Returned probabilities are the share of bootstrap rankings placing each
    driver first, in the top three, or in the top five.
    """
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")

    train_seasons = sorted(int(year) for year in train_df["year"].unique())
    test_seasons = sorted(int(year) for year in test_df["year"].unique())
    if not train_seasons or not test_seasons:
        raise ValueError("Bootstrap training and test data must contain seasons")
    if set(train_seasons).intersection(test_seasons):
        raise ValueError("Bootstrap train and test seasons must not overlap")
    if max(train_seasons) >= min(test_seasons):
        raise ValueError("Bootstrap training seasons must precede test seasons")

    random_source = rng if rng is not None else np.random.default_rng(random_state)
    X_test = test_df[FEATURE_COLUMNS].fillna(0)
    bootstrap_predictions = np.empty((n_bootstrap, len(test_df)), dtype=float)

    season_rows = {
        season: train_df[train_df["year"] == season] for season in train_seasons
    }
    for run in range(n_bootstrap):
        sampled_seasons = random_source.choice(
            train_seasons, size=len(train_seasons), replace=True
        )
        sampled_train = pd.concat(
            [season_rows[season] for season in sampled_seasons],
            ignore_index=True,
        )
        model = create_bootstrap_regressor(
            n_estimators=n_estimators,
            random_state=random_state + run,
            factory=RandomForestRegressor,
        )
        model.fit(
            sampled_train[FEATURE_COLUMNS].fillna(0), sampled_train["champ_position"]
        )
        bootstrap_predictions[run] = model.predict(X_test)

    bootstrap_ranks = pd.DataFrame(bootstrap_predictions).rank(
        axis=1, method="first", ascending=True
    )
    return pd.DataFrame(
        {
            "bootstrap_runs": n_bootstrap,
            "bootstrap_position_mean": bootstrap_predictions.mean(axis=0),
            "bootstrap_position_sd": bootstrap_predictions.std(axis=0, ddof=1),
            "bootstrap_position_p05": np.quantile(bootstrap_predictions, 0.05, axis=0),
            "bootstrap_position_p95": np.quantile(bootstrap_predictions, 0.95, axis=0),
            "champion_probability": (bootstrap_ranks == 1).mean(axis=0).to_numpy(),
            "top_3_probability": (bootstrap_ranks <= 3).mean(axis=0).to_numpy(),
            "top_5_probability": (bootstrap_ranks <= 5).mean(axis=0).to_numpy(),
        },
        index=test_df.index,
    )


def _rolling_cutoffs(
    df: pd.DataFrame,
    test_seasons: int,
    min_train_seasons: int,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame, list[int]]]:
    """Return chronological cutoffs with whole seasons kept intact."""
    evaluation_df = df.dropna(subset=["champ_position"]).copy()
    seasons = sorted(int(year) for year in evaluation_df["year"].unique())
    first_test_index = max(min_train_seasons, len(seasons) - test_seasons)
    cutoffs = []

    for test_year in seasons[first_test_index:]:
        train_df = evaluation_df[evaluation_df["year"] < test_year]
        test_df = evaluation_df[evaluation_df["year"] == test_year]
        train_years = sorted(int(year) for year in train_df["year"].unique())
        if len(train_years) < min_train_seasons or test_df.empty:
            continue
        cutoffs.append((test_year, train_df, test_df, train_years))

    return cutoffs


def train_model(
    forecast_year: int | None = None,
    skip_rolling_evaluation: bool = False,
) -> tuple[object | None, RandomForestClassifier]:
    """Train and save models, optionally skipping rolling-origin evaluation."""
    # ── Load ──────────────────────────────────────────────────────────────
    features_path = os.path.join(PROC_DIR, "features.csv")
    df = pd.read_csv(features_path)
    df = df.dropna(subset=["champ_position"])
    logger.info(
        "Loaded %s rows across %s seasons (%s–%s)",
        len(df),
        df["year"].nunique(),
        df["year"].min(),
        df["year"].max(),
    )

    partition = split_features(df, forecast_year)
    train_df = partition.train
    test_df = partition.test
    forecast_label = partition.label
    if train_df.empty:
        raise ValueError(f"No training seasons are available before {forecast_year}.")

    logger.info(
        "Time-based split (%s): train seasons %s–%s (%s seasons), test seasons %s–%s (%s seasons)",
        forecast_label,
        train_df["year"].min(),
        train_df["year"].max(),
        train_df["year"].nunique(),
        test_df["year"].min(),
        test_df["year"].max(),
        test_df["year"].nunique(),
    )

    X_train = train_df[FEATURE_COLUMNS].fillna(0)
    y_train = train_df["champ_position"]
    yt_train = train_df["champ_position"].apply(assign_tier)

    if not skip_rolling_evaluation:
        # ── Leak-free rolling-origin metrics ──────────────────────────────
        rolling_results = evaluate_rolling_origin(df)
        failures = rolling_results.attrs.get("failures")
        if failures is not None and not failures.empty:
            logger.warning("%s rolling-origin model folds failed", len(failures))
        logger.info(
            "\n── Rolling-origin backtest (%s seasons; baselines included) ───",
            rolling_results["test_year"].nunique(),
        )
        rolling_summary = rolling_results.groupby("model")[
            ["rmse", "r2", "spearman", "spearman_delta_vs_naive"]
        ].agg(["mean", "std"])
        for model_name, metrics in rolling_summary.iterrows():
            logger.info(
                "  %-24s | RMSE: %.3f +/- %.3f | R²: %.3f +/- %.3f | Spearman: %.3f +/- %.3f | Δ naive: %.3f +/- %.3f",
                model_name,
                metrics[("rmse", "mean")],
                metrics[("rmse", "std")],
                metrics[("r2", "mean")],
                metrics[("r2", "std")],
                metrics[("spearman", "mean")],
                metrics[("spearman", "std")],
                metrics[("spearman_delta_vs_naive", "mean")],
                metrics[("spearman_delta_vs_naive", "std")],
            )

        tier_results = evaluate_tier_rolling_origin(df)
        tier_summary = tier_results.groupby("model").agg(
            test_seasons=("test_year", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
        )
        logger.info("\n── Tier classification walk-forward metrics ────────────────")
        logger.info("\n%s", tier_summary.round(3).to_string())

    # ── Predeclared regression model for the user-facing forecast ─────────
    candidates = _regression_candidates()
    best_name = "Random Forest + cold-start flags"
    best_model, operational_features = candidates[best_name]
    X_train = train_df[operational_features].fillna(0)
    fit_model(best_model, X_train, y_train)  # type: ignore[arg-type]
    logger.info("\n  Operational forecast model: %s (predeclared)", best_name)

    if best_model is not None and hasattr(best_model, "feature_importances_"):
        imp = pd.Series(
            best_model.feature_importances_, index=FEATURE_COLUMNS
        ).sort_values(ascending=False)
        logger.info("\n  Top 10 features:\n%s", imp.head(10).to_string())

    # ── Tier classifier for the user-facing forecast ───────────────────────
    clf = create_tier_classifier(random_state=42, factory=RandomForestClassifier)
    fit_model(clf, X_train, yt_train)  # type: ignore[arg-type]

    # ── Save ──────────────────────────────────────────────────────────────
    reg_path, clf_path = save_model_artifacts(best_model, clf, MODEL_DIR)

    logger.info("\n  Saved regression model → %s", reg_path)
    logger.info("  Saved tier classifier  → %s", clf_path)
    return best_model, clf


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_model()
