import logging
import os
import pickle
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

HISTORY_FEATURE_COLUMNS = [
    "prev_season_races_started",
    "prev_season_avg_finish_pos",
    "prev_season_std_finish_pos",
    "prev_season_points_sum",
    "prev_season_avg_grid_pos",
    "prev_season_win_rate",
    "prev_season_podium_rate",
    "prev_season_dnf_rate",
    "prev_season_points_per_race",
    "prev_season_quali_to_race_delta",
    "prev_season_sprint_points_sum",
    "prev_team_final_points",
    "prev_team_final_position",
]
COLD_START_FEATURE_COLUMNS = [
    "is_rookie",
    "returning_after_gap",
    "missing_driver_history",
    "missing_constructor_history",
]
FEATURE_COLUMNS = [*HISTORY_FEATURE_COLUMNS, *COLD_START_FEATURE_COLUMNS]

TIER_LABELS = ["Champion", "Podium", "Top 5", "Top 10", "Midfield", "Backmarker"]
ROLLING_ORIGIN_TEST_SEASONS = 10
NAIVE_BASELINE_NAME = "Naive: previous-season final order"


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


def _regression_candidates() -> dict[str, tuple[object, list[str]]]:
    """Create fresh estimators and their predeclared feature sets."""
    return {
        "Ridge": (Ridge(alpha=1.0), FEATURE_COLUMNS),
        "Random Forest (history only)": (
            RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
            HISTORY_FEATURE_COLUMNS,
        ),
        "Random Forest + cold-start flags": (
            RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
            FEATURE_COLUMNS,
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
            FEATURE_COLUMNS,
        ),
    }


def previous_season_final_order(test_df: pd.DataFrame) -> pd.Series:
    """Predict current order by ranking entrants on prior-season points.

    This is a deliberately simple pre-season baseline. Drivers without a
    prior-season record receive zero points and ties use source-row order.
    """
    return (
        test_df["prev_season_points_sum"]
        .fillna(0)
        .rank(method="first", ascending=False)
    )


def bootstrap_position_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_bootstrap: int = 100,
    n_estimators: int = 200,
    random_state: int = 42,
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

    rng = np.random.default_rng(random_state)
    X_test = test_df[FEATURE_COLUMNS].fillna(0)
    bootstrap_predictions = np.empty((n_bootstrap, len(test_df)), dtype=float)

    season_rows = {
        season: train_df[train_df["year"] == season] for season in train_seasons
    }
    for run in range(n_bootstrap):
        sampled_seasons = rng.choice(
            train_seasons, size=len(train_seasons), replace=True
        )
        sampled_train = pd.concat(
            [season_rows[season] for season in sampled_seasons],
            ignore_index=True,
        )
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=random_state + run,
            n_jobs=-1,
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


def evaluate_rolling_origin(
    df: pd.DataFrame,
    test_seasons: int = ROLLING_ORIGIN_TEST_SEASONS,
    min_train_seasons: int = 20,
) -> pd.DataFrame:
    """Evaluate models and historical baselines on successive future seasons.

    For each test season, models train only on earlier seasons. The returned
    rows retain the train cutoff so the evaluation is auditable and cannot
    silently become a random split.
    """
    rows: list[dict[str, float | int | str]] = []

    for test_year, train_df, test_df, train_years in _rolling_cutoffs(
        df, test_seasons, min_train_seasons
    ):
        y_train = train_df["champ_position"]
        y_test = test_df["champ_position"]

        naive_predictions = previous_season_final_order(test_df)
        naive_spearman = get_spearman(y_test, naive_predictions)
        baseline_predictions = {
            NAIVE_BASELINE_NAME: naive_predictions,
            "Baseline: previous avg finish": test_df[
                "prev_season_avg_finish_pos"
            ].fillna(y_train.median()),
        }
        for name, predictions in baseline_predictions.items():
            spearman = get_spearman(y_test, predictions)
            rows.append(
                {
                    "test_year": test_year,
                    "train_end_year": train_years[-1],
                    "model": name,
                    "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                    "r2": float(r2_score(y_test, predictions)),
                    "spearman": spearman,
                    "spearman_delta_vs_naive": spearman - naive_spearman,
                }
            )

        for name, (model, feature_columns) in _regression_candidates().items():
            X_train = train_df[feature_columns].fillna(0)
            X_test = test_df[feature_columns].fillna(0)
            model.fit(X_train, y_train)  # type: ignore[attr-defined]
            predictions = model.predict(X_test)  # type: ignore[attr-defined]
            spearman = get_spearman(y_test, predictions)
            rows.append(
                {
                    "test_year": test_year,
                    "train_end_year": train_years[-1],
                    "model": name,
                    "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                    "r2": float(r2_score(y_test, predictions)),
                    "spearman": spearman,
                    "spearman_delta_vs_naive": spearman - naive_spearman,
                }
            )

    return pd.DataFrame(rows)


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

    for test_year, train_df, test_df, train_years in _rolling_cutoffs(
        df, test_seasons, min_train_seasons
    ):
        classifier = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        )
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
        }
        row.update(
            {
                f"f1_{tier.lower().replace(' ', '_')}": float(score)
                for tier, score in zip(TIER_LABELS, class_f1, strict=False)
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


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

    if forecast_year is None:
        test_years = sorted(df["year"].unique())[-5:]
        train_df = df[df["year"] < test_years[0]]
        test_df = df[df["year"] >= test_years[0]]
        forecast_label = f"latest five-season holdout beginning {test_years[0]}"
    else:
        train_df = df[df["year"] < forecast_year]
        test_df = df[df["year"] == forecast_year]
        forecast_label = f"forecast year {forecast_year}"
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
    best_model.fit(X_train, y_train)  # type: ignore[attr-defined]
    logger.info("\n  Operational forecast model: %s (predeclared)", best_name)

    if best_model is not None and hasattr(best_model, "feature_importances_"):
        imp = pd.Series(
            best_model.feature_importances_, index=FEATURE_COLUMNS
        ).sort_values(ascending=False)
        logger.info("\n  Top 10 features:\n%s", imp.head(10).to_string())

    # ── Tier classifier for the user-facing forecast ───────────────────────
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_train, yt_train)

    # ── Save ──────────────────────────────────────────────────────────────
    reg_path = os.path.join(MODEL_DIR, "championship_model.pkl")
    clf_path = os.path.join(MODEL_DIR, "tier_classifier.pkl")

    with open(reg_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)

    logger.info("\n  Saved regression model → %s", reg_path)
    logger.info("  Saved tier classifier  → %s", clf_path)
    return best_model, clf


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_model()
