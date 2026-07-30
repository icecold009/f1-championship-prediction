import logging
import os
import pickle
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLUMNS = [
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

TIER_LABELS = ["Champion", "Podium", "Top 5", "Top 10", "Midfield", "Backmarker"]
ROLLING_ORIGIN_TEST_SEASONS = 10


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


def _regression_candidates() -> dict[str, object]:
    """Create fresh regression estimators for one evaluation run."""
    return {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, random_state=42
        ),
    }


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
        X_train = train_df[FEATURE_COLUMNS].fillna(0)
        y_train = train_df["champ_position"]
        X_test = test_df[FEATURE_COLUMNS].fillna(0)
        y_test = test_df["champ_position"]

        baseline_predictions = {
            "Baseline: previous avg finish": test_df[
                "prev_season_avg_finish_pos"
            ].fillna(y_train.median()),
            "Baseline: previous points rank": test_df[
                "prev_season_points_sum"
            ].fillna(0).rank(method="first", ascending=False),
        }
        for name, predictions in baseline_predictions.items():
            rows.append(
                {
                    "test_year": test_year,
                    "train_end_year": train_years[-1],
                    "model": name,
                    "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                    "r2": float(r2_score(y_test, predictions)),
                    "spearman": get_spearman(y_test, predictions),
                }
            )

        for name, model in _regression_candidates().items():
            model.fit(X_train, y_train)  # type: ignore[attr-defined]
            predictions = model.predict(X_test)  # type: ignore[attr-defined]
            rows.append(
                {
                    "test_year": test_year,
                    "train_end_year": train_years[-1],
                    "model": name,
                    "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                    "r2": float(r2_score(y_test, predictions)),
                    "spearman": get_spearman(y_test, predictions),
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
                for tier, score in zip(TIER_LABELS, class_f1)
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def train_model() -> tuple[object | None, RandomForestClassifier]:
    """Train, evaluate, and save the regression and tier-classification models."""
    # ── Load ──────────────────────────────────────────────────────────────
    features_path = os.path.join(PROC_DIR, "features.csv")
    df = pd.read_csv(features_path)
    df = df.dropna(subset=["champ_position"])
    logger.info("Loaded %s rows across %s seasons (%s–%s)", len(df), df["year"].nunique(), df["year"].min(), df["year"].max())

    test_years = sorted(df["year"].unique())[-5:]
    test_start_year = test_years[0]
    train_df = df[df["year"] < test_start_year]
    test_df = df[df["year"] >= test_start_year]

    logger.info(
        "Time-based split: train seasons %s–%s (%s seasons), test seasons %s–%s (%s seasons)",
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

    # ── Leak-free rolling-origin metrics ──────────────────────────────────
    rolling_results = evaluate_rolling_origin(df)
    logger.info(
        "\n── Rolling-origin backtest (%s seasons; baselines included) ───",
        rolling_results["test_year"].nunique(),
    )
    rolling_summary = rolling_results.groupby("model")[["rmse", "r2", "spearman"]].agg(["mean", "std"])
    for model_name, metrics in rolling_summary.iterrows():
        logger.info(
            "  %-24s | RMSE: %.3f +/- %.3f | R²: %.3f +/- %.3f | Spearman: %.3f +/- %.3f",
            model_name,
            metrics[("rmse", "mean")],
            metrics[("rmse", "std")],
            metrics[("r2", "mean")],
            metrics[("r2", "std")],
            metrics[("spearman", "mean")],
            metrics[("spearman", "std")],
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
    best_name = "Random Forest"
    best_model = candidates[best_name]
    best_model.fit(X_train, y_train)  # type: ignore[attr-defined]
    logger.info("\n  Operational forecast model: %s (predeclared)", best_name)

    if best_model is not None and hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
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
