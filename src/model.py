import logging
import os
import pickle
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, r2_score

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


def evaluate_rolling_origin(
    df: pd.DataFrame,
    test_seasons: int = ROLLING_ORIGIN_TEST_SEASONS,
    min_train_seasons: int = 20,
) -> pd.DataFrame:
    """Evaluate regressors on successive future seasons.

    For each test season, models train only on earlier seasons. The returned
    rows retain the train cutoff so the evaluation is auditable and cannot
    silently become a random split.
    """
    evaluation_df = df.dropna(subset=["champ_position"]).copy()
    seasons = sorted(int(year) for year in evaluation_df["year"].unique())
    first_test_index = max(min_train_seasons, len(seasons) - test_seasons)
    rows: list[dict[str, float | int | str]] = []

    for test_year in seasons[first_test_index:]:
        train_df = evaluation_df[evaluation_df["year"] < test_year]
        test_df = evaluation_df[evaluation_df["year"] == test_year]
        train_years = sorted(int(year) for year in train_df["year"].unique())
        if len(train_years) < min_train_seasons or test_df.empty:
            continue

        X_train = train_df[FEATURE_COLUMNS].fillna(0)
        y_train = train_df["champ_position"]
        X_test = test_df[FEATURE_COLUMNS].fillna(0)
        y_test = test_df["champ_position"]

        for name, model in _regression_candidates().items():
            model.fit(X_train, y_train)  # type: ignore[attr-defined]
            predictions = model.predict(X_test)  # type: ignore[attr-defined]
            rows.append(
                {
                    "test_year": test_year,
                    "train_end_year": train_years[-1],
                    "model": name,
                    "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                    "spearman": get_spearman(y_test, predictions),
                }
            )

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
    X_test = test_df[FEATURE_COLUMNS].fillna(0)
    y_train = train_df["champ_position"]
    y_test = test_df["champ_position"]
    yt_train = train_df["champ_position"].apply(assign_tier)
    yt_test = test_df["champ_position"].apply(assign_tier)
    train_groups = train_df["year"]
    group_cv = GroupKFold(n_splits=5)
    # Stratify tier labels while keeping every season entirely within one fold.
    classifier_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Regression models ─────────────────────────────────────────────────
    rolling_results = evaluate_rolling_origin(df)
    logger.info(
        "\n── Rolling-origin backtest (%s seasons) ─────────────────────",
        rolling_results["test_year"].nunique(),
    )
    rolling_summary = (
        rolling_results.groupby("model")[["rmse", "spearman"]]
        .agg(["mean", "std"])
        .sort_values(("spearman", "mean"), ascending=False)
    )
    for model_name, metrics in rolling_summary.iterrows():
        logger.info(
            "  %-20s | RMSE: %.3f +/- %.3f | Spearman: %.3f +/- %.3f",
            model_name,
            metrics[("rmse", "mean")],
            metrics[("rmse", "std")],
            metrics[("spearman", "mean")],
            metrics[("spearman", "std")],
        )

    logger.info("\n── Regression ────────────────────────────────────────────────")
    candidates = _regression_candidates()

    best_model = None
    best_name  = ""
    best_spearman = -999

    for name, m in candidates.items():
        cv = cross_val_score(
            m,
            X_train,
            y_train,
            cv=group_cv,
            groups=train_groups,
            scoring="neg_mean_squared_error",
        )
        cv_rmse = float(np.sqrt(-cv.mean()))

        m.fit(X_train, y_train)
        y_pred    = m.predict(X_test)
        test_r2   = float(r2_score(y_test, y_pred))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        sp        = get_spearman(y_test, y_pred)

        logger.info("  %-20s | CV RMSE: %.3f | R²: %.3f | Spearman: %.3f", name, cv_rmse, test_r2, sp)

        if sp > best_spearman:
            best_spearman = sp
            best_model    = m
            best_name     = name

    logger.info("\n  Best: %s (Spearman=%.3f)", best_name, best_spearman)

    if best_model is not None and hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
        logger.info("\n  Top 10 features:\n%s", imp.head(10).to_string())

    # ── Tier classifier ───────────────────────────────────────────────────
    logger.info("\n── Classification ────────────────────────────────────────────")
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    cv_clf = cross_val_score(
        clf,
        X_train,
        yt_train,
        cv=classifier_cv,
        groups=train_groups,
        scoring="accuracy",
    )
    clf.fit(X_train, yt_train)
    yt_pred = clf.predict(X_test)
    test_acc = clf.score(X_test, yt_test)
    report = classification_report(
        yt_test,
        yt_pred,
        labels=TIER_LABELS,
        output_dict=True,
        zero_division=0,
    )
    logger.info(
        "  Tier Classifier | Stratified Grouped CV Acc: %.3f | Test Acc: %.3f | Test Macro F1: %.3f",
        cv_clf.mean(),
        test_acc,
        report["macro avg"]["f1-score"],
    )
    logger.info("  Per-class F1:")
    for tier in TIER_LABELS:
        logger.info("    %-10s | F1: %.3f", tier, report[tier]["f1-score"])

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
