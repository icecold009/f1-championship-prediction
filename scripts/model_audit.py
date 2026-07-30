"""Backtest forecast uncertainty and held-out permutation importance."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from model import (
    FEATURE_COLUMNS,
    _rolling_cutoffs,
    bootstrap_position_predictions,
)

logger = logging.getLogger(__name__)
DEFAULT_FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results"


def _new_model(n_estimators: int = 200) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )


def generate_point_oof_errors(
    features: pd.DataFrame,
    min_train_seasons: int = 20,
    n_estimators: int = 200,
) -> pd.DataFrame:
    """Generate expanding-window residuals for conformal calibration."""
    eligible_seasons = int(features["year"].nunique())
    rows = []
    for test_year, train_df, test_df, _ in _rolling_cutoffs(
        features,
        test_seasons=eligible_seasons,
        min_train_seasons=min_train_seasons,
    ):
        model = _new_model(n_estimators)
        model.fit(
            train_df[FEATURE_COLUMNS].fillna(0),
            train_df["champ_position"],
        )
        predictions = model.predict(test_df[FEATURE_COLUMNS].fillna(0))
        rows.append(
            pd.DataFrame(
                {
                    "test_year": test_year,
                    "absolute_residual": np.abs(
                        test_df["champ_position"].to_numpy() - predictions
                    ),
                }
            )
        )
    if not rows:
        raise ValueError("No out-of-fold residuals were produced.")
    return pd.concat(rows, ignore_index=True)


def _conformal_quantile(residuals: np.ndarray, coverage: float) -> float:
    """Return the finite-sample split-conformal absolute-residual quantile."""
    if residuals.size == 0:
        raise ValueError("Conformal calibration requires prior residuals.")
    level = min(1.0, np.ceil((residuals.size + 1) * coverage) / residuals.size)
    return float(np.quantile(residuals, level, method="higher"))


def evaluate_uncertainty(
    features: pd.DataFrame,
    test_seasons: int = 10,
    min_train_seasons: int = 20,
    coverage: float = 0.90,
    n_bootstrap: int = 100,
    n_estimators: int = 200,
    bootstrap_estimators: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest bootstrap probabilities and rolling conformal intervals."""
    oof_errors = generate_point_oof_errors(
        features,
        min_train_seasons=min_train_seasons,
        n_estimators=n_estimators,
    )
    rows = []
    for test_year, train_df, test_df, train_years in _rolling_cutoffs(
        features, test_seasons, min_train_seasons
    ):
        model = _new_model(n_estimators)
        model.fit(
            train_df[FEATURE_COLUMNS].fillna(0),
            train_df["champ_position"],
        )
        point_predictions = model.predict(test_df[FEATURE_COLUMNS].fillna(0))
        uncertainty = bootstrap_position_predictions(
            train_df,
            test_df,
            n_bootstrap=n_bootstrap,
            n_estimators=bootstrap_estimators,
            random_state=42 + test_year,
        ).reset_index(drop=True)
        prior_residuals = oof_errors.loc[
            oof_errors["test_year"] < test_year, "absolute_residual"
        ].to_numpy()
        conformal_q = _conformal_quantile(prior_residuals, coverage)
        season_size = len(test_df)
        output = pd.DataFrame(
            {
                "test_year": test_year,
                "train_end_year": train_years[-1],
                "driverId": test_df["driverId"].to_numpy(),
                "actual_position": test_df["champ_position"].to_numpy(),
                "point_prediction": point_predictions,
                "bootstrap_p05": uncertainty["bootstrap_position_p05"],
                "bootstrap_p95": uncertainty["bootstrap_position_p95"],
                "top_3_probability": uncertainty["top_3_probability"],
                "champion_probability": uncertainty["champion_probability"],
                "conformal_q": conformal_q,
                "conformal_low": np.maximum(1, point_predictions - conformal_q),
                "conformal_high": np.minimum(
                    season_size, point_predictions + conformal_q
                ),
            }
        )
        output["bootstrap_covered"] = output["actual_position"].between(
            output["bootstrap_p05"], output["bootstrap_p95"]
        )
        output["conformal_covered"] = output["actual_position"].between(
            output["conformal_low"], output["conformal_high"]
        )
        output["actual_top_3"] = output["actual_position"].le(3).astype(int)
        output["actual_champion"] = output["actual_position"].eq(1).astype(int)
        rows.append(output)

    driver_details = pd.concat(rows, ignore_index=True)
    summary = pd.DataFrame(
        [
            {
                "test_seasons": driver_details["test_year"].nunique(),
                "observations": len(driver_details),
                "nominal_coverage": coverage,
                "bootstrap_runs": n_bootstrap,
                "trees_per_bootstrap_model": bootstrap_estimators,
                "point_model_trees": n_estimators,
                "bootstrap_interval_coverage": driver_details[
                    "bootstrap_covered"
                ].mean(),
                "bootstrap_mean_width": (
                    driver_details["bootstrap_p95"] - driver_details["bootstrap_p05"]
                ).mean(),
                "conformal_interval_coverage": driver_details[
                    "conformal_covered"
                ].mean(),
                "conformal_mean_width": (
                    driver_details["conformal_high"] - driver_details["conformal_low"]
                ).mean(),
                "top_3_brier_score": np.mean(
                    (
                        driver_details["top_3_probability"]
                        - driver_details["actual_top_3"]
                    )
                    ** 2
                ),
                "champion_brier_score": np.mean(
                    (
                        driver_details["champion_probability"]
                        - driver_details["actual_champion"]
                    )
                    ** 2
                ),
            }
        ]
    ).round(3)

    bins = pd.cut(
        driver_details["top_3_probability"],
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"],
    )
    calibration = (
        driver_details.assign(probability_bin=bins)
        .groupby("probability_bin", observed=False)
        .agg(
            observations=("actual_top_3", "size"),
            mean_predicted_probability=("top_3_probability", "mean"),
            observed_top_3_rate=("actual_top_3", "mean"),
        )
        .reset_index()
        .round(3)
    )
    return driver_details.round(4), summary, calibration


def evaluate_permutation_importance(
    features: pd.DataFrame,
    test_seasons: int = 10,
    min_train_seasons: int = 20,
    n_estimators: int = 200,
    repeats: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure feature importance on each untouched future season."""
    rows = []
    for test_year, train_df, test_df, train_years in _rolling_cutoffs(
        features, test_seasons, min_train_seasons
    ):
        model = _new_model(n_estimators)
        X_train = train_df[FEATURE_COLUMNS].fillna(0)
        X_test = test_df[FEATURE_COLUMNS].fillna(0)
        model.fit(X_train, train_df["champ_position"])
        result = permutation_importance(
            model,
            X_test,
            test_df["champ_position"],
            scoring="neg_root_mean_squared_error",
            n_repeats=repeats,
            random_state=42 + test_year,
            n_jobs=-1,
        )
        baseline_rmse = (
            mean_squared_error(
                test_df["champ_position"],
                model.predict(X_test),
            )
            ** 0.5
        )
        for feature, mean_value, sd_value in zip(
            FEATURE_COLUMNS,
            result.importances_mean,
            result.importances_std,
            strict=False,
        ):
            rows.append(
                {
                    "test_year": test_year,
                    "train_end_year": train_years[-1],
                    "feature": feature,
                    "baseline_rmse": baseline_rmse,
                    "rmse_increase_mean": mean_value,
                    "rmse_increase_sd": sd_value,
                }
            )
    details = pd.DataFrame(rows)
    summary = (
        details.groupby("feature")
        .agg(
            test_seasons=("test_year", "nunique"),
            mean_rmse_increase=("rmse_increase_mean", "mean"),
            season_sd=("rmse_increase_mean", "std"),
            positive_seasons=(
                "rmse_increase_mean",
                lambda values: int((values > 0).sum()),
            ),
        )
        .reset_index()
        .sort_values("mean_rmse_increase", ascending=False)
        .round(3)
    )
    return details.round(4), summary


def run_model_audit(
    features_path: Path = DEFAULT_FEATURES_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    test_seasons: int = 10,
    min_train_seasons: int = 20,
    n_bootstrap: int = 100,
    n_estimators: int = 200,
    bootstrap_estimators: int = 50,
) -> dict[str, Path]:
    """Generate all calibration and held-out interpretation artifacts."""
    features = pd.read_csv(features_path)
    uncertainty_details, uncertainty_summary, calibration = evaluate_uncertainty(
        features,
        test_seasons=test_seasons,
        min_train_seasons=min_train_seasons,
        n_bootstrap=n_bootstrap,
        n_estimators=n_estimators,
        bootstrap_estimators=bootstrap_estimators,
    )
    importance_details, importance_summary = evaluate_permutation_importance(
        features,
        test_seasons=test_seasons,
        min_train_seasons=min_train_seasons,
        n_estimators=n_estimators,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "uncertainty_details": output_dir / "uncertainty_calibration_driver.csv",
        "uncertainty_summary": output_dir / "uncertainty_calibration_summary.csv",
        "uncertainty_bins": output_dir / "uncertainty_calibration_bins.csv",
        "importance_details": output_dir / "permutation_importance_details.csv",
        "importance_summary": output_dir / "permutation_importance_summary.csv",
    }
    uncertainty_details.to_csv(outputs["uncertainty_details"], index=False)
    uncertainty_summary.to_csv(outputs["uncertainty_summary"], index=False)
    calibration.to_csv(outputs["uncertainty_bins"], index=False)
    importance_details.to_csv(outputs["importance_details"], index=False)
    importance_summary.to_csv(outputs["importance_summary"], index=False)
    for label, path in outputs.items():
        logger.info("Saved %s -> %s", label, path)
    return outputs


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-seasons", type=int, default=10)
    parser.add_argument("--min-train-seasons", type=int, default=20)
    parser.add_argument("--bootstrap-runs", type=int, default=100)
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--bootstrap-estimators", type=int, default=50)
    args = parser.parse_args()
    try:
        run_model_audit(
            features_path=args.features,
            output_dir=args.output_dir,
            test_seasons=args.test_seasons,
            min_train_seasons=args.min_train_seasons,
            n_bootstrap=args.bootstrap_runs,
            n_estimators=args.estimators,
            bootstrap_estimators=args.bootstrap_estimators,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Model audit failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
