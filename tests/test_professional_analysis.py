import numpy as np
import pandas as pd

from scripts.evaluate import run_evaluation, summarize_paired_comparisons
from scripts.model_audit import evaluate_permutation_importance, evaluate_uncertainty
from src.model import FEATURE_COLUMNS, NAIVE_BASELINE_NAME


def _synthetic_features() -> pd.DataFrame:
    rows = []
    for year in range(2010, 2016):
        for driver_id, position in [(1, 1), (2, 2), (3, 3)]:
            row = {
                "year": year,
                "driverId": driver_id,
                "champ_position": position,
            }
            row.update(
                {
                    column: float(driver_id) + (year - 2010) / 100
                    for column in FEATURE_COLUMNS
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_paired_summary_reports_season_wins_and_interval():
    details = pd.DataFrame(
        {
            "test_year": [2022, 2023, 2022, 2023],
            "model": [NAIVE_BASELINE_NAME, NAIVE_BASELINE_NAME, "Model", "Model"],
            "rmse": [3.0, 4.0, 2.0, 5.0],
            "spearman": [0.8, 0.7, 0.9, 0.6],
        }
    )

    summary = summarize_paired_comparisons(details).iloc[0]

    assert summary["spearman_wins"] == 1
    assert summary["spearman_losses"] == 1
    assert summary["rmse_wins"] == 1
    assert summary["rmse_losses"] == 1
    assert np.isfinite(summary["spearman_delta_ci95_low"])


def test_uncertainty_and_permutation_audits_are_strictly_chronological():
    features = _synthetic_features()

    uncertainty, summary, _ = evaluate_uncertainty(
        features,
        test_seasons=2,
        min_train_seasons=3,
        n_bootstrap=3,
        n_estimators=3,
        bootstrap_estimators=3,
    )
    importance, importance_summary = evaluate_permutation_importance(
        features,
        test_seasons=2,
        min_train_seasons=3,
        n_estimators=3,
        repeats=2,
    )

    assert (uncertainty["train_end_year"] < uncertainty["test_year"]).all()
    assert 0 <= summary.iloc[0]["conformal_interval_coverage"] <= 1
    assert (importance["train_end_year"] < importance["test_year"]).all()
    assert set(importance_summary["feature"]) == set(FEATURE_COLUMNS)


def test_evaluation_pipeline_writes_professional_artifacts(tmp_path):
    features_path = tmp_path / "features.csv"
    output_path = tmp_path / "rolling_origin_summary.csv"
    _synthetic_features().to_csv(features_path, index=False)

    run_evaluation(
        features_path=features_path,
        output_path=output_path,
        test_seasons=2,
        min_train_seasons=3,
    )

    assert output_path.exists()
    assert (tmp_path / "rolling_origin_summary_details.csv").exists()
    assert (tmp_path / "model_vs_naive_summary.csv").exists()
    assert (tmp_path / "model_vs_naive_by_season.png").exists()
