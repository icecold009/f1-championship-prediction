import numpy as np
import pandas as pd

from scripts import evaluate as evaluate_script
from scripts.evaluate import (
    run_evaluation,
    summarize_paired_comparisons,
    validate_baseline_gate,
)
from scripts.model_audit import evaluate_permutation_importance, evaluate_uncertainty
from src.evaluation import (
    NAIVE_BASELINE_NAME,
    evaluate_rolling_origin_with_failures,
    rolling_origin_folds,
    summarize_results,
    summarize_tier_classes,
)
from src.model import FEATURE_COLUMNS


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


def test_paired_summary_uses_matching_seasons_and_counts_ties():
    details = pd.DataFrame(
        {
            "test_year": [2020, 2021, 2020, 2022],
            "model": [NAIVE_BASELINE_NAME, NAIVE_BASELINE_NAME, "Model", "Model"],
            "rmse": [3.0, 4.0, 3.0, 1.0],
            "spearman": [0.8, 0.7, 0.8, 1.0],
        }
    )

    summary = summarize_paired_comparisons(details).iloc[0]

    assert summary["test_seasons"] == 1
    assert summary["spearman_ties"] == 1
    assert summary["rmse_ties"] == 1
    assert summary["spearman_wins"] == 0
    assert summary["rmse_wins"] == 0


def test_paired_summary_has_stable_schema_when_no_candidates_succeed():
    details = pd.DataFrame(columns=["test_year", "model", "rmse", "spearman"])

    summary = summarize_paired_comparisons(details)

    assert summary.empty
    assert summary.columns.tolist() == [
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


def test_summaries_keep_rounding_and_ranking_order():
    details = pd.DataFrame(
        {
            "test_year": [2020, 2020],
            "model": ["Higher raw score", "Lower raw score"],
            "rmse": [1.1234, 2.5678],
            "r2": [0.1234, 0.5678],
            "spearman": [0.5014, 0.5006],
            "spearman_delta_vs_naive": [0.1, 0.2],
        }
    )

    summary = summarize_results(details)

    assert summary["model"].tolist() == ["Higher raw score", "Lower raw score"]
    assert summary["mean_spearman"].tolist() == [0.501, 0.501]


def test_tier_summary_keeps_empty_tiers_with_zero_f1():
    details = pd.DataFrame(
        {
            "test_year": [2020],
            "f1_champion": [1.0],
            "f1_podium": [0.0],
            "f1_top_5": [0.0],
            "f1_top_10": [0.0],
            "f1_midfield": [0.0],
            "f1_backmarker": [0.0],
        }
    )

    summary = summarize_tier_classes(details)

    assert summary["tier"].tolist() == [
        "Champion",
        "Podium",
        "Top 5",
        "Top 10",
        "Midfield",
        "Backmarker",
    ]
    assert summary.loc[summary["tier"] == "Backmarker", "mean_f1"].iloc[0] == 0


def test_explicit_folds_retain_strictly_earlier_training_seasons():
    features = _synthetic_features().loc[lambda frame: frame["year"] != 2012]
    folds = rolling_origin_folds(features, test_seasons=2, min_train_seasons=3)

    assert [fold.test_year for fold in folds] == [2014, 2015]
    for fold in folds:
        assert fold.train_end_year == max(fold.train_years)
        assert max(fold.train_years) < fold.test_year
        assert 2012 not in fold.train_years
        assert set(fold.train["year"]) == set(fold.train_years)
        assert set(fold.test["year"]) == {fold.test_year}


def test_one_failed_model_fold_is_reported_without_dropping_baselines():
    class FailOnce:
        def __init__(self):
            self.fit_count = 0

        def fit(self, features, targets):
            self.fit_count += 1
            if self.fit_count == 1:
                raise ValueError("synthetic fold failure")
            self.value = float(targets.mean())
            return self

        def predict(self, features):
            return features["prev_season_races_started"].to_numpy()

    estimator = FailOnce()
    run = evaluate_rolling_origin_with_failures(
        _synthetic_features(),
        test_seasons=2,
        min_train_seasons=3,
        candidate_factory=lambda: {"Flaky model": (estimator, FEATURE_COLUMNS)},
    )

    assert list(run.failures.columns) == [
        "test_year",
        "train_end_year",
        "model",
        "error_type",
        "error_message",
    ]
    assert run.failures.to_dict("records") == [
        {
            "test_year": 2014,
            "train_end_year": 2013,
            "model": "Flaky model",
            "error_type": "ValueError",
            "error_message": "synthetic fold failure",
        }
    ]
    assert set(
        run.details.loc[run.details["model"] == NAIVE_BASELINE_NAME, "test_year"]
    ) == {2014, 2015}
    assert run.details.loc[
        run.details["model"] == "Flaky model", "test_year"
    ].tolist() == [2015]
    summary = summarize_results(run.details)
    assert NAIVE_BASELINE_NAME in summary["model"].tolist()


def test_invalid_candidate_predictions_are_isolated_during_metric_scoring():
    class WrongLengthPrediction:
        def fit(self, features, targets):
            return self

        def predict(self, features):
            return np.zeros(max(0, len(features) - 1))

    run = evaluate_rolling_origin_with_failures(
        _synthetic_features(),
        test_seasons=2,
        min_train_seasons=3,
        candidate_factory=lambda: {
            "Invalid prediction": (WrongLengthPrediction(), FEATURE_COLUMNS)
        },
    )

    assert run.failures["test_year"].tolist() == [2014, 2015]
    assert run.failures["model"].tolist() == [
        "Invalid prediction",
        "Invalid prediction",
    ]
    assert run.failures["error_type"].tolist() == ["ValueError", "ValueError"]
    assert set(
        run.details.loc[run.details["model"] == NAIVE_BASELINE_NAME, "test_year"]
    ) == {2014, 2015}
    assert "Invalid prediction" not in run.details["model"].tolist()


def test_baseline_gate_reports_candidate_failure_without_hiding_baseline():
    summary = pd.DataFrame(
        {
            "model": [NAIVE_BASELINE_NAME, "Random Forest"],
            "mean_spearman": [0.82, 0.81],
        }
    )

    decision = validate_baseline_gate(summary)

    assert decision["baseline_model"] == NAIVE_BASELINE_NAME
    assert decision["passed"] is False
    assert decision["candidate_models"][0]["passes_baseline_gate"] is False


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
    assert (tmp_path / "tier_rolling_origin_confusion.csv").exists()
    failures = pd.read_csv(tmp_path / "rolling_origin_summary_failures.csv")
    assert list(failures.columns) == [
        "test_year",
        "train_end_year",
        "model",
        "error_type",
        "error_message",
    ]
    assert failures.empty


def test_pipeline_saves_failures_when_every_candidate_fails(tmp_path, monkeypatch):
    class AlwaysFail:
        def fit(self, features, targets):
            raise RuntimeError("synthetic model failure")

    monkeypatch.setattr(
        evaluate_script._evaluation,
        "create_regression_candidates",
        lambda: {"Failing model": (AlwaysFail(), FEATURE_COLUMNS)},
    )
    features_path = tmp_path / "features.csv"
    output_path = tmp_path / "rolling_origin_summary.csv"
    _synthetic_features().to_csv(features_path, index=False)

    run_evaluation(
        features_path=features_path,
        output_path=output_path,
        test_seasons=2,
        min_train_seasons=3,
    )

    failures = pd.read_csv(tmp_path / "rolling_origin_summary_failures.csv")
    summary = pd.read_csv(output_path)
    assert failures["test_year"].tolist() == [2014, 2015]
    assert failures["model"].tolist() == ["Failing model", "Failing model"]
    assert failures["error_message"].tolist() == [
        "synthetic model failure",
        "synthetic model failure",
    ]
    assert summary["model"].tolist() == [
        "Baseline: previous avg finish",
        NAIVE_BASELINE_NAME,
    ]
