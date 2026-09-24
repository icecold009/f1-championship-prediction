import hashlib
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src import model, model_registry, predict
from src.model import (
    FEATURE_COLUMNS,
    NAIVE_BASELINE_NAME,
    assign_tier,
    bootstrap_position_predictions,
    evaluate_rolling_origin,
    evaluate_tier_rolling_origin,
    get_spearman,
    previous_season_final_order,
)


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (1, "Champion"),
        (3, "Podium"),
        (5, "Top 5"),
        (10, "Top 10"),
        (15, "Midfield"),
        (16, "Backmarker"),
    ],
)
def test_assign_tier_boundaries(position, expected):
    assert assign_tier(position) == expected


def test_assign_tier_missing_position_is_unknown():
    assert assign_tier(None) == "Unknown"
    assert assign_tier(np.nan) == "Unknown"


def test_get_spearman_perfect_and_inverse_rankings():
    assert get_spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert get_spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)


def test_previous_season_final_order_is_an_explicit_naive_baseline():
    test_df = pd.DataFrame({"prev_season_points_sum": [100.0, 50.0, np.nan]})

    assert previous_season_final_order(test_df).tolist() == [1.0, 2.0, 3.0]


def test_previous_season_final_order_includes_sprint_points():
    test_df = pd.DataFrame(
        {
            "prev_season_points_sum": [100.0, 95.0],
            "prev_season_sprint_points_sum": [0.0, 10.0],
        }
    )

    assert previous_season_final_order(test_df).tolist() == [2.0, 1.0]


def test_bootstrap_position_predictions_returns_rank_probabilities_without_leakage():
    rows = []
    for year in range(2010, 2015):
        for driver_id, position in [(1, 1), (2, 2)]:
            row = {
                "year": year,
                "driverId": driver_id,
                "champ_position": position,
            }
            row.update({column: float(driver_id) for column in FEATURE_COLUMNS})
            rows.append(row)
    frame = pd.DataFrame(rows)

    train_df = frame[frame["year"] < 2014]
    test_df = frame[frame["year"] == 2014]
    uncertainty = bootstrap_position_predictions(
        train_df,
        test_df,
        n_bootstrap=4,
        n_estimators=3,
        random_state=42,
    )
    repeated = bootstrap_position_predictions(
        train_df,
        test_df,
        n_bootstrap=4,
        n_estimators=3,
        random_state=42,
    )
    injected_rng = bootstrap_position_predictions(
        train_df,
        test_df,
        n_bootstrap=4,
        n_estimators=3,
        random_state=42,
        rng=np.random.default_rng(42),
    )

    assert len(uncertainty) == 2
    pd.testing.assert_frame_equal(uncertainty, repeated)
    pd.testing.assert_frame_equal(uncertainty, injected_rng)
    assert uncertainty.index.tolist() == [8, 9]
    assert list(uncertainty.columns) == [
        "bootstrap_runs",
        "bootstrap_position_mean",
        "bootstrap_position_sd",
        "bootstrap_position_p05",
        "bootstrap_position_p95",
        "champion_probability",
        "top_3_probability",
        "top_5_probability",
    ]
    assert dict(
        zip(uncertainty.columns, uncertainty.dtypes.astype(str), strict=True)
    ) == {
        "bootstrap_runs": "int64",
        "bootstrap_position_mean": "float64",
        "bootstrap_position_sd": "float64",
        "bootstrap_position_p05": "float64",
        "bootstrap_position_p95": "float64",
        "champion_probability": "float64",
        "top_3_probability": "float64",
        "top_5_probability": "float64",
    }
    assert (uncertainty["bootstrap_runs"] == 4).all()
    assert uncertainty["champion_probability"].sum() == pytest.approx(1.0)
    assert uncertainty["top_3_probability"].sum() == pytest.approx(2.0)
    uncertainty_digest = hashlib.sha256(
        uncertainty.to_csv(
            index=False, float_format="%.17g", lineterminator="\n"
        ).encode("utf-8")
    ).hexdigest()
    assert (
        uncertainty_digest
        == "3ba1a53177c295f8a46fc5861b9d5e00f0b88dcf570804c240c5e9bfdf92684c"
    )

    estimator_seeds = {
        name: getattr(estimator, "random_state", None)
        for name, (estimator, _) in model._regression_candidates().items()
    }
    assert estimator_seeds == {
        "Ridge": None,
        "Random Forest (history only)": 42,
        "Random Forest + cold-start flags": 42,
        "Gradient Boosting": 42,
    }


def test_model_registry_preserves_names_features_and_estimator_defaults():
    candidates = model_registry.create_regression_candidates()

    assert list(candidates) == [
        "Ridge",
        "Random Forest (history only)",
        "Random Forest + cold-start flags",
        "Gradient Boosting",
    ]
    assert candidates["Ridge"][0].get_params()["alpha"] == 1.0
    assert candidates["Ridge"][1] is FEATURE_COLUMNS
    history_model, history_columns = candidates["Random Forest (history only)"]
    assert history_columns is model.HISTORY_FEATURE_COLUMNS
    assert history_model.get_params()["n_estimators"] == 200
    assert history_model.get_params()["max_depth"] == 10
    assert history_model.get_params()["random_state"] == 42
    operational_model, operational_columns = candidates[
        "Random Forest + cold-start flags"
    ]
    assert operational_columns is FEATURE_COLUMNS
    assert operational_model.get_params()["n_estimators"] == 200
    assert operational_model.get_params()["max_depth"] == 10
    assert operational_model.get_params()["random_state"] == 42
    boosting, _ = candidates["Gradient Boosting"]
    assert boosting.get_params()["n_estimators"] == 200
    assert boosting.get_params()["max_depth"] == 5
    assert boosting.get_params()["random_state"] == 42

    tier = model_registry.create_tier_classifier()
    assert tier.get_params()["n_estimators"] == 200
    assert tier.get_params()["max_depth"] == 8
    assert tier.get_params()["random_state"] == 42
    bootstrap = model_registry.create_bootstrap_regressor(3, random_state=45)
    assert bootstrap.get_params()["n_estimators"] == 3
    assert bootstrap.get_params()["max_depth"] == 10
    assert bootstrap.get_params()["random_state"] == 45
    assert bootstrap.get_params()["n_jobs"] == -1


def test_model_artifact_ports_round_trip_default_estimators(tmp_path):
    regression = model_registry.create_regression_candidates()["Ridge"][0]
    classifier = model_registry.create_tier_classifier()

    paths = model_registry.save_model_artifacts(regression, classifier, tmp_path)
    loaded_regression, loaded_classifier = model_registry.load_model_artifacts(tmp_path)

    assert paths == (
        tmp_path / "championship_model.pkl",
        tmp_path / "tier_classifier.pkl",
    )
    assert type(loaded_regression) is type(regression)
    assert type(loaded_classifier) is type(classifier)
    assert loaded_regression.get_params() == regression.get_params()
    assert loaded_classifier.get_params() == classifier.get_params()


def test_rolling_origin_keeps_test_seasons_after_training_cutoff():
    rows = []
    for year in range(2010, 2016):
        for driver_id, position in [(1, 1), (2, 2)]:
            row = {
                "year": year,
                "driverId": driver_id,
                "champ_position": position,
            }
            row.update(
                {column: float(driver_id) + year / 1000 for column in FEATURE_COLUMNS}
            )
            rows.append(row)

    results = evaluate_rolling_origin(
        pd.DataFrame(rows), test_seasons=2, min_train_seasons=3
    )
    expected_models = [
        NAIVE_BASELINE_NAME,
        "Baseline: previous avg finish",
        "Ridge",
        "Random Forest (history only)",
        "Random Forest + cold-start flags",
        "Gradient Boosting",
    ]

    assert list(results.columns) == [
        "test_year",
        "train_end_year",
        "model",
        "rmse",
        "r2",
        "spearman",
        "spearman_delta_vs_naive",
    ]
    assert list(
        zip(
            results["train_end_year"],
            results["test_year"],
            results["model"],
            strict=True,
        )
    ) == [
        *((2013, 2014, name) for name in expected_models),
        *((2014, 2015, name) for name in expected_models),
    ]
    assert dict(zip(results.columns, results.dtypes.astype(str), strict=True)) == {
        "test_year": "int64",
        "train_end_year": "int64",
        "model": "str",
        "rmse": "float64",
        "r2": "float64",
        "spearman": "float64",
        "spearman_delta_vs_naive": "float64",
    }
    metric_digest = hashlib.sha256(
        results.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode(
            "utf-8"
        )
    ).hexdigest()
    assert (
        metric_digest
        == "4e67683122b74eae5f2e0f9ef15d2f07a4ef95313efecfef88b57976a1791e9d"
    )
    assert set(results["test_year"]) == {2014, 2015}
    assert (results["train_end_year"] < results["test_year"]).all()
    assert set(results["model"]) == {
        "Ridge",
        "Random Forest (history only)",
        "Random Forest + cold-start flags",
        "Gradient Boosting",
        "Baseline: previous avg finish",
        NAIVE_BASELINE_NAME,
    }


def test_tier_rolling_origin_keeps_test_seasons_after_training_cutoff(monkeypatch):
    rows = []
    positions = [1, 3, 5, 10, 15, 16]
    for year in range(2010, 2016):
        for driver_id, position in enumerate(positions, start=1):
            row = {
                "year": year,
                "driverId": driver_id,
                "champ_position": position,
            }
            row.update({column: float(driver_id) for column in FEATURE_COLUMNS})
            rows.append(row)

    classifier_configs = []
    classifier_factory = model.RandomForestClassifier

    def record_classifier_config(*args, **kwargs):
        classifier_configs.append(kwargs.copy())
        return classifier_factory(*args, **kwargs)

    monkeypatch.setattr(model, "RandomForestClassifier", record_classifier_config)
    results = evaluate_tier_rolling_origin(
        pd.DataFrame(rows), test_seasons=2, min_train_seasons=3
    )
    assert classifier_configs == [
        {"n_estimators": 200, "max_depth": 8, "random_state": 42},
        {"n_estimators": 200, "max_depth": 8, "random_state": 42},
    ]
    assert list(results.columns) == [
        "test_year",
        "train_end_year",
        "model",
        "accuracy",
        "macro_f1",
        "confusion_matrix_json",
        "actual_support_json",
        "predicted_support_json",
        "f1_champion",
        "f1_podium",
        "f1_top_5",
        "f1_top_10",
        "f1_midfield",
        "f1_backmarker",
    ]
    assert list(zip(results["train_end_year"], results["test_year"], strict=True)) == [
        (2013, 2014),
        (2014, 2015),
    ]
    assert results["model"].tolist() == ["Random Forest", "Random Forest"]

    assert dict(zip(results.columns, results.dtypes.astype(str), strict=True)) == {
        "test_year": "int64",
        "train_end_year": "int64",
        "model": "str",
        "accuracy": "float64",
        "macro_f1": "float64",
        "confusion_matrix_json": "str",
        "actual_support_json": "str",
        "predicted_support_json": "str",
        "f1_champion": "float64",
        "f1_podium": "float64",
        "f1_top_5": "float64",
        "f1_top_10": "float64",
        "f1_midfield": "float64",
        "f1_backmarker": "float64",
    }
    tier_metric_digest = hashlib.sha256(
        results.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode(
            "utf-8"
        )
    ).hexdigest()
    assert (
        tier_metric_digest
        == "e54d782a42115d9f58c1cb1323fc80a8bb823bbb6145bbc90587f46919bb0109"
    )
    assert set(results["test_year"]) == {2014, 2015}
    assert (results["train_end_year"] < results["test_year"]).all()
    assert {"accuracy", "macro_f1", "f1_champion", "f1_backmarker"}.issubset(
        results.columns
    )


def test_load_models_reports_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "MODEL_DIR", str(tmp_path))

    with pytest.raises(RuntimeError, match="Run `python src/model.py` first"):
        predict.load_models()


def test_load_models_reports_corrupt_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "MODEL_DIR", str(tmp_path))
    (tmp_path / "championship_model.pkl").write_bytes(b"not a pickle")

    with pytest.raises(RuntimeError, match="could not be loaded"):
        predict.load_models()


def test_train_model_can_skip_rolling_evaluation(tmp_path, monkeypatch):
    rows = []
    for year in (2020, 2021, 2022, 2023):
        for driver_id, position in ((1, 1), (2, 2)):
            row = {
                "year": year,
                "driverId": driver_id,
                "champ_position": position,
            }
            row.update({column: float(driver_id) for column in FEATURE_COLUMNS})
            rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path / "features.csv", index=False)
    monkeypatch.setattr(model, "PROC_DIR", str(tmp_path))
    monkeypatch.setattr(model, "MODEL_DIR", str(tmp_path / "models"))
    (tmp_path / "models").mkdir()

    monkeypatch.setattr(
        model,
        "_regression_candidates",
        lambda: {
            "Random Forest + cold-start flags": (
                RandomForestRegressor(n_estimators=1, random_state=42),
                FEATURE_COLUMNS,
            )
        },
    )
    rolling_mock = Mock(
        return_value=pd.DataFrame(
            {
                "test_year": [2023],
                "model": ["fake"],
                "rmse": [1.0],
                "r2": [1.0],
                "spearman": [1.0],
                "spearman_delta_vs_naive": [0.0],
            }
        )
    )
    tier_mock = Mock(
        return_value=pd.DataFrame(
            {
                "test_year": [2023],
                "model": ["fake"],
                "accuracy": [1.0],
                "macro_f1": [1.0],
            }
        )
    )
    monkeypatch.setattr(model, "evaluate_rolling_origin", rolling_mock)
    monkeypatch.setattr(model, "evaluate_tier_rolling_origin", tier_mock)

    model.train_model(forecast_year=2023, skip_rolling_evaluation=True)
    rolling_mock.assert_not_called()
    tier_mock.assert_not_called()

    model.train_model(forecast_year=2023)
    rolling_mock.assert_called_once()
    tier_mock.assert_called_once()
