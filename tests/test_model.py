import numpy as np
import pandas as pd
import pytest

from src import predict
from src.model import (
    FEATURE_COLUMNS,
    NAIVE_BASELINE_NAME,
    assign_tier,
    evaluate_rolling_origin,
    evaluate_tier_rolling_origin,
    get_spearman,
    previous_season_final_order,
    bootstrap_position_predictions,
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

    uncertainty = bootstrap_position_predictions(
        frame[frame["year"] < 2014],
        frame[frame["year"] == 2014],
        n_bootstrap=4,
        n_estimators=3,
    )

    assert len(uncertainty) == 2
    assert (uncertainty["bootstrap_runs"] == 4).all()
    assert uncertainty["champion_probability"].sum() == pytest.approx(1.0)
    assert uncertainty["top_3_probability"].sum() == pytest.approx(2.0)


def test_rolling_origin_keeps_test_seasons_after_training_cutoff():
    rows = []
    for year in range(2010, 2016):
        for driver_id, position in [(1, 1), (2, 2)]:
            row = {
                "year": year,
                "driverId": driver_id,
                "champ_position": position,
            }
            row.update({column: float(driver_id) + year / 1000 for column in FEATURE_COLUMNS})
            rows.append(row)

    results = evaluate_rolling_origin(
        pd.DataFrame(rows), test_seasons=2, min_train_seasons=3
    )

    assert set(results["test_year"]) == {2014, 2015}
    assert (results["train_end_year"] < results["test_year"]).all()
    assert set(results["model"]) == {
        "Ridge",
        "Random Forest",
        "Gradient Boosting",
        "Baseline: previous avg finish",
        NAIVE_BASELINE_NAME,
    }


def test_tier_rolling_origin_keeps_test_seasons_after_training_cutoff():
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

    results = evaluate_tier_rolling_origin(
        pd.DataFrame(rows), test_seasons=2, min_train_seasons=3
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
