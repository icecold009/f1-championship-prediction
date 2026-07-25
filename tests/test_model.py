import numpy as np
import pandas as pd
import pytest

from src import predict
from src.model import FEATURE_COLUMNS, assign_tier, evaluate_rolling_origin, get_spearman


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
    assert set(results["model"]) == {"Ridge", "Random Forest", "Gradient Boosting"}


def test_load_models_reports_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "MODEL_DIR", str(tmp_path))

    with pytest.raises(RuntimeError, match="Run `python src/model.py` first"):
        predict.load_models()
