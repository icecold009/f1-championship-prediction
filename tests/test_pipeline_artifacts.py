import hashlib

import numpy as np
import pandas as pd
import pytest

from scripts.download_data import (
    REQUIRED_COLUMNS,
    collect_file_provenance,
    sha256_file,
    validate_raw_schema,
)
from src import predict, visualise
from src.model import FEATURE_COLUMNS


class _ConstantModel:
    def __init__(self, values):
        self.values = values

    def predict(self, frame):
        assert list(frame.columns) == FEATURE_COLUMNS
        return np.asarray(self.values[: len(frame)])


def test_prediction_pipeline_uses_canonical_features_and_writes_output(
    tmp_path, monkeypatch
):
    processed_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"
    results_dir = tmp_path / "results"
    for directory in (processed_dir, raw_dir, results_dir):
        directory.mkdir()

    rows = []
    for year in (2022, 2023):
        for driver_id, constructor_id, position in ((1, 10, 1), (2, 20, 2)):
            row = {
                "year": year,
                "driverId": driver_id,
                "constructorId": constructor_id,
                "champ_position": position,
                "champ_points": 100 - position,
            }
            row.update({column: float(driver_id) for column in FEATURE_COLUMNS})
            rows.append(row)
    pd.DataFrame(rows).to_csv(processed_dir / "features.csv", index=False)
    pd.DataFrame(
        {
            "driverId": [1, 2],
            "forename": ["Driver", "Driver"],
            "surname": ["One", "Two"],
        }
    ).to_csv(raw_dir / "drivers.csv", index=False)
    pd.DataFrame(
        {
            "constructorId": [10, 20],
            "name": ["Team A", "Team B"],
        }
    ).to_csv(raw_dir / "constructors.csv", index=False)

    monkeypatch.setattr(predict, "PROC_DIR", str(processed_dir))
    monkeypatch.setattr(predict, "RAW_DIR", str(raw_dir))
    monkeypatch.setattr(predict, "RESULTS_DIR", str(results_dir))
    monkeypatch.setattr(
        predict,
        "load_models",
        lambda: (_ConstantModel([1.2, 2.4]), _ConstantModel(["Champion", "Podium"])),
    )

    uncertainty_calls = []

    def fake_uncertainty(train_df, test_df, **kwargs):
        uncertainty_calls.append(kwargs.copy())
        assert kwargs["random_state"] == 42
        assert train_df["year"].max() < test_df["year"].min()
        return pd.DataFrame(
            {
                "bootstrap_runs": [3, 3],
                "bootstrap_position_mean": [1.3, 2.3],
                "bootstrap_position_sd": [0.2, 0.3],
                "bootstrap_position_p05": [1.0, 2.0],
                "bootstrap_position_p95": [1.6, 2.8],
                "champion_probability": [0.8, 0.2],
                "top_3_probability": [1.0, 1.0],
                "top_5_probability": [1.0, 1.0],
            }
        )

    monkeypatch.setattr(predict, "bootstrap_position_predictions", fake_uncertainty)
    output = predict.predict_championship(2023)

    assert output is not None
    assert output["Driver"].tolist() == ["Driver One", "Driver Two"]
    assert list(output.columns) == [
        "Predicted Rank",
        "Driver",
        "Team",
        "Predicted Tier",
        "Predicted Position",
        "Actual Position",
        "Actual Points",
        "Bootstrap Runs",
        "Bootstrap Position Mean",
        "Bootstrap Position SD",
        "Bootstrap Position P05",
        "Bootstrap Position P95",
        "Champion Probability",
        "Top 3 Probability",
        "Top 5 Probability",
    ]
    assert dict(zip(output.columns, output.dtypes.astype(str), strict=True)) == {
        "Predicted Rank": "int64",
        "Driver": "str",
        "Team": "str",
        "Predicted Tier": "str",
        "Predicted Position": "float64",
        "Actual Position": "int64",
        "Actual Points": "int64",
        "Bootstrap Runs": "int64",
        "Bootstrap Position Mean": "float64",
        "Bootstrap Position SD": "float64",
        "Bootstrap Position P05": "float64",
        "Bootstrap Position P95": "float64",
        "Champion Probability": "float64",
        "Top 3 Probability": "float64",
        "Top 5 Probability": "float64",
    }
    assert output["Predicted Rank"].tolist() == [1, 2]
    for column in (
        "Champion Probability",
        "Top 3 Probability",
        "Top 5 Probability",
    ):
        assert output[column].between(0, 1).all()

    prediction_path = results_dir / "2023_predictions.csv"
    first_csv = prediction_path.read_bytes()
    first_digest = hashlib.sha256(first_csv.replace(b"\r\n", b"\n")).hexdigest()
    assert (
        first_digest
        == "40bff07b4856ade3d78fa9ffe8366073562f5fa960425b26cd12a701cf52915b"
    )
    repeated = predict.predict_championship(2023)
    pd.testing.assert_frame_equal(output, repeated)
    assert prediction_path.read_bytes() == first_csv
    assert len(uncertainty_calls) == 2
    assert all(call["random_state"] == 42 for call in uncertainty_calls)


def test_visualisation_writes_reviewer_chart(tmp_path, monkeypatch):
    monkeypatch.setattr(visualise, "RESULTS_DIR", tmp_path)
    pd.DataFrame(
        {
            "Driver": ["Driver One", "Driver Two"],
            "Actual Position": [1, 2],
            "Predicted Rank": [1, 2],
        }
    ).to_csv(tmp_path / "2023_predictions.csv", index=False)

    output = visualise.create_visualisation(2023)

    assert output.exists()
    assert output.stat().st_size > 0


def test_visualisation_reports_missing_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(visualise, "RESULTS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="Prediction file not found"):
        visualise.create_visualisation(2023)


def test_raw_schema_and_hash_provenance(tmp_path):
    for filename, columns in REQUIRED_COLUMNS.items():
        pd.DataFrame(columns=sorted(columns)).to_csv(
            tmp_path / filename, index=False, lineterminator="\n"
        )
    # Tables not consumed by feature engineering still belong in provenance.
    from scripts.download_data import REQUIRED_FILES

    for filename in REQUIRED_FILES:
        path = tmp_path / filename
        if not path.exists():
            path.write_text("placeholder\n", encoding="utf-8")

    validate_raw_schema(tmp_path)
    provenance = collect_file_provenance(tmp_path)

    assert set(provenance) == set(REQUIRED_FILES)
    sample = tmp_path / "races.csv"
    assert sha256_file(sample) == hashlib.sha256(sample.read_bytes()).hexdigest()
    assert set(provenance["races.csv"]) == {"sha256", "bytes"}
    assert provenance["races.csv"] == {
        "sha256": "4fd307c59dbfde92dff1a913f8826391558d1f4608e0d8c8664c204ef0c91bc3",
        "bytes": 18,
    }


def test_raw_schema_reports_missing_required_column(tmp_path):
    for filename, columns in REQUIRED_COLUMNS.items():
        selected = sorted(columns)
        if filename == "races.csv":
            selected.remove("year")
        pd.DataFrame(columns=selected).to_csv(tmp_path / filename, index=False)

    with pytest.raises(RuntimeError, match="races.csv.*year"):
        validate_raw_schema(tmp_path)
