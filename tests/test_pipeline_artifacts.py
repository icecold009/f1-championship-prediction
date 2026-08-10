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

    def fake_uncertainty(train_df, test_df, **_):
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
    assert (results_dir / "2023_predictions.csv").exists()


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
        pd.DataFrame(columns=sorted(columns)).to_csv(tmp_path / filename, index=False)
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


def test_raw_schema_reports_missing_required_column(tmp_path):
    for filename, columns in REQUIRED_COLUMNS.items():
        selected = sorted(columns)
        if filename == "races.csv":
            selected.remove("year")
        pd.DataFrame(columns=selected).to_csv(tmp_path / filename, index=False)

    with pytest.raises(RuntimeError, match="races.csv.*year"):
        validate_raw_schema(tmp_path)
