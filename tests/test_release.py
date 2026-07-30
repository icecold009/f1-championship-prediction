import pandas as pd

from scripts.check_release import RAW_FILES, validate_release


def test_validate_release_reports_missing_artifacts(tmp_path):
    errors = validate_release(tmp_path, year=2023)

    assert any("Missing raw data files" in error for error in errors)
    assert any("Missing model artifact" in error for error in errors)
    assert any("Missing release artifact" in error for error in errors)


def test_validate_release_accepts_complete_artifact_layout(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    models_dir = tmp_path / "models"
    results_dir = tmp_path / "results"
    for directory in (raw_dir, processed_dir, models_dir, results_dir):
        directory.mkdir(parents=True)
    for filename in RAW_FILES:
        (raw_dir / filename).touch()

    pd.DataFrame(
        {"year": [2023], "driverId": [1], "champ_position": [1]}
    ).to_csv(processed_dir / "features.csv", index=False)
    (models_dir / "championship_model.pkl").touch()
    (models_dir / "tier_classifier.pkl").touch()
    pd.DataFrame(
        {
            "Predicted Rank": [1],
            "Driver": ["Driver"],
            "Predicted Position": [1.0],
            "Bootstrap Runs": [100],
            "Bootstrap Position SD": [0.2],
            "Bootstrap Position P05": [0.8],
            "Bootstrap Position P95": [1.2],
            "Champion Probability": [0.5],
            "Top 3 Probability": [0.8],
            "Top 5 Probability": [0.9],
        }
    ).to_csv(results_dir / "2023_predictions.csv", index=False)
    for filename in (
        "predicted_vs_actual_2023.png",
        "f1_prediction_report_2023.html",
        "rolling_origin_summary.csv",
        "rolling_origin_summary_details.csv",
        "tier_rolling_origin_summary.csv",
        "tier_rolling_origin_summary_details.csv",
        "tier_rolling_origin_class_summary.csv",
        "error_analysis_driver.csv",
        "error_analysis_season_summary.csv",
        "error_analysis_group_summary.csv",
        "release_manifest.json",
    ):
        (results_dir / filename).touch()

    assert validate_release(tmp_path, year=2023) == []
