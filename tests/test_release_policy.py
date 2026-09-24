import pandas as pd

from src.release_policy import (
    validate_artifact_manifest_policy,
    validate_feature_frame,
    validate_manifest_policy,
    validate_prediction_frame,
)


def test_manifest_policy_resolves_quick_and_full_audit_modes():
    quick, quick_errors = validate_manifest_policy(
        {"full_audit": False, "prediction_year": 2023}, 2023, None
    )
    full, full_errors = validate_manifest_policy(
        {"full_audit": False, "prediction_year": 2023}, 2023, True
    )

    assert quick is False
    assert quick_errors == []
    assert full is True
    assert full_errors == ["Release manifest does not record a completed full audit"]


def test_feature_and_prediction_policies_keep_release_thresholds():
    features = pd.DataFrame(
        {
            "year": [2022, 2022, 2023],
            "driverId": [1, 2, 1],
            "champ_position": [1.0, 2.0, None],
        }
    )
    predictions = pd.DataFrame(
        {
            "Predicted Rank": [1, 2],
            "Driver": ["One", "Two"],
            "Predicted Position": [1.0, 2.0],
            "Bootstrap Runs": [100, 100],
            "Bootstrap Position SD": [0.2, 0.3],
            "Bootstrap Position P05": [0.8, 1.7],
            "Bootstrap Position P95": [1.2, 2.3],
            "Champion Probability": [0.7, 0.4],
            "Top 3 Probability": [0.9, 0.8],
            "Top 5 Probability": [1.1, 0.9],
        }
    )

    assert validate_feature_frame(features) == []
    assert validate_prediction_frame(predictions) == [
        "Prediction probabilities are invalid: Top 5 Probability"
    ]


def test_artifact_manifest_policy_checks_expected_paths_and_audit_set():
    errors = validate_artifact_manifest_policy(
        {
            "artifacts": {
                "prediction": "results\\2023_predictions.csv",
                "chart": "results/predicted_vs_actual_2023.png",
                "report": "results/f1_prediction_report_2023.html",
                "uncertainty_calibration_details": (
                    "results/uncertainty_calibration_driver.csv"
                ),
            }
        },
        2023,
        True,
    )

    assert errors == [
        "Release manifest is missing audit artifact: uncertainty_calibration_summary.csv",
        "Release manifest is missing audit artifact: uncertainty_calibration_bins.csv",
        "Release manifest is missing audit artifact: permutation_importance_details.csv",
        "Release manifest is missing audit artifact: permutation_importance_summary.csv",
    ]
