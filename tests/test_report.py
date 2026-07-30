import pandas as pd

from src import report


def test_create_report_renders_prediction_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path)
    pd.DataFrame(
        {
            "Predicted Rank": [1, 2],
            "Driver": ["Driver One", "Driver Two"],
            "Team": ["Team A", "Team B"],
            "Predicted Tier": ["Champion", "Podium"],
            "Predicted Position": [1.2, 2.1],
            "Actual Position": [1, 2],
            "Actual Points": [100, 80],
            "Bootstrap Runs": [100, 100],
            "Bootstrap Position Mean": [1.1, 2.2],
            "Bootstrap Position SD": [0.2, 0.3],
            "Bootstrap Position P05": [0.8, 1.7],
            "Bootstrap Position P95": [1.5, 2.8],
            "Champion Probability": [0.5, 0.1],
            "Top 3 Probability": [0.9, 0.6],
            "Top 5 Probability": [1.0, 0.9],
        }
    ).to_csv(tmp_path / "2023_predictions.csv", index=False)
    pd.DataFrame({"test_year": [2022], "mae": [4.1], "rmse": [5.2]}).to_csv(
        tmp_path / "error_analysis_season_summary.csv", index=False
    )
    pd.DataFrame(
        {"dimension": ["Driver type"], "group": ["Rookie"], "mae": [8.2]}
    ).to_csv(tmp_path / "error_analysis_group_summary.csv", index=False)
    pd.DataFrame(
        {
            "model": ["Random Forest + cold-start flags"],
            "mean_spearman_delta_vs_naive": [-0.01],
            "spearman_wins": [4],
            "spearman_losses": [6],
        }
    ).to_csv(tmp_path / "model_vs_naive_summary.csv", index=False)
    pd.DataFrame(
        {
            "nominal_coverage": [0.9],
            "bootstrap_interval_coverage": [0.4],
            "conformal_interval_coverage": [0.91],
        }
    ).to_csv(tmp_path / "uncertainty_calibration_summary.csv", index=False)
    pd.DataFrame(
        {
            "probability_bin": ["0–20%"],
            "observations": [10],
            "observed_top_3_rate": [0.1],
        }
    ).to_csv(tmp_path / "uncertainty_calibration_bins.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["prev_team_final_position"],
            "mean_rmse_increase": [1.2],
        }
    ).to_csv(tmp_path / "permutation_importance_summary.csv", index=False)

    output_path = report.create_report(2023)

    assert output_path.exists()
    contents = output_path.read_text(encoding="utf-8")
    assert "2023 Drivers' Championship" in contents
    assert "Driver One" in contents
    assert "Run with <code>--visualise</code>" in contents
    assert "Bootstrap uncertainty" in contents
    assert "50%" in contents
    assert "Where this model breaks" in contents
    assert "Worst test seasons" in contents
    assert "Paired comparison with the naïve baseline" in contents
    assert "Uncertainty calibration" in contents
    assert "Held-out permutation importance" in contents
