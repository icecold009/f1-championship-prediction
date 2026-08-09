import hashlib
import json
import logging
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.build_release as build_release
from scripts.check_release import RAW_FILES, validate_release


def _stub_build_release(monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    features = pd.DataFrame(
        {"year": [2022, 2023], "driverId": [1, 2], "champ_position": [1, 2]}
    )
    monkeypatch.setattr(build_release, "BASE_DIR", tmp_path)
    monkeypatch.setattr(build_release, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(
        build_release, "write_data_manifest", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(build_release, "load_raw_data", lambda: ())
    monkeypatch.setattr(build_release, "create_features", lambda *_args: features)
    monkeypatch.setattr(build_release, "train_model", lambda **_kwargs: None)
    monkeypatch.setattr(build_release, "predict_championship", lambda _year: object())
    monkeypatch.setattr(
        build_release, "run_evaluation", lambda **_kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(build_release, "run_model_audit", lambda **_kwargs: None)
    monkeypatch.setattr(build_release, "run_error_analysis", lambda **_kwargs: None)
    monkeypatch.setattr(
        build_release,
        "create_visualisation",
        lambda year: results_dir / f"predicted_vs_actual_{year}.png",
    )
    monkeypatch.setattr(
        build_release,
        "create_report",
        lambda year: results_dir / f"f1_prediction_report_{year}.html",
    )
    monkeypatch.setattr(build_release, "collect_file_provenance", lambda _path: {})
    monkeypatch.setattr(build_release, "_package_versions", lambda: {})
    monkeypatch.setattr(build_release, "validate_release", lambda **_kwargs: [])
    return results_dir


def test_is_dirty_worktree_reports_git_status(monkeypatch):
    calls = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return SimpleNamespace(stdout=" M scripts/build_release.py\n")

    monkeypatch.setattr(build_release.subprocess, "run", fake_run)

    assert build_release._is_dirty_worktree() is True
    assert calls["command"] == ["git", "status", "--porcelain"]
    assert calls["kwargs"]["cwd"] == build_release.BASE_DIR


def test_build_release_blocks_dirty_worktree(tmp_path, monkeypatch):
    _stub_build_release(monkeypatch, tmp_path)
    monkeypatch.setattr(build_release, "_is_dirty_worktree", lambda: True)

    with pytest.raises(RuntimeError, match="commit your changes or pass --allow-dirty"):
        build_release.build_release()

    assert not (tmp_path / "results" / "release_manifest.json").exists()


def test_build_release_allow_dirty_records_manifest_state(tmp_path, monkeypatch):
    _stub_build_release(monkeypatch, tmp_path)
    monkeypatch.setattr(build_release, "_is_dirty_worktree", lambda: True)
    monkeypatch.setattr(build_release, "_git_commit", lambda: "current-commit")

    manifest_path = build_release.build_release(allow_dirty=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["git_commit"] == "current-commit"
    assert manifest["worktree_dirty"] is True


def test_build_release_logs_prior_manifest_supersession(tmp_path, monkeypatch, caplog):
    results_dir = _stub_build_release(monkeypatch, tmp_path)
    results_dir.mkdir(parents=True)
    (results_dir / "release_manifest.json").write_text(
        json.dumps({"git_commit": "prior-commit"}), encoding="utf-8"
    )
    monkeypatch.setattr(build_release, "_is_dirty_worktree", lambda: False)
    monkeypatch.setattr(build_release, "_git_commit", lambda: "current-commit")

    with caplog.at_level(logging.INFO):
        build_release.build_release()

    assert "supersedes the prior release manifest" in caplog.text


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
    files = {
        filename: {
            "sha256": hashlib.sha256(b"").hexdigest(),
            "bytes": 0,
        }
        for filename in RAW_FILES
    }
    (raw_dir / "data_manifest.json").write_text(
        json.dumps({"files": files}),
        encoding="utf-8",
    )

    pd.DataFrame({"year": [2023], "driverId": [1], "champ_position": [1]}).to_csv(
        processed_dir / "features.csv", index=False
    )
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
        "model_vs_naive_summary.csv",
        "model_vs_naive_by_season.png",
        "uncertainty_calibration_driver.csv",
        "uncertainty_calibration_summary.csv",
        "uncertainty_calibration_bins.csv",
        "permutation_importance_details.csv",
        "permutation_importance_summary.csv",
        "release_manifest.json",
    ):
        (results_dir / filename).touch()

    assert validate_release(tmp_path, year=2023) == []

    (results_dir / "release_manifest.json").write_text(
        json.dumps({"worktree_dirty": True}), encoding="utf-8"
    )
    assert validate_release(tmp_path, year=2023) == []
    errors = validate_release(tmp_path, year=2023, reject_dirty_manifest=True)
    assert any("dirty Git worktree" in error for error in errors)

    (raw_dir / RAW_FILES[0]).write_text("changed", encoding="utf-8")
    errors = validate_release(tmp_path, year=2023)
    assert any("Raw-data checksum mismatch" in error for error in errors)
