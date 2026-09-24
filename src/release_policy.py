"""Pure policy checks for release manifests and generated table artifacts."""

from typing import Any

import pandas as pd

RAW_FILES = (
    "circuits.csv",
    "constructor_results.csv",
    "constructor_standings.csv",
    "constructors.csv",
    "driver_standings.csv",
    "drivers.csv",
    "lap_times.csv",
    "pit_stops.csv",
    "qualifying.csv",
    "races.csv",
    "results.csv",
    "seasons.csv",
    "sprint_results.csv",
    "status.csv",
)
CORE_RELEASE_FILES = (
    "predicted_vs_actual_{year}.png",
    "f1_prediction_report_{year}.html",
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
    "release_manifest.json",
)
AUDIT_RELEASE_FILES = (
    "uncertainty_calibration_driver.csv",
    "uncertainty_calibration_summary.csv",
    "uncertainty_calibration_bins.csv",
    "permutation_importance_details.csv",
    "permutation_importance_summary.csv",
)
AUDIT_ARTIFACT_KEYS = {
    "uncertainty_calibration_driver.csv": "uncertainty_calibration_details",
    "uncertainty_calibration_summary.csv": "uncertainty_calibration_summary",
    "uncertainty_calibration_bins.csv": "uncertainty_calibration_bins",
    "permutation_importance_details.csv": "permutation_importance_details",
    "permutation_importance_summary.csv": "permutation_importance_summary",
}
RELEASE_SOURCE_PATHS = (
    ".github",
    "main.py",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "scripts",
    "src",
)
REQUIRED_FEATURE_COLUMNS = {"year", "driverId", "champ_position"}
REQUIRED_PREDICTION_COLUMNS = {
    "Predicted Rank",
    "Driver",
    "Predicted Position",
    "Bootstrap Runs",
    "Bootstrap Position SD",
    "Bootstrap Position P05",
    "Bootstrap Position P95",
    "Champion Probability",
    "Top 3 Probability",
    "Top 5 Probability",
}
PREDICTION_PROBABILITY_COLUMNS = (
    "Champion Probability",
    "Top 3 Probability",
    "Top 5 Probability",
)


def normalise_artifact_path(value: object) -> str:
    """Return the platform-independent path spelling used in manifests."""
    return str(value).replace("\\", "/")


def validate_manifest_policy(
    release_manifest: dict[str, Any],
    year: int,
    require_full_audit: bool | None,
) -> tuple[bool, list[str]]:
    """Resolve quick/full mode and validate manifest identity fields."""
    manifest_audit = release_manifest.get("full_audit")
    if require_full_audit is None:
        require_full_audit = manifest_audit is not False
    errors = []
    if require_full_audit and manifest_audit is not True:
        errors.append("Release manifest does not record a completed full audit")
    if release_manifest and release_manifest.get("prediction_year") != year:
        errors.append("Release manifest prediction year does not match requested year")
    return require_full_audit, errors


def validate_raw_data_manifest(
    data_manifest: dict[str, Any],
    actual_files: dict[str, dict[str, Any]],
    raw_files: tuple[str, ...] = RAW_FILES,
) -> list[str]:
    """Compare raw-table provenance values with current file evidence."""
    errors = []
    recorded_files = data_manifest.get("files", {})
    if not isinstance(recorded_files, dict):
        return ["Invalid raw-data provenance manifest: files must be an object"]
    for filename in raw_files:
        recorded_entry = recorded_files.get(filename, {})
        if not isinstance(recorded_entry, dict):
            recorded_entry = {}
        recorded_hash = recorded_entry.get("sha256")
        if not recorded_hash:
            errors.append(f"Raw-data manifest missing hash for {filename}")
        elif recorded_hash != actual_files[filename]["sha256"]:
            errors.append(f"Raw-data checksum mismatch: {filename}")
        recorded_bytes = recorded_entry.get("bytes")
        if (
            recorded_bytes is not None
            and recorded_bytes != actual_files[filename]["bytes"]
        ):
            errors.append(f"Raw-data byte-size mismatch: {filename}")
    archive_sha256 = data_manifest.get("archive_sha256")
    if archive_sha256 in (None, "unknown"):
        errors.append("Raw-data manifest lacks an immutable archive SHA-256")
    return errors


def validate_release_raw_manifest(
    release_manifest: dict[str, Any],
    actual_files: dict[str, dict[str, Any]],
    raw_files: tuple[str, ...] = RAW_FILES,
) -> list[str]:
    """Compare release-manifest raw hashes and sizes with current file evidence."""
    release_data = release_manifest.get("data", {})
    release_raw_files = (
        release_data.get("raw_files", {}) if isinstance(release_data, dict) else {}
    )
    if not isinstance(release_raw_files, dict):
        return ["Release manifest raw_files must be an object"]
    errors = []
    for filename in raw_files:
        expected = release_raw_files.get(filename, {})
        if not isinstance(expected, dict):
            expected = {}
        if expected.get("sha256") != actual_files[filename]["sha256"]:
            errors.append(f"Release manifest checksum mismatch: {filename}")
        if expected.get("bytes") != actual_files[filename]["bytes"]:
            errors.append(f"Release manifest byte-size mismatch: {filename}")
    return errors


def validate_feature_frame(features: pd.DataFrame) -> list[str]:
    """Validate required feature columns, target completeness, and unique keys."""
    errors = []
    missing_features = sorted(REQUIRED_FEATURE_COLUMNS - set(features.columns))
    if missing_features:
        errors.append(
            f"Processed features missing columns: {', '.join(missing_features)}"
        )
    if features.empty:
        errors.append("Processed features are empty")
    elif "champ_position" in features.columns:
        missing_target_years = (
            set(
                pd.to_numeric(
                    features.loc[features["champ_position"].isna(), "year"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
            )
            if "year" in features.columns
            else set()
        )
        years = (
            pd.to_numeric(features["year"], errors="coerce")
            if "year" in features.columns
            else pd.Series(dtype=float)
        )
        latest_year = years.max()
        if pd.notna(latest_year) and missing_target_years - {int(latest_year)}:
            errors.append(
                "Processed features have missing championship targets before "
                "the latest season"
            )
        if {"year", "driverId"}.issubset(features.columns) and features.duplicated(
            ["year", "driverId"]
        ).any():
            errors.append("Processed features contain duplicate driver-season rows")
    return errors


def validate_prediction_frame(predictions: pd.DataFrame) -> list[str]:
    """Validate prediction columns, ordered ranks, and probability bounds."""
    errors = []
    missing_columns = sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
    if missing_columns:
        errors.append(
            "Prediction artifact missing columns: " + ", ".join(missing_columns)
        )
    if predictions.empty:
        errors.append("Prediction artifact is empty")
    elif not missing_columns:
        numeric_ranks = pd.to_numeric(predictions["Predicted Rank"], errors="coerce")
        expected_ranks = list(range(1, len(predictions) + 1))
        if (
            numeric_ranks.isna().any()
            or not numeric_ranks.eq(numeric_ranks.astype("Int64")).all()
            or not numeric_ranks.is_unique
            or numeric_ranks.astype(int).tolist() != expected_ranks
        ):
            errors.append("Prediction ranks are not a complete ordered sequence")
        for column in PREDICTION_PROBABILITY_COLUMNS:
            values = pd.to_numeric(predictions[column], errors="coerce")
            if values.isna().any() or not values.between(0, 1).all():
                errors.append(f"Prediction probabilities are invalid: {column}")
    return errors


def required_release_artifacts(year: int, require_full_audit: bool) -> list[str]:
    """Return the artifact filenames required for the selected release mode."""
    required = [name.format(year=year) for name in CORE_RELEASE_FILES]
    if require_full_audit:
        required.extend(AUDIT_RELEASE_FILES)
    return required


def validate_artifact_manifest_policy(
    release_manifest: dict[str, Any], year: int, require_full_audit: bool
) -> list[str]:
    """Validate manifest artifact paths and full-audit artifact declarations."""
    artifact_manifest = release_manifest.get("artifacts", {})
    if not isinstance(artifact_manifest, dict):
        artifact_manifest = {}
        errors = ["Release manifest artifacts must be an object"]
    else:
        errors = []
    expected_artifacts = {
        "prediction": f"results/{year}_predictions.csv",
        "chart": f"results/predicted_vs_actual_{year}.png",
        "report": f"results/f1_prediction_report_{year}.html",
    }
    for key, expected in expected_artifacts.items():
        if normalise_artifact_path(artifact_manifest.get(key)) != expected:
            errors.append(f"Release manifest artifact path mismatch: {key}")
    if require_full_audit:
        for filename in AUDIT_RELEASE_FILES:
            key = AUDIT_ARTIFACT_KEYS[filename]
            if (
                normalise_artifact_path(artifact_manifest.get(key))
                != f"results/{filename}"
            ):
                errors.append(f"Release manifest is missing audit artifact: {filename}")
    return errors
