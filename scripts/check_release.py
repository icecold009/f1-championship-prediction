"""Validate that a generated report contains the required release artifacts."""

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
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
logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(base_dir: Path) -> str | None:
    """Return the current commit, or None when validation runs outside Git."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=base_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _normalise_artifact_path(value: object) -> str:
    return str(value).replace("\\", "/")


def _source_changed_since(base_dir: Path, commit: str) -> bool:
    """Return whether pipeline source changed after a manifest commit."""
    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                f"{commit}..HEAD",
                "--",
                *RELEASE_SOURCE_PATHS,
            ],
            cwd=base_dir,
            capture_output=True,
            check=False,
        )
        return result.returncode != 0
    except OSError:
        return True


def validate_release(
    base_dir: Path = BASE_DIR,
    year: int = 2023,
    reject_dirty_manifest: bool = False,
    require_full_audit: bool | None = None,
) -> list[str]:
    """Return actionable errors for a release directory, or an empty list."""
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    errors: list[str] = []

    release_manifest_path = results_dir / "release_manifest.json"
    release_manifest: dict = {}
    if not release_manifest_path.exists():
        errors.append("Missing release manifest: results/release_manifest.json")
    else:
        try:
            release_manifest = json.loads(
                release_manifest_path.read_text(encoding="utf-8")
            )
            if not isinstance(release_manifest, dict):
                raise TypeError("manifest root must be an object")
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            errors.append(f"Invalid release manifest: {exc}")

    manifest_audit = release_manifest.get("full_audit")
    if require_full_audit is None:
        require_full_audit = manifest_audit is not False
    if require_full_audit and manifest_audit is not True:
        errors.append("Release manifest does not record a completed full audit")
    if release_manifest and release_manifest.get("prediction_year") != year:
        errors.append("Release manifest prediction year does not match requested year")

    current_commit = _git_commit(base_dir)
    manifest_commit = release_manifest.get("git_commit")
    if (
        current_commit
        and manifest_commit
        and manifest_commit != "unknown"
        and manifest_commit != current_commit
        and _source_changed_since(base_dir, str(manifest_commit))
    ):
        errors.append(
            f"Release manifest commit {manifest_commit} does not match current source "
            f"at HEAD {current_commit}"
        )

    missing_raw = [name for name in RAW_FILES if not (raw_dir / name).exists()]
    if missing_raw:
        errors.append(f"Missing raw data files: {', '.join(missing_raw)}")
    data_manifest_path = raw_dir / "data_manifest.json"
    if not data_manifest_path.exists():
        errors.append(
            "Missing raw-data provenance manifest: data/raw/data_manifest.json"
        )
    elif not missing_raw:
        try:
            data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
            if not isinstance(data_manifest, dict):
                raise TypeError("manifest root must be an object")
            recorded_files = data_manifest.get("files", {})
            if not isinstance(recorded_files, dict):
                raise TypeError("files must be an object")
            for filename in RAW_FILES:
                recorded_entry = recorded_files.get(filename, {})
                if not isinstance(recorded_entry, dict):
                    recorded_entry = {}
                recorded_hash = recorded_entry.get("sha256")
                if not recorded_hash:
                    errors.append(f"Raw-data manifest missing hash for {filename}")
                elif recorded_hash != _sha256(raw_dir / filename):
                    errors.append(f"Raw-data checksum mismatch: {filename}")
                recorded_bytes = recorded_entry.get("bytes")
                if (
                    recorded_bytes is not None
                    and recorded_bytes != (raw_dir / filename).stat().st_size
                ):
                    errors.append(f"Raw-data byte-size mismatch: {filename}")
            archive_sha256 = data_manifest.get("archive_sha256")
            if archive_sha256 in (None, "unknown"):
                errors.append("Raw-data manifest lacks an immutable archive SHA-256")
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            errors.append(f"Invalid raw-data provenance manifest: {exc}")

        release_data = release_manifest.get("data", {})
        release_raw_files = (
            release_data.get("raw_files", {}) if isinstance(release_data, dict) else {}
        )
        if not isinstance(release_raw_files, dict):
            errors.append("Release manifest raw_files must be an object")
        else:
            for filename in RAW_FILES:
                expected = release_raw_files.get(filename, {})
                if not isinstance(expected, dict):
                    expected = {}
                if expected.get("sha256") != _sha256(raw_dir / filename):
                    errors.append(f"Release manifest checksum mismatch: {filename}")
                if expected.get("bytes") != (raw_dir / filename).stat().st_size:
                    errors.append(f"Release manifest byte-size mismatch: {filename}")

    features_path = processed_dir / "features.csv"
    if not features_path.exists():
        errors.append("Missing processed features: data/processed/features.csv")
    else:
        try:
            features = pd.read_csv(features_path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            errors.append(f"Invalid processed features: {exc}")
            features = pd.DataFrame()
        required_features = {"year", "driverId", "champ_position"}
        missing_features = sorted(required_features - set(features.columns))
        if missing_features:
            errors.append(
                f"Processed features missing columns: {', '.join(missing_features)}"
            )
        if features.empty:
            errors.append("Processed features are empty")
        elif "champ_position" in features.columns:
            missing_target_years = set(
                pd.to_numeric(
                    features.loc[features["champ_position"].isna(), "year"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
            )
            latest_year = int(pd.to_numeric(features["year"], errors="coerce").max())
            if missing_target_years - {latest_year}:
                errors.append(
                    "Processed features have missing championship targets before "
                    "the latest season"
                )
            if features.duplicated(["year", "driverId"]).any():
                errors.append("Processed features contain duplicate driver-season rows")

    for filename in ("championship_model.pkl", "tier_classifier.pkl"):
        model_path = models_dir / filename
        if not model_path.exists():
            errors.append(f"Missing model artifact: models/{filename}")
        elif model_path.stat().st_size == 0:
            errors.append(f"Empty model artifact: models/{filename}")

    prediction_path = results_dir / f"{year}_predictions.csv"
    if not prediction_path.exists():
        errors.append(f"Missing prediction artifact: results/{year}_predictions.csv")
    else:
        try:
            predictions = pd.read_csv(prediction_path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            errors.append(f"Invalid prediction artifact: {exc}")
            predictions = pd.DataFrame()
        required_prediction_columns = {
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
        missing_prediction_columns = sorted(
            required_prediction_columns - set(predictions.columns)
        )
        if missing_prediction_columns:
            errors.append(
                "Prediction artifact missing columns: "
                + ", ".join(missing_prediction_columns)
            )
        if predictions.empty:
            errors.append("Prediction artifact is empty")
        elif not missing_prediction_columns:
            ranks = predictions["Predicted Rank"]
            numeric_ranks = pd.to_numeric(ranks, errors="coerce")
            expected_ranks = list(range(1, len(predictions) + 1))
            if (
                numeric_ranks.isna().any()
                or not numeric_ranks.eq(numeric_ranks.astype("Int64")).all()
                or not numeric_ranks.is_unique
                or numeric_ranks.astype(int).tolist() != expected_ranks
            ):
                errors.append("Prediction ranks are not a complete ordered sequence")
            for column in (
                "Champion Probability",
                "Top 3 Probability",
                "Top 5 Probability",
            ):
                values = pd.to_numeric(predictions[column], errors="coerce")
                if values.isna().any() or not values.between(0, 1).all():
                    errors.append(f"Prediction probabilities are invalid: {column}")

    required_artifacts = [name.format(year=year) for name in CORE_RELEASE_FILES]
    if require_full_audit:
        required_artifacts.extend(AUDIT_RELEASE_FILES)
    artifact_manifest = release_manifest.get("artifacts", {})
    if not isinstance(artifact_manifest, dict):
        errors.append("Release manifest artifacts must be an object")
        artifact_manifest = {}
    for filename in required_artifacts:
        if not (results_dir / filename).exists():
            errors.append(f"Missing release artifact: results/{filename}")
        elif (results_dir / filename).stat().st_size == 0:
            errors.append(f"Empty release artifact: results/{filename}")

    expected_artifacts = {
        "prediction": f"results/{year}_predictions.csv",
        "chart": f"results/predicted_vs_actual_{year}.png",
        "report": f"results/f1_prediction_report_{year}.html",
    }
    for key, expected in expected_artifacts.items():
        actual = _normalise_artifact_path(artifact_manifest.get(key))
        if actual != expected:
            errors.append(f"Release manifest artifact path mismatch: {key}")

    if require_full_audit:
        for filename in AUDIT_RELEASE_FILES:
            key = AUDIT_ARTIFACT_KEYS[filename]
            if (
                _normalise_artifact_path(artifact_manifest.get(key))
                != f"results/{filename}"
            ):
                errors.append(f"Release manifest is missing audit artifact: {filename}")

    if reject_dirty_manifest and release_manifest_path.exists():
        if release_manifest.get("worktree_dirty") is True:
            errors.append("Release manifest was generated from a dirty Git worktree")

    return errors


def main() -> int:
    """Validate the default local release and return a shell-friendly status."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--year", type=int, default=2023, help="Prediction season to validate"
    )
    parser.add_argument(
        "--reject-dirty-manifest",
        action="store_true",
        help="Fail if the release manifest records a dirty Git worktree",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Validate a quick release that intentionally omits the full audit",
    )
    args = parser.parse_args()

    errors = validate_release(
        year=args.year,
        reject_dirty_manifest=args.reject_dirty_manifest,
        require_full_audit=not args.quick,
    )
    if errors:
        logger.error("Release check failed:")
        for error in errors:
            logger.error("- %s", error)
        return 1

    logger.info("Release check passed for %s.", args.year)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
