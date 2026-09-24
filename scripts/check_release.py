"""Validate that a generated report contains the required release artifacts."""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.release_policy import (
    AUDIT_ARTIFACT_KEYS,
    AUDIT_RELEASE_FILES,
    CORE_RELEASE_FILES,
    RAW_FILES,
    RELEASE_SOURCE_PATHS,
    normalise_artifact_path,
    required_release_artifacts,
    validate_artifact_manifest_policy,
    validate_feature_frame,
    validate_manifest_policy,
    validate_prediction_frame,
    validate_raw_data_manifest,
    validate_release_raw_manifest,
)

logger = logging.getLogger(__name__)
__all__ = [
    "AUDIT_ARTIFACT_KEYS",
    "AUDIT_RELEASE_FILES",
    "CORE_RELEASE_FILES",
    "RAW_FILES",
    "RELEASE_SOURCE_PATHS",
    "validate_release",
]


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
    return normalise_artifact_path(value)


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

    require_full_audit, policy_errors = validate_manifest_policy(
        release_manifest, year, require_full_audit
    )
    errors.extend(policy_errors)

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
        actual_raw_files = {
            filename: {
                "sha256": _sha256(raw_dir / filename),
                "bytes": (raw_dir / filename).stat().st_size,
            }
            for filename in RAW_FILES
        }
        try:
            data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
            if not isinstance(data_manifest, dict):
                raise TypeError("manifest root must be an object")
            errors.extend(validate_raw_data_manifest(data_manifest, actual_raw_files))
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            errors.append(f"Invalid raw-data provenance manifest: {exc}")
        errors.extend(validate_release_raw_manifest(release_manifest, actual_raw_files))

    features_path = processed_dir / "features.csv"
    if not features_path.exists():
        errors.append("Missing processed features: data/processed/features.csv")
    else:
        try:
            features = pd.read_csv(features_path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            errors.append(f"Invalid processed features: {exc}")
            features = pd.DataFrame()
        errors.extend(validate_feature_frame(features))

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
        errors.extend(validate_prediction_frame(predictions))

    required_artifacts = required_release_artifacts(year, require_full_audit)
    for filename in required_artifacts:
        if not (results_dir / filename).exists():
            errors.append(f"Missing release artifact: results/{filename}")
        elif (results_dir / filename).stat().st_size == 0:
            errors.append(f"Empty release artifact: results/{filename}")
    errors.extend(
        validate_artifact_manifest_policy(release_manifest, year, require_full_audit)
    )

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
