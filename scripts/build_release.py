"""Build a reproducible prediction report release bundle."""

import argparse
import json
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from check_release import validate_release
from download_data import collect_file_provenance, write_data_manifest
from error_analysis import run_error_analysis
from evaluate import run_evaluation
from model_audit import run_model_audit

from data_processing import PROC_DIR, create_features, load_raw_data
from model import train_model
from predict import predict_championship
from report import create_report
from visualise import create_visualisation

logger = logging.getLogger(__name__)
RESULTS_DIR = BASE_DIR / "results"


def _git_commit() -> str:
    """Return the current Git commit when the build runs inside a checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=BASE_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _is_dirty_worktree() -> bool:
    """Return whether the checkout has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=BASE_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        logger.warning("Unable to inspect Git worktree status; treating it as dirty.")
        return True


def _package_versions() -> dict[str, str]:
    """Collect installed versions for the libraries used by the pipeline."""
    packages = ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib")
    return {package: version(package) for package in packages}


def prediction_path(year: int) -> Path:
    """Return the generated prediction path for a season."""
    return RESULTS_DIR / f"{year}_predictions.csv"


def build_release(
    year: int = 2023,
    download: bool = False,
    allow_dirty: bool = False,
) -> Path:
    """Regenerate all artifacts, validate them, and write a provenance manifest."""
    if download:
        from download_data import download_data

        download_data()
    worktree_dirty = _is_dirty_worktree()
    if worktree_dirty and not allow_dirty:
        raise RuntimeError(
            "Refusing to build a release from a dirty Git worktree; "
            "commit your changes or pass --allow-dirty."
        )

    raw_manifest_path = BASE_DIR / "data" / "raw" / "data_manifest.json"
    if not raw_manifest_path.exists():
        write_data_manifest(
            BASE_DIR / "data" / "raw",
            provenance_note=(
                "Manifest created for the existing local snapshot; original "
                "download timestamp and archive digest were unavailable."
            ),
        )

    logger.info("Building processed features")
    features = create_features(*load_raw_data())
    logger.info("Training saved models")
    train_model(forecast_year=year)
    logger.info("Generating prediction and report artifacts")
    prediction = predict_championship(year)
    if prediction is None:
        raise RuntimeError(f"No prediction was produced for season {year}.")
    summary = run_evaluation(features_path=Path(PROC_DIR) / "features.csv")
    run_model_audit(
        features_path=Path(PROC_DIR) / "features.csv",
        output_dir=RESULTS_DIR,
    )
    run_error_analysis(
        features_path=Path(PROC_DIR) / "features.csv",
        output_dir=RESULTS_DIR,
    )
    chart_path = create_visualisation(year)
    report_path = create_report(year)

    manifest_path = RESULTS_DIR / "release_manifest.json"
    current_git_commit = _git_commit()
    if manifest_path.exists():
        try:
            prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            prior_manifest = {}
        prior_git_commit = prior_manifest.get("git_commit")
        if prior_git_commit and prior_git_commit != current_git_commit:
            logger.info(
                "This build supersedes the prior release manifest from Git commit "
                "%s with current commit %s.",
                prior_git_commit,
                current_git_commit,
            )
    source_manifest = (
        json.loads(raw_manifest_path.read_text(encoding="utf-8"))
        if raw_manifest_path.exists()
        else {}
    )
    manifest = {
        "prediction_year": year,
        "git_commit": current_git_commit,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "packages": _package_versions(),
        "data": {
            "source": "https://www.kaggle.com/datasets/jtrotman/formula-1-race-data",
            "downloaded_at_utc": source_manifest.get("downloaded_at_utc", "unknown"),
            "source_etag": source_manifest.get("source_etag"),
            "source_last_modified": source_manifest.get("source_last_modified"),
            "archive_sha256": source_manifest.get("archive_sha256", "unknown"),
            "raw_files": collect_file_provenance(BASE_DIR / "data" / "raw"),
            "rows": int(len(features)),
            "season_min": int(features["year"].min()),
            "season_max": int(features["year"].max()),
            "season_count": int(features["year"].nunique()),
        },
        "evaluation": summary.to_dict(orient="records"),
        "artifacts": {
            "prediction": str(prediction_path(year).relative_to(BASE_DIR)),
            "chart": str(chart_path.relative_to(BASE_DIR)),
            "report": str(report_path.relative_to(BASE_DIR)),
            "evaluation_summary": "results/rolling_origin_summary.csv",
            "evaluation_details": "results/rolling_origin_summary_details.csv",
            "tier_evaluation_summary": "results/tier_rolling_origin_summary.csv",
            "tier_evaluation_details": "results/tier_rolling_origin_summary_details.csv",
            "tier_class_summary": "results/tier_rolling_origin_class_summary.csv",
            "error_analysis_driver": "results/error_analysis_driver.csv",
            "error_analysis_season": "results/error_analysis_season_summary.csv",
            "error_analysis_group": "results/error_analysis_group_summary.csv",
            "paired_baseline_summary": "results/model_vs_naive_summary.csv",
            "paired_baseline_chart": "results/model_vs_naive_by_season.png",
            "uncertainty_calibration_summary": "results/uncertainty_calibration_summary.csv",
            "uncertainty_calibration_details": "results/uncertainty_calibration_driver.csv",
            "uncertainty_calibration_bins": "results/uncertainty_calibration_bins.csv",
            "permutation_importance_summary": "results/permutation_importance_summary.csv",
            "permutation_importance_details": "results/permutation_importance_details.csv",
        },
    }
    if worktree_dirty:
        manifest["worktree_dirty"] = True
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    errors = validate_release(year=year)
    if errors:
        raise RuntimeError("Release check failed: " + " | ".join(errors))
    logger.info("Release manifest saved to %s", manifest_path)
    return manifest_path


def main() -> int:
    """Parse release-build options and return a shell-friendly status."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--year", type=int, default=2023, help="Prediction season to build"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the raw dataset before building",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow building from a dirty Git worktree and record that state",
    )
    args = parser.parse_args()
    try:
        build_release(
            year=args.year,
            download=args.download,
            allow_dirty=args.allow_dirty,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("Release build failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
