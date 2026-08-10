"""Download the public Ergast-compatible F1 CSV dataset used by the pipeline."""

import argparse
import csv
import hashlib
import json
import logging
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "raw"
DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/jtrotman/formula-1-race-data"
)
REQUIRED_FILES = (
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
REQUIRED_COLUMNS = {
    "constructor_standings.csv": {"raceId", "constructorId", "points", "position"},
    "constructors.csv": {"constructorId", "constructorRef"},
    "driver_standings.csv": {"raceId", "driverId", "points", "position"},
    "drivers.csv": {"driverId", "driverRef", "forename", "surname"},
    "races.csv": {"raceId", "year", "round"},
    "results.csv": {
        "raceId",
        "driverId",
        "constructorId",
        "positionOrder",
        "grid",
        "points",
        "statusId",
    },
    "sprint_results.csv": {"raceId", "driverId", "points"},
    "status.csv": {"statusId", "status"},
}
MIN_ROW_COUNTS = {
    "circuits.csv": 50,
    "constructor_results.csv": 10_000,
    "constructor_standings.csv": 10_000,
    "constructors.csv": 100,
    "driver_standings.csv": 30_000,
    "drivers.csv": 500,
    "lap_times.csv": 500_000,
    "pit_stops.csv": 10_000,
    "qualifying.csv": 10_000,
    "races.csv": 1_000,
    "results.csv": 20_000,
    "seasons.csv": 70,
    "sprint_results.csv": 300,
    "status.csv": 100,
}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for an artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw_schema(output_dir: Path) -> None:
    """Fail fast when a downloaded table no longer matches the expected schema."""
    for filename, expected in REQUIRED_COLUMNS.items():
        path = output_dir / filename
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            try:
                columns = set(next(csv.reader(source)))
            except StopIteration as exc:
                raise RuntimeError(f"{filename} is empty") from exc
        missing = sorted(expected - columns)
        if missing:
            raise RuntimeError(
                f"{filename} is missing required columns: {', '.join(missing)}"
            )


def validate_raw_content(
    output_dir: Path,
    minimum_rows: dict[str, int] | None = None,
) -> None:
    """Reject empty or materially truncated tables before publishing a snapshot."""
    expected_rows = minimum_rows or MIN_ROW_COUNTS
    failures = []
    for filename in REQUIRED_FILES:
        path = output_dir / filename
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            next(reader, None)
            row_count = sum(1 for _ in reader)
        required_rows = expected_rows.get(filename, 1)
        if row_count < required_rows:
            failures.append(
                f"{filename} has {row_count} data rows; expected at least {required_rows}"
            )
    if failures:
        raise RuntimeError("Raw data content validation failed: " + "; ".join(failures))


def collect_file_provenance(output_dir: Path) -> dict[str, dict[str, int | str]]:
    """Collect hashes and sizes for every required raw table."""
    return {
        filename: {
            "sha256": sha256_file(output_dir / filename),
            "bytes": (output_dir / filename).stat().st_size,
        }
        for filename in REQUIRED_FILES
    }


def write_data_manifest(
    output_dir: Path,
    *,
    downloaded_at_utc: str = "unknown",
    source_etag: str | None = None,
    source_last_modified: str | None = None,
    archive_sha256: str = "unknown",
    provenance_note: str = "Downloaded by scripts/download_data.py",
) -> Path:
    """Write immutable identifiers for an existing raw-data snapshot."""
    validate_raw_schema(output_dir)
    validate_raw_content(output_dir)
    manifest = {
        "source_url": DATASET_URL,
        "downloaded_at_utc": downloaded_at_utc,
        "manifest_created_at_utc": datetime.now(UTC).isoformat(),
        "source_etag": source_etag,
        "source_last_modified": source_last_modified,
        "archive_sha256": archive_sha256,
        "provenance_note": provenance_note,
        "files": collect_file_provenance(output_dir),
    }
    manifest_path = output_dir / "data_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def download_data(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Download and extract the expected raw CSV tables into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    request = Request(
        DATASET_URL, headers={"User-Agent": "f1-championship-prediction/1.0"}
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "formula-1-race-data.zip"
        staging_dir = Path(temp_dir) / "staged"
        staging_dir.mkdir()
        logger.info("Downloading dataset from %s", DATASET_URL)
        with (
            urlopen(request, timeout=120) as response,
            archive_path.open("wb") as archive_file,
        ):
            source_etag = response.headers.get("ETag")
            source_last_modified = response.headers.get("Last-Modified")
            shutil.copyfileobj(response, archive_file)

        with ZipFile(archive_path) as archive:
            members = {Path(name).name: name for name in archive.namelist()}
            missing = [name for name in REQUIRED_FILES if name not in members]
            if missing:
                raise RuntimeError(
                    f"Dataset is missing expected CSV files: {', '.join(missing)}"
                )

            for filename in REQUIRED_FILES:
                target = staging_dir / filename
                with (
                    archive.open(members[filename]) as source,
                    target.open("wb") as destination,
                ):
                    shutil.copyfileobj(source, destination)

        validate_raw_schema(staging_dir)
        validate_raw_content(staging_dir)
        for filename in REQUIRED_FILES:
            (staging_dir / filename).replace(output_dir / filename)

        write_data_manifest(
            output_dir,
            downloaded_at_utc=datetime.now(UTC).isoformat(),
            source_etag=source_etag,
            source_last_modified=source_last_modified,
            archive_sha256=sha256_file(archive_path),
        )

    logger.info("Downloaded %s CSV files to %s", len(REQUIRED_FILES), output_dir)


def main() -> None:
    """Parse command-line options and download the raw dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for downloaded CSV files (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download_data(args.output_dir)


if __name__ == "__main__":
    main()
