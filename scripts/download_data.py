"""Download the public Ergast-compatible F1 CSV dataset used by the pipeline."""

import argparse
import logging
import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "raw"
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/jtrotman/formula-1-race-data"
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


def download_data(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Download and extract the expected raw CSV tables into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    request = Request(DATASET_URL, headers={"User-Agent": "f1-championship-prediction/1.0"})

    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "formula-1-race-data.zip"
        logger.info("Downloading dataset from %s", DATASET_URL)
        with urlopen(request, timeout=120) as response, archive_path.open("wb") as archive_file:
            shutil.copyfileobj(response, archive_file)

        with ZipFile(archive_path) as archive:
            members = {Path(name).name: name for name in archive.namelist()}
            missing = [name for name in REQUIRED_FILES if name not in members]
            if missing:
                raise RuntimeError(f"Dataset is missing expected CSV files: {', '.join(missing)}")

            for filename in REQUIRED_FILES:
                target = output_dir / filename
                with archive.open(members[filename]) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

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
