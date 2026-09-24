import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

NON_START_STATUS_NAMES = {
    "Did not qualify",
    "Did not prequalify",
    "Did not start",
    "DNS",
    "Excluded",
    "Not accepted",
    "Withdrew",
}


@dataclass(frozen=True)
class RawTables:
    races: pd.DataFrame
    results: pd.DataFrame
    drivers: pd.DataFrame
    constructors: pd.DataFrame
    driver_standings: pd.DataFrame
    constructor_standings: pd.DataFrame
    qualifying: pd.DataFrame
    status: pd.DataFrame
    sprint_results: pd.DataFrame
    pit_stops: pd.DataFrame

    def as_legacy_tuple(self) -> tuple[pd.DataFrame, ...]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__)


@dataclass(frozen=True)
class ValidatedTables:
    tables: RawTables


@dataclass(frozen=True)
class CleanedRaceResults:
    frame: pd.DataFrame
    finished_status_ids: set[int]
    non_started_status_ids: set[int]


@dataclass(frozen=True)
class FeatureResult:
    frame: pd.DataFrame


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    label: str


_TABLE_REQUIREMENTS = {
    "races": ("raceId", "year", "round"),
    "results": (
        "raceId",
        "driverId",
        "constructorId",
        "statusId",
        "positionOrder",
        "grid",
        "points",
    ),
    "drivers": ("driverId", "driverRef"),
    "constructors": ("constructorId", "constructorRef"),
    "driver_standings": ("raceId", "driverId", "position", "points"),
    "constructor_standings": ("raceId", "constructorId", "position", "points"),
    "status": ("statusId", "status"),
    "sprint_results": ("raceId", "driverId", "points"),
}
_TABLE_KEYS = {
    "races": ("raceId",),
    "results": ("raceId", "driverId"),
    "drivers": ("driverId",),
    "constructors": ("constructorId",),
    "driver_standings": ("raceId", "driverId"),
    "constructor_standings": ("raceId", "constructorId"),
    "status": ("statusId",),
    "sprint_results": ("raceId", "driverId"),
}
_TABLE_REQUIRED_VALUES = {
    "races": ("year", "round"),
    "results": ("constructorId", "statusId"),
    "drivers": ("driverRef",),
    "constructors": ("constructorRef",),
    "driver_standings": ("position", "points"),
    "constructor_standings": ("position", "points"),
    "status": ("status",),
}


def load_raw_tables(raw_dir: str | Path = RAW_DIR) -> RawTables:
    """Load the ten source tables as a named, typed boundary value."""
    directory = Path(raw_dir)
    return RawTables(
        races=pd.read_csv(directory / "races.csv"),
        results=pd.read_csv(directory / "results.csv"),
        drivers=pd.read_csv(directory / "drivers.csv"),
        constructors=pd.read_csv(directory / "constructors.csv"),
        driver_standings=pd.read_csv(directory / "driver_standings.csv"),
        constructor_standings=pd.read_csv(directory / "constructor_standings.csv"),
        qualifying=pd.read_csv(directory / "qualifying.csv"),
        status=pd.read_csv(directory / "status.csv"),
        sprint_results=pd.read_csv(directory / "sprint_results.csv"),
        pit_stops=pd.read_csv(directory / "pit_stops.csv"),
    )


def validate_raw_tables(tables: RawTables) -> ValidatedTables:
    """Validate the required source shape and deterministic row-order contracts."""
    for name, columns in _TABLE_REQUIREMENTS.items():
        frame = getattr(tables, name)
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(
                f"{name} is missing required columns: {', '.join(missing)}"
            )
        keys = _TABLE_KEYS[name]
        if frame[list(keys)].isna().any(axis=None):
            raise ValueError(f"{name} has missing key values: {', '.join(keys)}")
        if frame.duplicated(list(keys)).any():
            raise ValueError(f"{name} has duplicate keys: {', '.join(keys)}")
        required_values = _TABLE_REQUIRED_VALUES.get(name, ())
        missing_values = [
            column for column in required_values if frame[column].isna().any()
        ]
        if missing_values:
            raise ValueError(
                f"{name} has missing required values: {', '.join(missing_values)}"
            )

    for name, order in {
        "races": ("year", "round", "raceId"),
        "results": ("raceId", "driverId"),
    }.items():
        frame = getattr(tables, name)
        if not frame.reset_index(drop=True).equals(
            frame.sort_values(list(order), kind="stable").reset_index(drop=True)
        ):
            raise ValueError(f"{name} rows must be ordered by {', '.join(order)}")

    for name, frame in tables.__dict__.items():
        for column in frame.select_dtypes(include="number"):
            if frame[column].isin([float("inf"), float("-inf")]).any():
                raise ValueError(f"{name}.{column} contains non-finite values")
    return ValidatedTables(tables)


def clean_race_results(tables: ValidatedTables) -> CleanedRaceResults:
    """Join source identifiers and normalize result fields for feature work."""
    raw = tables.tables
    frame = (
        raw.results.merge(
            raw.races[["raceId", "year", "round"]], on="raceId", how="left"
        )
        .merge(raw.drivers[["driverId", "driverRef"]], on="driverId", how="left")
        .merge(
            raw.constructors[["constructorId", "constructorRef"]],
            on="constructorId",
            how="left",
        )
    )
    finished_ids, non_started_ids = _status_ids(raw.status)
    frame["positionOrder"] = pd.to_numeric(frame["positionOrder"], errors="coerce")
    frame["grid"] = pd.to_numeric(frame["grid"], errors="coerce")
    frame["points"] = pd.to_numeric(frame["points"], errors="coerce").fillna(0)
    frame["started"] = ~frame["statusId"].isin(non_started_ids)
    frame["dnf"] = (frame["started"] & ~frame["statusId"].isin(finished_ids)).astype(
        int
    )
    frame.loc[~frame["started"], ["positionOrder", "grid"]] = pd.NA
    return CleanedRaceResults(frame, finished_ids, non_started_ids)


def split_features(
    features: pd.DataFrame, forecast_year: int | None = None
) -> TemporalSplit:
    """Apply the model's existing chronological holdout rule without model imports."""
    labeled = features.dropna(subset=["champ_position"])
    if forecast_year is None:
        test_years = sorted(labeled["year"].unique())[-5:]
        split_year = test_years[0] if test_years else None
        train = (
            labeled[labeled["year"] < split_year]
            if split_year is not None
            else labeled.iloc[0:0]
        )
        test = (
            labeled[labeled["year"] >= split_year]
            if split_year is not None
            else labeled.iloc[0:0]
        )
        label = f"latest five-season holdout beginning {split_year}"
    else:
        train = labeled[labeled["year"] < forecast_year]
        test = labeled[labeled["year"] == forecast_year]
        label = f"forecast year {forecast_year}"
    return TemporalSplit(train=train, test=test, label=label)


def _path(filename: str) -> str:
    return os.path.join(RAW_DIR, filename)


def _status_ids(status: pd.DataFrame) -> tuple[set[int], set[int]]:
    """Return status IDs for classified finishes and non-starts."""
    labels = status["status"].astype("string").fillna("")
    finished = labels.eq("Finished") | labels.str.fullmatch(r"\+\d+\s+Laps?")
    non_started = labels.isin(NON_START_STATUS_NAMES)
    return (
        set(status.loc[finished, "statusId"].astype(int)),
        set(status.loc[non_started, "statusId"].astype(int)),
    )


def _completed_season_years(races: pd.DataFrame, results: pd.DataFrame) -> set[int]:
    """Return seasons for which every scheduled race has result rows."""
    scheduled = races.groupby("year")["raceId"].agg(set)
    observed = (
        results.merge(races[["raceId", "year"]], on="raceId", how="inner")
        .groupby("year")["raceId"]
        .agg(set)
    )
    return {
        int(year)
        for year, race_ids in scheduled.items()
        if race_ids.issubset(observed.get(year, set()))
    }


def _complete_driver_standings(
    season_entries: pd.DataFrame,
    final_driver_standings: pd.DataFrame,
    driver_stats: pd.DataFrame,
    sprint_agg: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct positions for entrants absent from source standings.

    Ergast-compatible ``driver_standings`` omits some zero-point entrants.
    Preserve official positions where available and assign the remaining
    positions after the known standings using deterministic points/tiebreaker
    fields from the season results.
    """
    known = final_driver_standings.copy()
    entries = season_entries[["year", "driverId"]].drop_duplicates()
    missing = entries.merge(
        known[["year", "driverId"]], on=["year", "driverId"], how="left", indicator=True
    )
    missing = missing.loc[missing["_merge"] == "left_only", ["year", "driverId"]]
    if missing.empty:
        return known

    summary = driver_stats[
        [
            "year",
            "driverId",
            "points_sum",
            "wins",
            "podiums",
            "best_finish",
            "races_started",
        ]
    ].merge(sprint_agg, on=["year", "driverId"], how="left")
    summary["sprint_points_sum"] = summary["sprint_points_sum"].fillna(0)
    summary["total_points"] = summary["points_sum"] + summary["sprint_points_sum"]
    missing = missing.merge(
        summary[
            [
                "year",
                "driverId",
                "total_points",
                "wins",
                "podiums",
                "best_finish",
                "races_started",
            ]
        ],
        on=["year", "driverId"],
        how="left",
    )
    missing["champ_points"] = missing["total_points"].fillna(0)
    missing["champ_position"] = 0

    assignments = []
    for year, group in missing.groupby("year", sort=True):
        known_positions = known.loc[known["year"] == year, "champ_position"]
        next_position = (
            int(pd.to_numeric(known_positions).max()) + 1
            if not known_positions.empty
            else 1
        )
        ordered = group.sort_values(
            [
                "total_points",
                "wins",
                "podiums",
                "best_finish",
                "races_started",
                "driverId",
            ],
            ascending=[False, False, False, True, False, True],
            na_position="last",
        ).copy()
        ordered["champ_position"] = range(next_position, next_position + len(ordered))
        assignments.append(
            ordered[["year", "driverId", "champ_position", "champ_points"]]
        )

    return pd.concat([known, *assignments], ignore_index=True)


def load_raw_data() -> tuple[pd.DataFrame, ...]:
    """Load the raw Ergast-compatible CSV tables required by feature engineering."""
    return load_raw_tables().as_legacy_tuple()


def build_feature_result(
    validated: ValidatedTables,
    cleaned: CleanedRaceResults,
) -> FeatureResult:
    """Build a typed feature result from the validated and cleaned stages.

    Predictors describe the driver's prior season and the prior final strength
    of the constructor they enter with. Current-season results are retained
    only to identify entrants and construct the final championship targets.
    """
    raw = validated.tables
    races = raw.races
    results = raw.results
    driver_standings = raw.driver_standings
    constructor_standings = raw.constructor_standings
    sprint_results = raw.sprint_results
    df = cleaned.frame

    # Build historical driver-season statistics. These are shifted forward by
    # one year below, so no same-season race outcome becomes a predictor.
    driver_stats = (
        df.groupby(["year", "driverId", "driverRef"])
        .agg(
            races_started=("started", "sum"),
            avg_finish_pos=("positionOrder", "mean"),
            best_finish=("positionOrder", "min"),
            std_finish_pos=("positionOrder", "std"),
            points_sum=("points", "sum"),
            avg_grid_pos=("grid", "mean"),
            wins=("positionOrder", lambda values: (values == 1).sum()),
            podiums=("positionOrder", lambda values: (values <= 3).sum()),
            points_finishes=("positionOrder", lambda values: (values <= 10).sum()),
            dnf_count=("dnf", "sum"),
        )
        .reset_index()
    )

    started_counts = (
        driver_stats["races_started"]
        .astype(float)
        .where(driver_stats["races_started"].gt(0))
    )
    driver_stats["win_rate"] = driver_stats["wins"].div(started_counts)
    driver_stats["podium_rate"] = driver_stats["podiums"].div(started_counts)
    driver_stats["dnf_rate"] = driver_stats["dnf_count"].div(started_counts)
    driver_stats["points_per_race"] = driver_stats["points_sum"].div(started_counts)
    driver_stats["quali_to_race_delta"] = (
        driver_stats["avg_grid_pos"] - driver_stats["avg_finish_pos"]
    )

    sprint_results = sprint_results.copy()
    sprint_results["sprintPoints"] = pd.to_numeric(
        sprint_results["points"], errors="coerce"
    ).fillna(0)
    sprint_agg = (
        sprint_results.merge(races[["raceId", "year"]], on="raceId", how="left")
        .groupby(["year", "driverId"])["sprintPoints"]
        .sum()
        .reset_index()
        .rename(columns={"sprintPoints": "sprint_points_sum"})
    )
    driver_stats = driver_stats.merge(sprint_agg, on=["year", "driverId"], how="left")
    driver_stats["sprint_points_sum"] = driver_stats["sprint_points_sum"].fillna(0)

    # Take the last constructor standings entry per constructor and season.
    constructor_rows = constructor_standings.merge(
        races[["raceId", "year"]], on="raceId", how="left"
    )
    constructor_final = (
        constructor_rows.sort_values(["year", "raceId"])
        .groupby(["year", "constructorId"])
        .tail(1)[["year", "constructorId", "points", "position"]]
        .rename(
            columns={
                "points": "team_final_points",
                "position": "team_final_position",
            }
        )
    )

    # The first observed constructor is the season-opening team available at
    # forecast time; later same-season team changes must not affect features.
    season_entries = df.sort_values(["year", "round", "raceId"]).drop_duplicates(
        ["year", "driverId"]
    )[["year", "driverId", "constructorId", "driverRef", "constructorRef"]]

    history_columns = [
        "races_started",
        "avg_finish_pos",
        "std_finish_pos",
        "points_sum",
        "avg_grid_pos",
        "win_rate",
        "podium_rate",
        "dnf_rate",
        "points_per_race",
        "quali_to_race_delta",
        "sprint_points_sum",
    ]
    prior_driver = driver_stats.copy()
    prior_driver["year"] += 1
    prior_driver = prior_driver[["year", "driverId", *history_columns]].rename(
        columns={column: f"prev_season_{column}" for column in history_columns}
    )
    features = season_entries.merge(prior_driver, on=["year", "driverId"], how="left")

    # Cold-start context is known before a season begins. Keeping explicit
    # indicators prevents "no prior history" from being represented only by
    # zero-imputed performance values.
    first_entry_year = features.groupby("driverId")["year"].transform("min")
    missing_driver_history = features["prev_season_races_started"].isna()
    features["is_rookie"] = features["year"].eq(first_entry_year).astype(int)
    features["returning_after_gap"] = (
        missing_driver_history & features["is_rookie"].eq(0)
    ).astype(int)
    features["missing_driver_history"] = missing_driver_history.astype(int)

    prior_constructor = constructor_final.copy()
    prior_constructor["year"] += 1
    prior_constructor = prior_constructor.rename(
        columns={
            "team_final_points": "prev_team_final_points",
            "team_final_position": "prev_team_final_position",
        }
    )
    features = features.merge(
        prior_constructor[
            [
                "year",
                "constructorId",
                "prev_team_final_points",
                "prev_team_final_position",
            ]
        ],
        on=["year", "constructorId"],
        how="left",
    )
    features["missing_constructor_history"] = (
        features["prev_team_final_position"].isna().astype(int)
    )

    # Targets are the final driver standings for the current season.
    standings_rows = driver_standings.merge(
        races[["raceId", "year"]], on="raceId", how="left"
    )
    final_driver_standings = (
        standings_rows.sort_values(["year", "raceId"])
        .groupby(["year", "driverId"])
        .tail(1)[["year", "driverId", "position", "points"]]
        .rename(
            columns={
                "position": "champ_position",
                "points": "champ_points",
            }
        )
    )
    completed_years = _completed_season_years(races, results)
    final_driver_standings = final_driver_standings.loc[
        final_driver_standings["year"].isin(completed_years)
    ]
    final_driver_standings = _complete_driver_standings(
        season_entries.loc[season_entries["year"].isin(completed_years)],
        final_driver_standings,
        driver_stats,
        sprint_agg,
    )
    features = features.merge(
        final_driver_standings, on=["year", "driverId"], how="left"
    )

    return FeatureResult(features)


def create_features(
    races: pd.DataFrame,
    results: pd.DataFrame,
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    driver_standings: pd.DataFrame,
    constructor_standings: pd.DataFrame,
    qualifying: pd.DataFrame,
    status: pd.DataFrame,
    sprint_results: pd.DataFrame,
    pit_stops: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compatibility orchestrator: build features and write the legacy CSV."""
    raw = RawTables(
        races,
        results,
        drivers,
        constructors,
        driver_standings,
        constructor_standings,
        qualifying,
        status,
        sprint_results,
        pit_stops,
    )
    validated = validate_raw_tables(raw)
    cleaned = clean_race_results(validated)
    result = build_feature_result(validated, cleaned)
    destination = (
        Path(output_path)
        if output_path is not None
        else Path(PROC_DIR) / "features.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.frame.to_csv(destination, index=False)
    logger.info("Saved %s rows -> %s", len(result.frame), destination)
    return result.frame


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = load_raw_data()
    features = create_features(*data)
    logger.info("%s", features.head())
    logger.info("Features shape: %s", features.shape)
    logger.info("Columns: %s", list(features.columns))
