import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

NON_START_STATUS_NAMES = {
    "Did not qualify",
    "Did not prequalify",
    "Did not start",
    "DNS",
    "Excluded",
    "Not accepted",
    "Withdrew",
}


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
    races = pd.read_csv(_path("races.csv"))
    results = pd.read_csv(_path("results.csv"))
    drivers = pd.read_csv(_path("drivers.csv"))
    constructors = pd.read_csv(_path("constructors.csv"))
    driver_standings = pd.read_csv(_path("driver_standings.csv"))
    constructor_standings = pd.read_csv(_path("constructor_standings.csv"))
    qualifying = pd.read_csv(_path("qualifying.csv"))
    status = pd.read_csv(_path("status.csv"))
    sprint_results = pd.read_csv(_path("sprint_results.csv"))
    pit_stops = pd.read_csv(_path("pit_stops.csv"))

    return (
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
) -> pd.DataFrame:
    """Build leakage-safe, one-row-per-driver-season forecasting features.

    Predictors describe the driver's prior season and the prior final strength
    of the constructor they enter with. Current-season results are retained
    only to identify entrants and construct the final championship targets.
    """
    del qualifying, pit_stops  # Reserved for future pre-season features.

    # Join race metadata and stable driver/constructor identifiers.
    df = (
        results.merge(races[["raceId", "year", "round"]], on="raceId", how="left")
        .merge(drivers[["driverId", "driverRef"]], on="driverId", how="left")
        .merge(
            constructors[["constructorId", "constructorRef"]],
            on="constructorId",
            how="left",
        )
    )

    finished_ids, non_started_ids = _status_ids(status)

    df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0)
    df["started"] = ~df["statusId"].isin(non_started_ids)
    df["dnf"] = (df["started"] & ~df["statusId"].isin(finished_ids)).astype(int)
    df.loc[~df["started"], ["positionOrder", "grid"]] = pd.NA

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

    out_path = os.path.join(PROC_DIR, "features.csv")
    features.to_csv(out_path, index=False)
    logger.info("Saved %s rows -> %s", len(features), out_path)
    return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = load_raw_data()
    features = create_features(*data)
    logger.info("%s", features.head())
    logger.info("Features shape: %s", features.shape)
    logger.info("Columns: %s", list(features.columns))
