import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

def _path(filename: str) -> str:
    return os.path.join(RAW_DIR, filename)

# ── Load ───────────────────────────────────────────────────────────────────────
def load_raw_data() -> tuple[pd.DataFrame, ...]:
    """Load the raw Ergast-compatible CSV tables required by feature engineering."""
    races             = pd.read_csv(_path("races.csv"))
    results           = pd.read_csv(_path("results.csv"))
    drivers           = pd.read_csv(_path("drivers.csv"))
    constructors      = pd.read_csv(_path("constructors.csv"))
    driver_standings  = pd.read_csv(_path("driver_standings.csv"))
    constructor_standings = pd.read_csv(_path("constructor_standings.csv"))
    qualifying        = pd.read_csv(_path("qualifying.csv"))
    status            = pd.read_csv(_path("status.csv"))
    sprint_results    = pd.read_csv(_path("sprint_results.csv"))
    pit_stops         = pd.read_csv(_path("pit_stops.csv"))

    return (races, results, drivers, constructors,
            driver_standings, constructor_standings,
            qualifying, status, sprint_results, pit_stops)

# ── Feature Engineering ────────────────────────────────────────────────────────
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
    """Aggregate raw race tables into one driver-season feature DataFrame."""

    # ── Base merge ──────────────────────────────────────────────────────────
    df = (results
          .merge(races[["raceId", "year", "round"]], on="raceId", how="left")
          .merge(drivers[["driverId", "driverRef"]], on="driverId", how="left")
          .merge(constructors[["constructorId", "constructorRef"]], on="constructorId", how="left"))

    # ── DNF flag using status.csv ────────────────────────────────────────────
    # statusId == 1 means "Finished"; everything else is a non-finish
    finished_ids = status.loc[status["status"] == "Finished", "statusId"].values
    df["dnf"] = (~df["statusId"].isin(finished_ids)).astype(int)

    # ── Position: replace "\N" (Ergast null) with NaN, then cast ────────────
    df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
    df["grid"]          = pd.to_numeric(df["grid"],          errors="coerce")
    df["points"]        = pd.to_numeric(df["points"],        errors="coerce").fillna(0)

    # ── Per-driver, per-season aggregates ───────────────────────────────────
    agg = df.groupby(["year", "driverId", "constructorId", "driverRef", "constructorRef"]).agg(
        races_started       = ("raceId",        "count"),
        avg_finish_pos      = ("positionOrder", "mean"),
        std_finish_pos      = ("positionOrder", "std"),
        points_sum          = ("points",        "sum"),
        avg_grid_pos        = ("grid",          "mean"),
        wins                = ("positionOrder", lambda x: (x == 1).sum()),
        podiums             = ("positionOrder", lambda x: (x <= 3).sum()),
        points_finishes     = ("positionOrder", lambda x: (x <= 10).sum()),
        dnf_count           = ("dnf",           "sum"),
    ).reset_index()

    agg["win_rate"]        = agg["wins"]    / agg["races_started"]
    agg["podium_rate"]     = agg["podiums"] / agg["races_started"]
    agg["dnf_rate"]        = agg["dnf_count"] / agg["races_started"]
    agg["points_per_race"] = agg["points_sum"]  / agg["races_started"]

    # ── Qualifying delta (avg grid vs avg finish — positive = gained places) ─
    agg["quali_to_race_delta"] = agg["avg_grid_pos"] - agg["avg_finish_pos"]

    # ── Sprint points ────────────────────────────────────────────────────────
    sprint_results["sprintPoints"] = pd.to_numeric(
        sprint_results["points"], errors="coerce").fillna(0)
    sprint_agg = (sprint_results
                  .merge(races[["raceId", "year"]], on="raceId", how="left")
                  .groupby(["year", "driverId"])["sprintPoints"]
                  .sum()
                  .reset_index()
                  .rename(columns={"sprintPoints": "sprint_points_sum"}))
    agg = agg.merge(sprint_agg, on=["year", "driverId"], how="left")
    agg["sprint_points_sum"] = agg["sprint_points_sum"].fillna(0)

    # ── Constructor season strength (from constructor_standings) ─────────────
    cs_merged = constructor_standings.merge(races[["raceId", "year"]], on="raceId", how="left")
    # Take the last standings entry per constructor per season
    cs_final = (cs_merged.sort_values("raceId")
                          .groupby(["year", "constructorId"])
                          .tail(1)[["year", "constructorId", "points", "position"]]
                          .rename(columns={"points":   "team_final_points",
                                           "position": "team_final_position"}))
    agg = agg.merge(cs_final, on=["year", "constructorId"], how="left")

    # ── Previous-season features ─────────────────────────────────────────────
    agg = agg.sort_values(["driverId", "year"])
    agg["prev_season_points"]      = agg.groupby("driverId")["points_sum"].shift(1)
    agg["prev_season_avg_pos"]     = agg.groupby("driverId")["avg_finish_pos"].shift(1)
    agg["prev_season_win_rate"]    = agg.groupby("driverId")["win_rate"].shift(1)

    # ── Target variable: driver's FINAL championship position & points ───────
    ds_merged = driver_standings.merge(races[["raceId", "year"]], on="raceId", how="left")
    final_ds  = (ds_merged.sort_values("raceId")
                           .groupby(["year", "driverId"])
                           .tail(1)[["year", "driverId", "position", "points"]]
                           .rename(columns={"position": "champ_position",
                                            "points":   "champ_points"}))
    agg = agg.merge(final_ds, on=["year", "driverId"], how="left")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(PROC_DIR, "features.csv")
    agg.to_csv(out_path, index=False)
    logger.info("Saved %s rows → %s", len(agg), out_path)
    return agg

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = load_raw_data()
    features = create_features(*data)
    logger.info("%s", features.head())
    logger.info("\nFeatures shape: %s", features.shape)
    logger.info("Columns: %s", list(features.columns))
