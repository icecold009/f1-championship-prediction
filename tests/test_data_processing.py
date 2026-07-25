import pandas as pd

from src import data_processing


def test_create_features_builds_driver_season_rows(tmp_path, monkeypatch):
    races = pd.DataFrame(
        {
            "raceId": [1, 2],
            "year": [2020, 2020],
            "round": [1, 2],
        }
    )
    results = pd.DataFrame(
        {
            "raceId": [1, 1, 2, 2],
            "driverId": [10, 20, 10, 20],
            "constructorId": [100, 200, 100, 200],
            "statusId": [1, 1, 1, 1],
            "positionOrder": [1, 2, 1, 2],
            "grid": [1, 2, 1, 2],
            "points": [25.0, 18.0, 25.0, 18.0],
        }
    )
    drivers = pd.DataFrame({"driverId": [10, 20], "driverRef": ["one", "two"]})
    constructors = pd.DataFrame({"constructorId": [100, 200], "constructorRef": ["a", "b"]})
    driver_standings = pd.DataFrame(
        {
            "raceId": [1, 1, 2, 2],
            "driverId": [10, 20, 10, 20],
            "position": [1, 2, 1, 2],
            "points": [25.0, 18.0, 50.0, 36.0],
        }
    )
    constructor_standings = pd.DataFrame(
        {
            "raceId": [1, 1, 2, 2],
            "constructorId": [100, 200, 100, 200],
            "position": [1, 2, 1, 2],
            "points": [25.0, 18.0, 50.0, 36.0],
        }
    )
    status = pd.DataFrame({"statusId": [1], "status": ["Finished"]})
    empty_qualifying = pd.DataFrame()
    empty_sprints = pd.DataFrame(columns=["raceId", "driverId", "points"])
    empty_pit_stops = pd.DataFrame()
    monkeypatch.setattr(data_processing, "PROC_DIR", str(tmp_path))

    features = data_processing.create_features(
        races,
        results,
        drivers,
        constructors,
        driver_standings,
        constructor_standings,
        empty_qualifying,
        status,
        empty_sprints,
        empty_pit_stops,
    )

    assert len(features) == 2
    assert set(features["driverId"]) == {10, 20}
    assert set(features["champ_position"]) == {1, 2}
    assert dict(zip(features["driverId"], features["points_sum"])) == {10: 50.0, 20: 36.0}
    assert (tmp_path / "features.csv").exists()
