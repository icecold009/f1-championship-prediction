import pandas as pd

from src import data_processing


def test_create_features_uses_prior_season_only(tmp_path, monkeypatch):
    races = pd.DataFrame(
        {
            "raceId": [1, 2, 3, 4],
            "year": [2019, 2019, 2020, 2020],
            "round": [1, 2, 1, 2],
        }
    )
    results = pd.DataFrame(
        {
            "raceId": [1, 1, 2, 2, 3, 3, 4, 4],
            "driverId": [10, 20, 10, 20, 10, 20, 10, 20],
            "constructorId": [100, 200, 100, 200, 100, 200, 100, 200],
            "statusId": [1] * 8,
            "positionOrder": [1, 2, 1, 2, 2, 1, 2, 1],
            "grid": [1, 2, 1, 2, 2, 1, 2, 1],
            "points": [25.0, 18.0, 25.0, 18.0, 18.0, 25.0, 18.0, 25.0],
        }
    )
    drivers = pd.DataFrame({"driverId": [10, 20], "driverRef": ["one", "two"]})
    constructors = pd.DataFrame(
        {"constructorId": [100, 200], "constructorRef": ["a", "b"]}
    )
    driver_standings = pd.DataFrame(
        {
            "raceId": [1, 1, 2, 2, 3, 3, 4, 4],
            "driverId": [10, 20, 10, 20, 10, 20, 10, 20],
            "position": [1, 2, 1, 2, 2, 1, 2, 1],
            "points": [25.0, 18.0, 50.0, 36.0, 18.0, 25.0, 36.0, 50.0],
        }
    )
    constructor_standings = pd.DataFrame(
        {
            "raceId": [1, 1, 2, 2, 3, 3, 4, 4],
            "constructorId": [100, 200, 100, 200, 100, 200, 100, 200],
            "position": [1, 2, 1, 2, 1, 2, 1, 2],
            "points": [25.0, 18.0, 50.0, 36.0, 18.0, 25.0, 36.0, 50.0],
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

    current = features[features["year"] == 2020].sort_values("driverId")
    assert len(features) == 4
    assert set(features["driverId"]) == {10, 20}
    assert set(current["champ_position"]) == {1, 2}
    assert current["prev_season_points_sum"].tolist() == [50.0, 36.0]
    assert current["prev_team_final_position"].tolist() == [1, 2]
    assert "points_sum" not in features.columns
    assert "team_final_position" not in features.columns
    assert (tmp_path / "features.csv").exists()
