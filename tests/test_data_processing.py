import pandas as pd

from src import data_processing


def test_status_classification_distinguishes_finished_dnf_and_non_start():
    status = pd.DataFrame(
        {
            "statusId": [1, 2, 3, 4, 5],
            "status": [
                "Finished",
                "+1 Lap",
                "Engine",
                "Did not start",
                "Withdrew",
            ],
        }
    )

    finished_ids, non_started_ids = data_processing._status_ids(status)

    assert finished_ids == {1, 2}
    assert non_started_ids == {4, 5}
    assert 3 not in finished_ids | non_started_ids


def test_completed_seasons_exclude_partial_calendar():
    races = pd.DataFrame({"raceId": [1, 2, 3], "year": [2024, 2024, 2025]})
    results = pd.DataFrame({"raceId": [1, 2], "driverId": [10, 10]})

    assert data_processing._completed_season_years(races, results) == {2024}


def test_complete_driver_standings_assigns_missing_zero_point_entrants():
    entries = pd.DataFrame({"year": [2020, 2020], "driverId": [1, 2]})
    known = pd.DataFrame(
        {
            "year": [2020],
            "driverId": [1],
            "champ_position": [1],
            "champ_points": [25.0],
        }
    )
    stats = pd.DataFrame(
        {
            "year": [2020, 2020],
            "driverId": [1, 2],
            "points_sum": [25.0, 0.0],
            "wins": [1, 0],
            "podiums": [1, 0],
            "best_finish": [1, pd.NA],
            "races_started": [1, 0],
        }
    )
    sprint = pd.DataFrame(columns=["year", "driverId", "sprint_points_sum"])

    completed = data_processing._complete_driver_standings(
        entries, known, stats, sprint
    )

    missing = completed.loc[completed["driverId"] == 2].iloc[0]
    assert missing["champ_position"] == 2
    assert missing["champ_points"] == 0


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
    assert current["is_rookie"].tolist() == [0, 0]
    assert current["missing_driver_history"].tolist() == [0, 0]
    assert current["missing_constructor_history"].tolist() == [0, 0]
    assert "points_sum" not in features.columns
    assert "team_final_position" not in features.columns
    assert (tmp_path / "features.csv").exists()
