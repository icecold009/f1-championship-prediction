import pandas as pd

from scripts.error_analysis import annotate_driver_context, generate_error_rows
from src.model import FEATURE_COLUMNS


def _synthetic_features() -> pd.DataFrame:
    rows = []
    for year in range(2010, 2016):
        for driver_id, position in [(1, 1), (2, 2)]:
            row = {
                "year": year,
                "driverId": driver_id,
                "champ_position": position,
                "prev_season_races_started": (
                    None if year == 2010 and driver_id == 1 else 10
                ),
            }
            row.update({column: float(driver_id) for column in FEATURE_COLUMNS})
            row["prev_season_races_started"] = (
                None if year == 2010 and driver_id == 1 else 10
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_error_analysis_labels_driver_types_and_constructor_swaps():
    races = pd.DataFrame(
        {
            "raceId": [1, 2, 3, 4, 5, 6, 7, 8],
            "year": [2010, 2011, 2012, 2013, 2014, 2015, 2015, 2015],
        }
    )
    results = pd.DataFrame(
        {
            "raceId": [1, 2, 3, 4, 5, 6, 7, 8],
            "driverId": [1, 2, 1, 2, 1, 1, 2, 1],
            "constructorId": [10, 20, 10, 20, 10, 11, 20, 10],
        }
    )

    annotated = annotate_driver_context(_synthetic_features(), results, races)

    rookie = annotated[(annotated["year"] == 2010) & (annotated["driverId"] == 1)]
    swap = annotated[(annotated["year"] == 2015) & (annotated["driverId"] == 1)]
    assert rookie.iloc[0]["driver_type"] == "Rookie"
    assert bool(swap.iloc[0]["mid_season_swap"])

    errors = generate_error_rows(
        annotated, test_seasons=2, min_train_seasons=3, n_estimators=3
    )
    assert (errors["train_end_year"] < errors["test_year"]).all()
