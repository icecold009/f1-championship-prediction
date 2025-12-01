import os
print("CWD:", os.getcwd())
print("Items in CWD:", os.listdir())

# races = pd.read_csv('raw data/races.csv')
# results = pd.read_csv('raw data/results.csv')
# ...


import pandas as pd

def load_and_merge_data():
    races = pd.read_csv('raw data/races.csv')
    results = pd.read_csv('raw data/results.csv')
    drivers = pd.read_csv('raw data/drivers.csv')
    constructors = pd.read_csv('raw data/constructors.csv')
    driver_standings = pd.read_csv('raw data/driver_standings.csv')

    df = results.merge(races, on='raceId', how='left')
    df = df.merge(drivers, on='driverId', how='left')
    df = df.merge(constructors, on='constructorId', how='left')

    return df, driver_standings

def create_features(df, driver_standings):
    features = df.groupby(['year', 'driverId', 'constructorId']).agg(
        position_mean=('position', 'mean'),
        position_std=('position', 'std'),
        position_count=('position', 'count'),
        points_sum=('points', 'sum'),
        grid_mean=('grid', 'mean')
    ).reset_index()

    features = features.sort_values(['driverId', 'year'])

    features['prev_season_points'] = features.groupby('driverId')['points_sum'].shift(1)
    features['prev_season_avg_position'] = features.groupby('driverId')['position_mean'].shift(1)

    races = df[['raceId', 'year']].drop_duplicates()
    ds = driver_standings.merge(races, on='raceId', how='left')
    final_ds = ds.sort_values('raceId').groupby(['year', 'driverId']).tail(1)

    final_ds = final_ds[['year', 'driverId', 'position', 'points']]
    final_ds = final_ds.rename(columns={
        'position': 'position_final',
        'points': 'points_final'
    })

    features = features.merge(
        final_ds,
        on=['year', 'driverId'],
        how='left'
    )

    features.to_csv('processed data/features.csv', index=False)
    return features

if __name__ == "__main__":
    df, driver_standings = load_and_merge_data()
    create_features(df, driver_standings)
    print("Data processing complete. Saved to processed data/features.csv")