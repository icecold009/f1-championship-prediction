import pandas as pd
import pickle
import os

def predict_championship(year):
    # Load model
    with open('models/championship_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load features
    df = pd.read_csv('processed data/features.csv')

    # Filter to chosen season
    season_df = df[df['year'] == year].copy()

    feature_columns = [
        'position_mean',
        'position_std',
        'position_count',
        'points_sum',
        'grid_mean',
        'prev_season_points',
        'prev_season_avg_position'
    ]

    X = season_df[feature_columns].fillna(0)

    preds = model.predict(X)

    season_df['predicted_position'] = preds

    # Keep only needed columns for output
    output = season_df[['year', 'driverId', 'constructorId',
                        'position_final', 'predicted_position']].copy()

    # Sort by predicted championship position (lower is better)
    output = output.sort_values('predicted_position')

    os.makedirs('results', exist_ok=True)
    output.to_csv('results/predictions.csv', index=False)

    print("Saved predictions to results/predictions.csv")
    print(output.head())

if __name__ == "__main__":
    # Change the year below to the season you want to examine
    predict_championship(2020)