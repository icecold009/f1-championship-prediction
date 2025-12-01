import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

def train_model():
    # Load processed features
    df = pd.read_csv('processed data/features.csv')

    # Drop rows without final position (target)
    df = df.dropna(subset=['position_final'])

    feature_columns = [
        'position_mean',
        'position_std',
        'position_count',
        'points_sum',
        'grid_mean',
        'prev_season_points',
        'prev_season_avg_position'
    ]

    X = df[feature_columns].fillna(0)
    y = df['position_final']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.2f}")
    print(f"R2: {r2:.2f}")

    os.makedirs('models', exist_ok=True)
    with open('models/championship_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

if __name__ == "__main__":
    train_model()