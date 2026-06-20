import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(BASE_DIR, "Data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "avg_finish_pos",
    "std_finish_pos",
    "races_started",
    "points_sum",
    "avg_grid_pos",
    "win_rate",
    "podium_rate",
    "dnf_rate",
    "points_per_race",
    "quali_to_race_delta",
    "sprint_points_sum",
    "team_final_points",
    "team_final_position",
    "prev_season_points",
    "prev_season_avg_pos",
    "prev_season_win_rate",
]

def assign_tier(pos):
    if pd.isna(pos):
        return "Unknown"
    p = int(pos)
    if p == 1:
        return "Champion"
    elif p <= 3:
        return "Podium"
    elif p <= 5:
        return "Top 5"
    elif p <= 10:
        return "Top 10"
    elif p <= 15:
        return "Midfield"
    else:
        return "Backmarker"

def get_spearman(y_true, y_pred):
    corr_val, _ = spearmanr(y_true, y_pred)
    return float(corr_val)  # type: ignore

def train_model():
    # ── Load ──────────────────────────────────────────────────────────────
    features_path = os.path.join(PROC_DIR, "features.csv")
    df = pd.read_csv(features_path)
    df = df.dropna(subset=["champ_position"])
    print(f"Loaded {len(df)} rows across {df['year'].nunique()} seasons ({df['year'].min()}–{df['year'].max()})")

    X      = df[FEATURE_COLUMNS].fillna(0)
    y_reg  = df["champ_position"]
    y_tier = df["champ_position"].apply(assign_tier)

    X_train, X_test, y_train, y_test, yt_train, yt_test = train_test_split(
        X, y_reg, y_tier, test_size=0.2, random_state=42
    )

    # ── Regression models ─────────────────────────────────────────────────
    print("\n── Regression ────────────────────────────────────────────────")
    candidates = {
        "Ridge":             Ridge(alpha=1.0),
        "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
    }

    best_model = None
    best_name  = ""
    best_spearman = -999

    for name, m in candidates.items():
        cv = cross_val_score(m, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
        cv_rmse = float(np.sqrt(-cv.mean()))

        m.fit(X_train, y_train)
        y_pred    = m.predict(X_test)
        test_r2   = float(r2_score(y_test, y_pred))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        sp        = get_spearman(y_test, y_pred)

        print(f"  {name:20s} | CV RMSE: {cv_rmse:.3f} | R²: {test_r2:.3f} | Spearman: {sp:.3f}")

        if sp > best_spearman:
            best_spearman = sp
            best_model    = m
            best_name     = name

    print(f"\n  Best: {best_name} (Spearman={best_spearman:.3f})")

    if best_model is not None and hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
        print("\n  Top 10 features:")
        print(imp.head(10).to_string())

    # ── Tier classifier ───────────────────────────────────────────────────
    print("\n── Classification ────────────────────────────────────────────")
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    cv_clf = cross_val_score(clf, X_train, yt_train, cv=5, scoring="accuracy")
    clf.fit(X_train, yt_train)
    test_acc = clf.score(X_test, yt_test)
    print(f"  Tier Classifier | CV Acc: {cv_clf.mean():.3f} | Test Acc: {test_acc:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────
    reg_path = os.path.join(MODEL_DIR, "championship_model.pkl")
    clf_path = os.path.join(MODEL_DIR, "tier_classifier.pkl")

    with open(reg_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"\n  Saved regression model → {reg_path}")
    print(f"  Saved tier classifier  → {clf_path}")
    return best_model, clf

if __name__ == "__main__":
    train_model()