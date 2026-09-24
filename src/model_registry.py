"""Stable estimator definitions and ports for model training and artifacts."""

import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge

HISTORY_FEATURE_COLUMNS = [
    "prev_season_races_started",
    "prev_season_avg_finish_pos",
    "prev_season_std_finish_pos",
    "prev_season_points_sum",
    "prev_season_avg_grid_pos",
    "prev_season_win_rate",
    "prev_season_podium_rate",
    "prev_season_dnf_rate",
    "prev_season_points_per_race",
    "prev_season_quali_to_race_delta",
    "prev_season_sprint_points_sum",
    "prev_team_final_points",
    "prev_team_final_position",
]
COLD_START_FEATURE_COLUMNS = [
    "is_rookie",
    "returning_after_gap",
    "missing_driver_history",
    "missing_constructor_history",
]
FEATURE_COLUMNS = [*HISTORY_FEATURE_COLUMNS, *COLD_START_FEATURE_COLUMNS]

REGRESSION_HISTORY_NAME = "Random Forest (history only)"
REGRESSION_OPERATIONAL_NAME = "Random Forest + cold-start flags"
REGRESSION_BOOSTING_NAME = "Gradient Boosting"
REGRESSION_RIDGE_NAME = "Ridge"
REGRESSION_FEATURES = {
    "history": HISTORY_FEATURE_COLUMNS,
    "all": FEATURE_COLUMNS,
}


@dataclass(frozen=True)
class RegressionModelSpec:
    name: str
    factory: Callable[[], Any]
    feature_set: str


REGRESSION_MODEL_REGISTRY = (
    RegressionModelSpec(
        REGRESSION_RIDGE_NAME,
        lambda: Ridge(alpha=1.0),
        "all",
    ),
    RegressionModelSpec(
        REGRESSION_HISTORY_NAME,
        lambda: RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "history",
    ),
    RegressionModelSpec(
        REGRESSION_OPERATIONAL_NAME,
        lambda: RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "all",
    ),
    RegressionModelSpec(
        REGRESSION_BOOSTING_NAME,
        lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=5, random_state=42
        ),
        "all",
    ),
)


class TrainableModel(Protocol):
    def fit(self, features: pd.DataFrame, targets: pd.Series) -> Any: ...


class PredictiveModel(Protocol):
    def predict(self, features: pd.DataFrame) -> Any: ...


class ModelArtifactStore(Protocol):
    def save(self, model: Any, path: Path) -> None: ...

    def load(self, path: Path) -> Any: ...


class PickleModelArtifactStore:
    """Default local serializer used by the model and prediction entrypoints."""

    def save(self, model: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as artifact:
            pickle.dump(model, artifact)

    def load(self, path: Path) -> Any:
        with path.open("rb") as artifact:
            return pickle.load(artifact)


DEFAULT_MODEL_ARTIFACT_STORE = PickleModelArtifactStore()
REGRESSION_ARTIFACT = "championship_model.pkl"
TIER_ARTIFACT = "tier_classifier.pkl"


def create_regression_candidates() -> dict[str, tuple[Any, list[str]]]:
    """Create fresh estimators while preserving the public stable names/order."""
    return {
        spec.name: (spec.factory(), REGRESSION_FEATURES[spec.feature_set])
        for spec in REGRESSION_MODEL_REGISTRY
    }


def create_tier_classifier(
    random_state: int = 42,
    factory: Callable[..., Any] | None = None,
) -> Any:
    """Create the fixed operational/rolling tier classifier."""
    classifier_factory = RandomForestClassifier if factory is None else factory
    return classifier_factory(n_estimators=200, max_depth=8, random_state=random_state)


def create_bootstrap_regressor(
    n_estimators: int,
    random_state: int,
    factory: Callable[..., Any] | None = None,
) -> Any:
    """Create one bootstrap estimator with the frozen hyperparameters."""
    regressor_factory = RandomForestRegressor if factory is None else factory
    return regressor_factory(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,
    )


def fit_model(
    model: TrainableModel, features: pd.DataFrame, targets: pd.Series
) -> TrainableModel:
    """Training port shared by the model-training entrypoint."""
    model.fit(features, targets)
    return model


def predict_model(model: PredictiveModel, features: pd.DataFrame) -> Any:
    """Prediction port shared by the user-facing prediction entrypoint."""
    return model.predict(features)


def save_model_artifacts(
    regression_model: Any,
    tier_classifier: Any,
    model_dir: str | Path,
    store: ModelArtifactStore | None = None,
) -> tuple[Path, Path]:
    """Save the two established model artifacts through an injected store."""
    artifact_store = DEFAULT_MODEL_ARTIFACT_STORE if store is None else store
    directory = Path(model_dir)
    regression_path = directory / REGRESSION_ARTIFACT
    tier_path = directory / TIER_ARTIFACT
    artifact_store.save(regression_model, regression_path)
    artifact_store.save(tier_classifier, tier_path)
    return regression_path, tier_path


def load_model_artifacts(
    model_dir: str | Path,
    store: ModelArtifactStore | None = None,
) -> tuple[Any, Any]:
    """Load artifacts in the established regression/classifier order."""
    artifact_store = DEFAULT_MODEL_ARTIFACT_STORE if store is None else store
    directory = Path(model_dir)
    regression_model = artifact_store.load(directory / REGRESSION_ARTIFACT)
    tier_classifier = artifact_store.load(directory / TIER_ARTIFACT)
    return regression_model, tier_classifier
