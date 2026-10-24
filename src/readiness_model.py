
"""Baseline model for enterprise digital transformation readiness."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


FEATURES = [
    "innovation_investment",
    "digital_skills",
    "technology_adoption",
    "leadership_support",
    "process_integration",
]
TARGET = "readiness_score"


def train_model(data_path: str = "data/enterprise_innovation.csv") -> dict:
    """Train and evaluate a baseline model on a documented dataset."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Add the dataset before training."
        )

    data = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "feature_importance": dict(
            zip(FEATURES, model.feature_importances_.tolist())
        ),
    }


if __name__ == "__main__":
    print(train_model())
  
