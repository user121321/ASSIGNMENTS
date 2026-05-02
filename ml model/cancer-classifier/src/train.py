from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import load_cancer_data


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "cancer_model.pkl"


def train_model() -> None:
    dataset = load_cancer_data()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=10000, random_state=42)),
        ]
    )

    model.fit(dataset.x_train, dataset.y_train)
    predictions = model.predict(dataset.x_test)
    accuracy = accuracy_score(dataset.y_test, predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": dataset.feature_names,
            "target_names": dataset.target_names,
        },
        MODEL_PATH,
    )

    print("Dataset: Breast Cancer Wisconsin")
    print(f"Classes: {', '.join(dataset.target_names)}")
    print(f"Training samples: {len(dataset.x_train)}")
    print(f"Testing samples: {len(dataset.x_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
