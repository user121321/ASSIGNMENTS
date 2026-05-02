from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import load_iris_data


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "knn_iris_model.pkl"


def train_model(n_neighbors: int = 5) -> None:
    dataset = load_iris_data()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
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
            "n_neighbors": n_neighbors,
        },
        MODEL_PATH,
    )

    print("Dataset: Iris")
    print(f"Features: {', '.join(dataset.feature_names)}")
    print(f"Classes: {', '.join(dataset.target_names)}")
    print(f"K value: {n_neighbors}")
    print(f"Training samples: {len(dataset.x_train)}")
    print(f"Testing samples: {len(dataset.x_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
