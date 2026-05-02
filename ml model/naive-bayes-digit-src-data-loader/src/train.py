from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from data_loader import load_digit_data


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "nb_model.pkl"


def train_model() -> None:
    dataset = load_digit_data()

    model = MultinomialNB()
    model.fit(dataset.x_train, dataset.y_train)

    predictions = model.predict(dataset.x_test)
    accuracy = accuracy_score(dataset.y_test, predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Dataset source: {dataset.source}")
    print(f"Training samples: {len(dataset.x_train)}")
    print(f"Testing samples: {len(dataset.x_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
