from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_loader import load_digit_data


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "nb_model.pkl"


def evaluate_model() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 'python src/train.py' first."
        )

    dataset = load_digit_data()
    model = joblib.load(MODEL_PATH)

    predictions = model.predict(dataset.x_test)

    print(f"Dataset source: {dataset.source}")
    print(f"Accuracy: {accuracy_score(dataset.y_test, predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(dataset.y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(dataset.y_test, predictions))


if __name__ == "__main__":
    evaluate_model()
