from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_loader import load_cancer_data


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "cancer_model.pkl"


def evaluate_model() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 'python src/train.py' first."
        )

    dataset = load_cancer_data()
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]

    predictions = model.predict(dataset.x_test)
    matrix = confusion_matrix(dataset.y_test, predictions)

    print("Dataset: Breast Cancer Wisconsin")
    print("Class order: 0 = malignant, 1 = benign")
    print(f"Accuracy: {accuracy_score(dataset.y_test, predictions):.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            dataset.y_test,
            predictions,
            target_names=dataset.target_names,
        )
    )
    print("Confusion Matrix:")
    print(matrix)


if __name__ == "__main__":
    evaluate_model()
