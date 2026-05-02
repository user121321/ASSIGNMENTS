from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "knn_iris_model.pkl"


def predict_flower(measurements: list[float]) -> None:
    if len(measurements) != 4:
        raise ValueError(
            "Provide exactly 4 values: sepal_length sepal_width petal_length petal_width"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 'python src/train.py' first."
        )

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_names = saved["feature_names"]
    target_names = saved["target_names"]

    sample = np.array([measurements], dtype=np.float64)
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    print("Input measurements:")
    for feature_name, value in zip(feature_names, measurements):
        print(f"{feature_name}: {value}")

    print(f"\nPredicted Iris class: {target_names[prediction]}")
    print("Class probabilities:")
    for class_name, probability in zip(target_names, probabilities):
        print(f"{class_name}: {probability:.2%}")


def main() -> None:
    if len(sys.argv) != 5:
        print("Usage: python src/predict.py sepal_length sepal_width petal_length petal_width")
        print("Example: python src/predict.py 5.1 3.5 1.4 0.2")
        return

    measurements = [float(value) for value in sys.argv[1:]]
    predict_flower(measurements)


if __name__ == "__main__":
    main()
