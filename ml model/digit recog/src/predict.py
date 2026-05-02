"""
predict.py
Run inference on a single image or a folder of images.

Usage
-----
  python predict.py --image path/to/digit.png
  python predict.py --image path/to/digit.png --model svm
  python predict.py --image path/to/digit.png --show
"""

import argparse
import os
import sys

import joblib
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import Preprocessor

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_artifacts(model_name: str = "svm"):
    model_path = os.path.join(MODEL_DIR, f"{model_name}_classifier.pkl")
    prep_path = os.path.join(MODEL_DIR, "preprocessor.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            f"Run: python src/train.py --model {model_name}"
        )
    if not os.path.exists(prep_path):
        raise FileNotFoundError(
            f"Preprocessor not found at '{prep_path}'.\n"
            f"Run: python src/train.py"
        )

    clf = joblib.load(model_path)
    prep = Preprocessor.load(prep_path)
    return clf, prep


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load any image file, convert to 28×28 greyscale, invert if needed,
    and return a normalised flat (1, 784) array.
    """
    img = Image.open(image_path).convert("L")   # greyscale
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)

    # If background is light (like a white paper scan), invert so digit is bright
    if arr.mean() > 127:
        arr = 255.0 - arr

    arr /= 255.0
    return arr.flatten().reshape(1, -1)


def predict(image_path: str, model_name: str = "svm", show: bool = False):
    clf, prep = load_artifacts(model_name)

    X_raw = preprocess_image(image_path)
    X_proc = prep.transform(X_raw)

    label = clf.predict(X_proc)[0]

    # Confidence — available for SVM (decision_function) and RF/KNN (predict_proba)
    confidence_str = ""
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_proc)[0]
        conf = proba.max() * 100
        confidence_str = f"  (confidence: {conf:.1f}%)"
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_proc)[0]
        # Softmax-like normalisation for display only
        e = np.exp(scores - scores.max())
        conf = (e / e.sum()).max() * 100
        confidence_str = f"  (confidence: {conf:.1f}%)"

    print(f"Predicted digit : {label}{confidence_str}")

    if show:
        import matplotlib.pyplot as plt

        img_arr = X_raw.reshape(28, 28)
        plt.figure(figsize=(4, 4))
        plt.imshow(img_arr, cmap="gray")
        plt.title(f"Prediction: {label}{confidence_str}", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return int(label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict digit from an image.")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--model", default="svm", choices=["svm", "rf", "knn"])
    parser.add_argument("--show", action="store_true", help="Display the image with prediction")
    args = parser.parse_args()

    predict(args.image, model_name=args.model, show=args.show)
