from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
from PIL import Image, ImageOps


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "nb_model.pkl"


def prepare_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    image = ImageOps.autocontrast(image)

    pixels = np.asarray(image)

    # MNIST digits are light strokes on a dark background.
    if pixels.mean() > 127:
        image = ImageOps.invert(image)

    pixels = np.asarray(image)
    mask = pixels > 30

    if mask.any():
        y_coords, x_coords = np.where(mask)
        image = image.crop(
            (
                x_coords.min(),
                y_coords.min(),
                x_coords.max() + 1,
                y_coords.max() + 1,
            )
        )

    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    left = (28 - image.width) // 2
    top = (28 - image.height) // 2
    canvas.paste(image, (left, top))

    features = np.asarray(canvas, dtype=np.float32) / 255.0
    return features.reshape(1, -1)


def predict_digit(image_path: str) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 'python src/train.py' first."
        )

    model = joblib.load(MODEL_PATH)
    features = prepare_image(image_path)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    print(f"Image: {image_path}")
    print(f"Predicted digit: {prediction}")
    print("Top guesses:")

    top_results = sorted(
        zip(model.classes_, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )[:3]

    for digit, probability in top_results:
        print(f"{digit}: {probability:.2%}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/digit_image.png")
        return

    predict_digit(sys.argv[1])


if __name__ == "__main__":
    main()
