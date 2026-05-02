"""
generate_sample.py
Saves a few sample digits from MNIST as PNG files so you can test predict.py.

Usage
-----
  python src/generate_sample.py
"""

import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import fetch_mnist

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "samples")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _, X_test, _, y_test = fetch_mnist()

    # Save one sample for each digit 0-9
    saved = {}
    for i, (img_flat, label) in enumerate(zip(X_test, y_test)):
        if label not in saved:
            img_arr = (img_flat * 255).astype(np.uint8).reshape(28, 28)
            path = os.path.join(OUT_DIR, f"digit_{label}.png")
            Image.fromarray(img_arr).save(path)
            saved[label] = path
            print(f"Saved digit {label} → {path}")
        if len(saved) == 10:
            break

    print(f"\nDone! Now run:")
    print(f'  python src/predict.py --image data/samples/digit_3.png --show')

if __name__ == "__main__":
    main()
