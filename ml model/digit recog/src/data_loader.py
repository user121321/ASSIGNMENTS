"""
data_loader.py
Fetches the MNIST dataset online via sklearn's OpenML integration.
Returns raw numpy arrays split into train/test sets.
"""

import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def fetch_mnist(test_size: float = 0.2, random_state: int = 42):
    """
    Download (or load from cache) MNIST and return train/test splits.

    Returns
    -------
    X_train, X_test : np.ndarray, shape (N, 784)  — pixel values in [0, 1]
    y_train, y_test : np.ndarray, shape (N,)       — integer labels 0-9
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Fetching MNIST from OpenML (cached after first download)...")
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        data_home=DATA_DIR,
        parser="liac-arff",
    )

    X = mnist.data.astype(np.float32) / 255.0   # normalise to [0, 1]
    y = mnist.target.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train samples : {len(X_train)}")
    print(f"Test  samples : {len(X_test)}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = fetch_mnist()
    print("Sample label:", y_train[0])
