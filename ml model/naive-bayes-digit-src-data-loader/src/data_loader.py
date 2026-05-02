from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class DigitDataset:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    source: str


def load_digit_data(
    test_size: float = 0.2,
    random_state: int = 42,
    max_samples: int | None = 10000,
) -> DigitDataset:
    """Load an online digit dataset and return train/test splits.

    The assignment asks for an online dataset, so MNIST is loaded from OpenML.
    A small fallback dataset is used only when OpenML is unavailable.
    """
    try:
        x, y = _load_openml_mnist(max_samples=max_samples, random_state=random_state)
        source = "MNIST from OpenML"
    except Exception as exc:
        print(f"Could not download MNIST from OpenML: {exc}")
        print("Using sklearn's built-in digits dataset as a fallback.")
        digits = load_digits()
        x = digits.data.astype(np.float64)
        y = digits.target.astype(str)
        source = "sklearn built-in digits fallback"

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return DigitDataset(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        source=source,
    )


def _load_openml_mnist(
    max_samples: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    x = mnist.data.astype(np.float64)
    y = mnist.target.astype(str)

    if max_samples is not None and max_samples < len(x):
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(x), size=max_samples, replace=False)
        x = x[indices]
        y = y[indices]

    return x, y
