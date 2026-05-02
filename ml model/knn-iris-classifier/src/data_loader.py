from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class IrisDataset:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_names: np.ndarray


def load_iris_data(test_size: float = 0.2, random_state: int = 42) -> IrisDataset:
    iris = load_iris()

    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=test_size,
        random_state=random_state,
        stratify=iris.target,
    )

    return IrisDataset(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(iris.feature_names),
        target_names=iris.target_names,
    )
