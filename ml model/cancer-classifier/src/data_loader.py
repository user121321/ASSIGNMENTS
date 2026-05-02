from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class CancerDataset:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: np.ndarray
    target_names: np.ndarray


def load_cancer_data(test_size: float = 0.2, random_state: int = 42) -> CancerDataset:
    cancer = load_breast_cancer()

    x_train, x_test, y_train, y_test = train_test_split(
        cancer.data,
        cancer.target,
        test_size=test_size,
        random_state=random_state,
        stratify=cancer.target,
    )

    return CancerDataset(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=cancer.feature_names,
        target_names=cancer.target_names,
    )
