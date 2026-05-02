"""
preprocess.py
Feature engineering pipeline for classical ML digit recognition.

Steps:
  1. Reshape flat 784-d vectors → 28×28 images
  2. Extract HOG (Histogram of Oriented Gradients) features
  3. Optionally reduce dimensionality with PCA
"""

import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def extract_hog_features(X: np.ndarray) -> np.ndarray:
    """
    Convert an array of flat 784-d pixel vectors to HOG feature vectors.

    Parameters
    ----------
    X : np.ndarray, shape (N, 784)

    Returns
    -------
    np.ndarray, shape (N, hog_feature_size)
    """
    features = []
    for img in X:
        img_2d = img.reshape(28, 28)
        feat = hog(
            img_2d,
            orientations=9,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )
        features.append(feat)
    return np.array(features, dtype=np.float32)


class Preprocessor:
    """
    Fit-once / transform-many pipeline:
      raw pixels → HOG features → StandardScaler → (optional) PCA
    """

    def __init__(self, use_pca: bool = True, n_components: int = 100):
        self.use_pca = use_pca
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42) if use_pca else None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        print("Extracting HOG features (train)...")
        X_hog = extract_hog_features(X)
        X_scaled = self.scaler.fit_transform(X_hog)
        if self.use_pca:
            print(f"Applying PCA → {self.n_components} components...")
            X_out = self.pca.fit_transform(X_scaled)
        else:
            X_out = X_scaled
        return X_out

    def transform(self, X: np.ndarray) -> np.ndarray:
        print("Extracting HOG features (test)...")
        X_hog = extract_hog_features(X)
        X_scaled = self.scaler.transform(X_hog)
        return self.pca.transform(X_scaled) if self.use_pca else X_scaled

    def save(self, path: str = None):
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = path or os.path.join(MODEL_DIR, "preprocessor.pkl")
        joblib.dump(self, path)
        print(f"Preprocessor saved → {path}")

    @staticmethod
    def load(path: str = None) -> "Preprocessor":
        path = path or os.path.join(MODEL_DIR, "preprocessor.pkl")
        return joblib.load(path)


if __name__ == "__main__":
    from data_loader import fetch_mnist

    X_train, X_test, y_train, y_test = fetch_mnist()
    prep = Preprocessor(use_pca=True, n_components=100)
    X_train_proc = prep.fit_transform(X_train)
    X_test_proc = prep.transform(X_test)
    print("Train features shape:", X_train_proc.shape)
    print("Test  features shape:", X_test_proc.shape)
