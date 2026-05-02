"""
train.py
Train classical ML classifiers on HOG+PCA features.

Supported models
----------------
  svm   — RBF-kernel SVM  (best accuracy, ~98%)
  rf    — Random Forest    (fast, interpretable)
  knn   — k-Nearest Neighbours (simple baseline)

Usage
-----
  python train.py --model svm
  python train.py --model rf --n-estimators 200
  python train.py --model knn --k 5
"""

import argparse
import os
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import fetch_mnist
from preprocess import Preprocessor

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def build_model(name: str, **kwargs):
    if name == "svm":
        return SVC(
            kernel="rbf",
            C=kwargs.get("C", 10),
            gamma=kwargs.get("gamma", "scale"),
            decision_function_shape="ovr",
            verbose=True,
        )
    elif name == "rf":
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", None),
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )
    elif name == "knn":
        return KNeighborsClassifier(
            n_neighbors=kwargs.get("k", 5),
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Choose svm | rf | knn.")


def train(model_name: str = "svm", use_pca: bool = True, n_components: int = 100, **model_kwargs):
    # ── 1. Data ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = fetch_mnist()

    # ── 2. Features ──────────────────────────────────────────────────────────
    prep = Preprocessor(use_pca=use_pca, n_components=n_components)
    X_train_proc = prep.fit_transform(X_train)
    X_test_proc = prep.transform(X_test)
    prep.save()

    # ── 3. Train ─────────────────────────────────────────────────────────────
    clf = build_model(model_name, **model_kwargs)
    print(f"\nTraining {model_name.upper()}...")
    t0 = time.time()
    clf.fit(X_train_proc, y_train)
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s")

    # ── 4. Evaluate ──────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test_proc)
    acc = np.mean(y_pred == y_test) * 100
    print(f"\nTest Accuracy : {acc:.2f}%\n")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # ── 5. Save ──────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{model_name}_classifier.pkl")
    joblib.dump(clf, model_path)
    print(f"\nModel saved → {model_path}")

    return clf, prep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a digit recognizer (no neural nets).")
    parser.add_argument("--model", default="svm", choices=["svm", "rf", "knn"], help="Classifier to use")
    parser.add_argument("--no-pca", action="store_true", help="Skip PCA reduction")
    parser.add_argument("--pca-components", type=int, default=100, help="Number of PCA components")
    # SVM
    parser.add_argument("--C", type=float, default=10, help="SVM regularisation parameter")
    # RF
    parser.add_argument("--n-estimators", type=int, default=200, help="RF number of trees")
    # KNN
    parser.add_argument("--k", type=int, default=5, help="KNN neighbours")
    args = parser.parse_args()

    model_kwargs = {"C": args.C, "n_estimators": args.n_estimators, "k": args.k}
    train(
        model_name=args.model,
        use_pca=not args.no_pca,
        n_components=args.pca_components,
        **model_kwargs,
    )
