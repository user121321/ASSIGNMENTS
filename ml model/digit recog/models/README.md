# Digit Recognizer

Classical machine-learning pipeline for handwritten digit recognition (MNIST).  
**No neural networks** — uses HOG features + PCA + SVM / Random Forest / k-NN.

---

## Project Structure

```
digit-recognizer/
├── data/
│   └── dataset_info.txt      # dataset notes; MNIST cache stored here
├── src/
│   ├── data_loader.py        # fetches MNIST via OpenML
│   ├── preprocess.py         # HOG feature extraction + StandardScaler + PCA
│   ├── train.py              # trains & evaluates classifier; saves model
│   └── predict.py            # inference on a single image
├── models/                   # saved .pkl files (created after training)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** `scikit-image` is required for HOG. It is pulled in automatically as a dependency of scikit-learn extras; if not, run `pip install scikit-image`.

---

## Usage

### 1 — Train

```bash
# RBF-SVM (recommended, ~98% accuracy)
python src/train.py --model svm

# Random Forest (~97% accuracy, faster training)
python src/train.py --model rf --n-estimators 200

# k-Nearest Neighbours (simple baseline)
python src/train.py --model knn --k 5
```

First run downloads MNIST (~55 MB) and caches it under `data/`.  
Trained models are saved to `models/`.

### 2 — Predict

```bash
python src/predict.py --image path/to/digit.png --model svm

# Show the preprocessed image alongside the prediction
python src/predict.py --image path/to/digit.png --show
```

The image can be any size or colour — it is automatically converted to 28×28 greyscale and background-inverted if needed.

---

## Feature Pipeline

```
Raw pixels (784-d)
  → HOG descriptors   orientations=9, pixels_per_cell=(7,7), cells_per_block=(2,2)
  → StandardScaler    zero mean, unit variance
  → PCA               100 components (retains ~95% variance)
  → Classifier        SVM / RF / KNN
```

---

## Expected Accuracy (test set, 20% split)

| Model         | Accuracy |
|---------------|----------|
| SVM (RBF)     | ~98%     |
| Random Forest | ~97%     |
| k-NN (k=5)    | ~96%     |

---

## CLI Reference

### train.py

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `svm` | Classifier: `svm`, `rf`, `knn` |
| `--no-pca` | off | Skip PCA reduction |
| `--pca-components` | 100 | Number of PCA components |
| `--C` | 10 | SVM regularisation |
| `--n-estimators` | 200 | RF number of trees |
| `--k` | 5 | KNN neighbours |

### predict.py

| Flag | Required | Description |
|------|----------|-------------|
| `--image` | ✓ | Path to image file |
| `--model` | | `svm` (default), `rf`, `knn` |
| `--show` | | Display image + prediction |
