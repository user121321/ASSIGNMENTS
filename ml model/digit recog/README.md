# Digit Recognizer

A handwritten digit recognition project built with **classical machine learning** — no neural networks. Uses HOG feature extraction + PCA dimensionality reduction, trained on the MNIST dataset.

---

## How It Works

```
Raw pixel image (28×28)
        ↓
  HOG Features          — captures edge & gradient patterns
        ↓
  StandardScaler        — zero mean, unit variance
        ↓
  PCA (100 components)  — reduces noise, speeds up training
        ↓
  Classifier            — SVM / Random Forest / k-NN
        ↓
  Predicted Digit (0–9)
```

---

## Project Structure

```
digit-recognizer/
├── data/
│   ├── dataset_info.txt       # dataset notes
│   └── samples/               # generated test images (after running generate_sample.py)
├── src/
│   ├── data_loader.py         # downloads MNIST from OpenML (cached after first run)
│   ├── preprocess.py          # HOG + StandardScaler + PCA pipeline
│   ├── train.py               # trains & evaluates classifier, saves model to models/
│   ├── predict.py             # runs inference on any image file
│   └── generate_sample.py     # saves sample MNIST digits as PNG for quick testing
├── models/                    # saved .pkl files (created after training)
├── requirements.txt
└── README.md
```

---

## Dataset

- **Name:** MNIST (Modified National Institute of Standards and Technology)
- **Size:** 70,000 greyscale images, 28×28 pixels
- **Classes:** 10 (digits 0–9)
- **Source:** Downloaded automatically from [OpenML](https://www.openml.org/d/554) on first run
- **Cache:** Stored locally in `data/` — no internet needed after the first download
- **Split:** 80% train / 20% test (stratified)

---

## Setup

### 1. Clone or download the project

```bash
cd digit-recognizer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**

| Package | Purpose |
|---|---|
| scikit-learn | SVM, Random Forest, k-NN, PCA, StandardScaler |
| scikit-image | HOG feature extraction |
| numpy | Array operations |
| matplotlib | Visualisation |
| Pillow | Image loading & resizing |
| joblib | Saving/loading models |
| tqdm | Progress bars |
| pandas | Required by scikit-learn's OpenML fetcher |

---

## Usage

### Step 1 — Train a model

```bash
# RBF-SVM — best accuracy (~98%), recommended
python src/train.py --model svm

# Random Forest — faster training (~97%)
python src/train.py --model rf --n-estimators 200

# k-Nearest Neighbours — simple baseline (~96%)
python src/train.py --model knn --k 5
```

On first run, MNIST (~55 MB) is downloaded and cached in `data/`. Training an SVM on 56,000 samples takes 2–5 minutes depending on your machine.

Trained models are saved to `models/`:
- `models/svm_classifier.pkl`
- `models/preprocessor.pkl`

---

### Step 2 — Generate test images (optional)

If you don't have your own digit images, generate samples directly from MNIST:

```bash
python src/generate_sample.py
```

This saves one PNG per digit (0–9) into `data/samples/`.

---

### Step 3 — Predict

```bash
# Predict and print result
python src/predict.py --image data/samples/digit_3.png

# Predict and show image with result
python src/predict.py --image data/samples/digit_3.png --show

# Use a different model
python src/predict.py --image data/samples/digit_7.png --model rf --show
```

The image can be **any size or colour** — it is automatically resized to 28×28 greyscale. If the background is light (e.g. white paper), it is auto-inverted so the digit appears bright on dark.

---

## Model Comparison

| Model | Test Accuracy | Training Time | Notes |
|---|---|---|---|
| SVM (RBF kernel) | ~98% | ~3 min | Best accuracy, recommended |
| Random Forest | ~97% | ~1 min | Fast, good interpretability |
| k-NN (k=5) | ~96% | instant | No training step, slow at inference |

---

## CLI Reference

### `train.py`

| Flag | Default | Description |
|---|---|---|
| `--model` | `svm` | Classifier: `svm`, `rf`, `knn` |
| `--no-pca` | off | Skip PCA reduction |
| `--pca-components` | `100` | Number of PCA components |
| `--C` | `10` | SVM regularisation parameter |
| `--n-estimators` | `200` | Random Forest number of trees |
| `--k` | `5` | k-NN number of neighbours |

### `predict.py`

| Flag | Required | Description |
|---|---|---|
| `--image` | ✓ | Path to image file (any format) |
| `--model` | | `svm` (default), `rf`, `knn` |
| `--show` | | Display the image alongside the prediction |

---

## Example Output

```
$ python src/train.py --model svm

Fetching MNIST from OpenML (cached after first download)...
Train samples : 56000
Test  samples : 14000
Extracting HOG features (train)...
Applying PCA → 100 components...
Training SVM...
Training completed in 187.3s

Test Accuracy : 98.21%

              precision    recall  f1-score
           0     0.9893    0.9939    0.9916
           1     0.9947    0.9956    0.9951
           ...
```

```
$ python src/predict.py --image data/samples/digit_3.png --show

Predicted digit : 3  (confidence: 96.4%)
```
