# Cancer Classification Model

This project classifies breast cancer data using a machine learning model.
It uses the Breast Cancer Wisconsin dataset from `scikit-learn`.

The target classes are:

- malignant
- benign

## Algorithm

This project uses Logistic Regression. The features are scaled using
`StandardScaler` before training.

## Project Structure

```text
cancer-classifier/
|-- src/
|   |-- data_loader.py
|   |-- train.py
|   `-- evaluate.py
|-- models/
|   `-- cancer_model.pkl
|-- requirements.txt
`-- README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

## Evaluate

```bash
python src/evaluate.py
```

The evaluation prints:

- Accuracy
- Classification report
- Confusion matrix

## Confusion Matrix Meaning

The confusion matrix compares actual test results with predicted results.

Rows represent the actual class.
Columns represent the predicted class.

Class order:

```text
0 = malignant
1 = benign
```
