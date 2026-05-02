# Machine Learning Assignment Projects

This repository contains multiple beginner-friendly machine learning
classification projects created for assignment work.

Each project is kept in a separate folder with its own source code, saved model,
requirements file, and README.

## Projects Included

### 1. Naive Bayes Digit Classification

Folder:

```text
naive-bayes-digit-src-data-loader/
```

This project classifies handwritten digit images using the MNIST dataset.

Algorithm used:

```text
Multinomial Naive Bayes
```

Dataset:

```text
MNIST from OpenML
```

Output:

```text
Predicted digit: 0 to 9
```

Run:

```bash
cd "naive-bayes-digit-src-data-loader"
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```

Predict one digit image:

```bash
python src/predict.py "path/to/digit_image.png"
```

Example result:

```text
Accuracy: 0.8300
```

### 2. KNN Iris Classification

Folder:

```text
knn-iris-classifier/
```

This project classifies Iris flowers using flower measurements.

Algorithm used:

```text
K-Nearest Neighbors
```

Dataset:

```text
Iris dataset
```

Input features:

```text
Sepal length
Sepal width
Petal length
Petal width
```

Output classes:

```text
setosa
versicolor
virginica
```

Run:

```bash
cd "knn-iris-classifier"
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```

Predict one Iris flower:

```bash
python src/predict.py 5.1 3.5 1.4 0.2
```

Example result:

```text
Accuracy: 0.9333
Predicted Iris class: setosa
```

### 3. Cancer Classification

Folder:

```text
cancer-classifier/
```

This project classifies breast cancer data as malignant or benign.

Algorithm used:

```text
Logistic Regression
```

Dataset:

```text
Breast Cancer Wisconsin dataset
```

Output classes:

```text
malignant
benign
```

Run:

```bash
cd "cancer-classifier"
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```

Example result:

```text
Accuracy: 0.9825
```

Confusion matrix:

```text
Class order: 0 = malignant, 1 = benign

[[41  1]
 [ 1 71]]
```

Meaning:

```text
41 malignant cases correctly predicted as malignant
1 malignant case wrongly predicted as benign
1 benign case wrongly predicted as malignant
71 benign cases correctly predicted as benign
```

## Repository Structure

```text
ml-models/
|-- naive-bayes-digit-src-data-loader/
|-- knn-iris-classifier/
|-- cancer-classifier/
`-- README.md
```

## Requirements

Each project has its own `requirements.txt` file.

Common libraries used:

```text
numpy
scikit-learn
joblib
pillow
```

`pillow` is only required for digit image prediction.

## How to Upload to GitHub

1. Create a new GitHub repository.
2. Upload these project folders and this `README.md`.
3. Keep each model folder separate.
4. Open each folder's README for project-specific details.

## Notes

These projects are made for learning and assignment purposes. They show how to:

- Load datasets
- Split data into training and testing sets
- Train classification models
- Save trained models
- Evaluate models using accuracy, classification report, and confusion matrix
- Make predictions using saved models
