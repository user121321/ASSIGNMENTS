# KNN Iris Classification

This project implements Iris flower classification using the K-Nearest
Neighbors algorithm.

The Iris dataset contains flower measurements and three target classes:

- Iris setosa
- Iris versicolor
- Iris virginica

## Project Structure

```text
knn-iris-classifier/
|-- src/
|   |-- data_loader.py
|   |-- train.py
|   |-- evaluate.py
|   `-- predict.py
|-- models/
|   `-- knn_iris_model.pkl
|-- requirements.txt
`-- README.md
```

## Algorithm

K-Nearest Neighbors is a supervised machine learning algorithm. It classifies a
new sample by comparing it with the closest training samples. The class most
common among the nearest neighbors becomes the predicted class.

This project uses `KNeighborsClassifier` from `scikit-learn` with `k = 5`.

## Features Used

The model uses four Iris flower measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python src/train.py
```

This trains the KNN model and saves it to:

```text
models/knn_iris_model.pkl
```

## Evaluate the Model

Run:

```bash
python src/evaluate.py
```

This prints the accuracy score, classification report, and confusion matrix.

## Predict One Flower

Pass four measurements in this order:

```text
sepal_length sepal_width petal_length petal_width
```

Example:

```bash
python src/predict.py 5.1 3.5 1.4 0.2
```

Expected output:

```text
Predicted Iris class: setosa
```
