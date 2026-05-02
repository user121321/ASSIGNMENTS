# Naive Bayes Digit Classification

This project implements digit classification using a Naive Bayes classifier.
It uses the MNIST handwritten digit dataset from OpenML as the online dataset.
If MNIST cannot be downloaded, the code automatically falls back to the built-in
`sklearn` digits dataset so the assignment can still run offline.

## Project Structure

```text
naive-bayes-digit/
├── src/
│   ├── data_loader.py
│   ├── train.py
│   └── evaluate.py
├── models/
│   └── nb_model.pkl
├── requirements.txt
└── README.md
```

## Algorithm

Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.
It assumes that each feature contributes independently to the final class label.
For digit images, each pixel value is treated as a feature.

This project uses `MultinomialNB`, which works well with non-negative pixel
intensity features.

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

This will:

1. Download/load the digit dataset.
2. Split it into training and testing data.
3. Train a Multinomial Naive Bayes classifier.
4. Save the trained model to `models/nb_model.pkl`.

## Evaluate the Model

Run:

```bash
python src/evaluate.py
```

This will print:

- Accuracy score
- Classification report
- Confusion matrix

## Notes for Assignment

The main requirement is Naive Bayes classification using an online dataset.
MNIST from OpenML satisfies the online dataset requirement. A neural network is
not required for this implementation, but it can be mentioned as an alternative
model for comparison.
