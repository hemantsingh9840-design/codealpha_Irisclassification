# CodeAlpha_IrisClassification 🌸

## Data Science Internship — CodeAlpha
**Task 1: Iris Flower Classification**

A machine learning model that classifies Iris flowers into one of three
species — *setosa*, *versicolor*, or *virginica* — based on their sepal and
petal measurements.

## Dataset
The classic Iris dataset (150 samples, 4 features, 3 balanced classes),
loaded directly via `scikit-learn`'s built-in `load_iris()` function — no
external download required.

## What the Script Does
1. Loads and explores the dataset (summary stats, class distribution)
2. Visualizes feature relationships with a pairplot
3. Splits data into train/test sets (80/20) and scales features
4. Trains a `RandomForestClassifier`
5. Evaluates the model: accuracy, classification report, confusion matrix
6. Shows which features matter most (feature importance)
7. Predicts the species of a brand-new sample measurement

## Results
- **Accuracy: 90%** on held-out test data
- Petal length and petal width are the most important features for
  classification (setosa is perfectly separable; versicolor/virginica have
  minor overlap)

## Key Concepts Used
- pandas, matplotlib, seaborn
- scikit-learn: `train_test_split`, `StandardScaler`, `RandomForestClassifier`
- Classification evaluation metrics (accuracy, precision, recall, F1,
  confusion matrix)

## How to Run
```bash
pip install pandas scikit-learn matplotlib seaborn
python3 iris_classification.py
```

Generates three chart images in the same folder:
- `iris_pairplot.png`
- `iris_confusion_matrix.png`
- `iris_feature_importance.png`

## Author
Submitted as part of the CodeAlpha Data Science Internship.

## About CodeAlpha
CodeAlpha is a software development company offering internship programs in
Data Science, Python, Web Development, and more.
[www.codealpha.tech](https://www.codealpha.tech)
