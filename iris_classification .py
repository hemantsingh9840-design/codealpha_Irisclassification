"""
CodeAlpha Data Science Internship
Task 1: Iris Flower Classification

Trains a machine learning model to classify Iris flowers into one of three
species (setosa, versicolor, virginica) based on their sepal/petal
measurements, then evaluates its accuracy on held-out test data.

Dataset: The classic Iris dataset, loaded directly from scikit-learn
(no external download needed — it's the same dataset CodeAlpha links to).

Key Concepts Used: pandas, scikit-learn, train/test split, classification,
model evaluation (accuracy, confusion matrix, classification report).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def load_data():
    """Load the Iris dataset into a pandas DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris.target_names


def explore_data(df):
    """Print a quick summary of the dataset and save a pairplot figure."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nClass distribution:")
    print(df["species"].value_counts())
    print("\nSummary statistics:")
    print(df.describe())

    # Visualize relationships between features, colored by species
    sns.pairplot(df, hue="species", diag_kind="hist")
    plt.savefig("iris_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved feature pairplot to 'iris_pairplot.png'")


def train_model(df):
    """Split the data, scale features, and train a RandomForest classifier."""
    X = df.drop(columns=["species"])
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling (not strictly required for RandomForest, but good practice
    # and keeps the pipeline reusable if the model is swapped later)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_test_scaled, y_test, X.columns


def evaluate_model(model, X_test_scaled, y_test):
    """Print accuracy, classification report, and save a confusion matrix plot."""
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Iris Classification")
    plt.savefig("iris_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved confusion matrix to 'iris_confusion_matrix.png'")

    return accuracy


def show_feature_importance(model, feature_names):
    """Print and plot which features mattered most to the model."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)

    print("\nFeature Importance:")
    print(importances)

    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh", color="teal")
    plt.xlabel("Importance")
    plt.title("Feature Importance - Iris Classification")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("iris_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved feature importance chart to 'iris_feature_importance.png'")


def predict_new_sample(model, scaler, feature_names):
    """Demonstrate predicting the species of a brand-new flower measurement."""
    # Example measurement: sepal_length, sepal_width, petal_length, petal_width
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=feature_names)
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]

    print("\n" + "=" * 60)
    print("SAMPLE PREDICTION")
    print("=" * 60)
    print(f"Input measurements: {sample.iloc[0].to_dict()}")
    print(f"Predicted species: {prediction}")


def main():
    df, target_names = load_data()
    explore_data(df)
    model, scaler, X_test_scaled, y_test, feature_names = train_model(df)
    evaluate_model(model, X_test_scaled, y_test)
    show_feature_importance(model, feature_names)
    predict_new_sample(model, scaler, feature_names)


if __name__ == "__main__":
    main()
