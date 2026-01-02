import argparse
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description="Train an Iris Decision Tree classifier.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data used for testing.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    # Train Decision Tree
    model = DecisionTreeClassifier(random_state=args.random_state)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Model trained with test_size={args.test_size}, random_state={args.random_state}")
    print(f"Accuracy: {acc:.4f}")

    # Ensure outputs directory exists at project root
    project_root = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(project_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Decision Tree)")

    out_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved confusion matrix to: {out_path}")


if __name__ == "__main__":
    main()
