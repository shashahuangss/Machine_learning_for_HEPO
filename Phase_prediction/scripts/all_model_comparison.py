import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def load_data(path="../data/phase_train_data.xlsx"):
    """Load Excel data, extract features (36 columns) and target (column 37)."""
    df = pd.read_excel(path)
    X = df.iloc[:, :36].values
    y = df.iloc[:, 36].values
    feature_names = df.columns[:36].tolist()
    return X, y, feature_names


def get_classifiers():
    """Return a dictionary of classifiers to compare."""
    return {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=42),
        "Neural Network": MLPClassifier(max_iter=1000, random_state=42),
    }


def evaluate_classifiers(classifiers, X_train, X_test, y_train, y_test):
    """Train all classifiers and return metrics."""
    results = []

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        f1 = f1_score(y_test, y_pred, average="binary")
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"{name}: Accuracy={acc:.4f}, Precision={precision:.4f}, "
              f"Recall={recall:.4f}, F1={f1:.4f}, ROC AUC={roc_auc:.4f}")

        results.append({
            "Classifier": name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
        })

    return pd.DataFrame(results)


def main():
    # Load data
    X, y, feature_names = load_data()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Get classifiers
    classifiers = get_classifiers()

    # Evaluate all classifiers
    print("=" * 60)
    print("Classifier Comparison Results")
    print("=" * 60)
    results_df = evaluate_classifiers(
        classifiers, X_train_scaled, X_test_scaled, y_train, y_test
    )

    # Save results
    results_df.to_excel("../results/model_comparison_accuracy.xlsx", index=False)
    print(f"\nResults saved to ../results/model_comparison_accuracy.xlsx")


if __name__ == "__main__":
    main()
