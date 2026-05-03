import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error as MAE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.ensemble import RandomForestRegressor as RFR, GradientBoostingRegressor as GBR
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor as XGB


def load_data(path="../data/Ef_train_data.xlsx"):
    """Load Excel data, drop empty columns, duplicates, and NaN rows."""
    data = pd.read_excel(path, sheet_name="Sheet1")
    data = data.dropna(axis=1, how="all").drop_duplicates().dropna()
    X = data.iloc[:, :62]
    y = data["Formation energy"]
    return X, y


def get_models():
    """Return a dictionary of models to compare."""
    return {
        "LR": LinearRegression(),
        "LASSO": Lasso(alpha=0.001, max_iter=100000),
        "Ridge": Ridge(alpha=0.001),
        "KRR": KRR(kernel="poly", alpha=0.001),
        "KNN": KNR(n_neighbors=10),
        "SVM": SVR(kernel="rbf", C=1000, gamma=0.01),
        "RF": RFR(n_estimators=87, max_depth=17, random_state=20),
        "GBoost": GBR(n_estimators=100, max_depth=10, random_state=20),
        "XGBoost": XGB(n_estimators=70, max_depth=13, random_state=20),
        "ANN": MLPRegressor(
            activation="identity",
            max_iter=100000,
            hidden_layer_sizes=(9, 1),
            random_state=20,
        ),
    }


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Train all models and return metrics."""
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mae_train = MAE(y_train, y_train_pred)
        mae_test = MAE(y_test, y_test_pred)

        print(f"{name}: R2 train={r2_train:.4f}, test={r2_test:.4f} | "
              f"MAE train={mae_train:.4f}, test={mae_test:.4f}")

        results.append({
            "Model": name,
            "R2_train": r2_train,
            "R2_test": r2_test,
            "MAE_train": mae_train,
            "MAE_test": mae_test,
        })

    return pd.DataFrame(results)


def main():
    # Load data
    X, y = load_data()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=20, shuffle=True
    )

    # Standardize
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Get models
    models = get_models()

    # Evaluate all models
    print("=" * 60)
    print("Model Comparison Results")
    print("=" * 60)
    results_df = evaluate_models(models, X_train_scaled, X_test_scaled, y_train, y_test)

    # Save results
    results_df.to_excel("../results/model_comparison_accuracy.xlsx", index=False)
    print(f"\nResults saved to ../results/model_comparison_accuracy.xlsx")


if __name__ == "__main__":
    main()
