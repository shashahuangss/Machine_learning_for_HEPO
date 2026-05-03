import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import r2_score, mean_absolute_error as MAE


def load_and_clean(path="../data/Ef_train_data.xlsx"):
    """Load Excel data, drop empty columns, duplicates, and NaN rows."""
    data = pd.read_excel(path, sheet_name="Sheet1")
    data = data.dropna(axis=1, how="all").drop_duplicates().dropna()
    return data


def main():
    data = load_and_clean()

    X = data.iloc[:, :62]
    y = data["Formation energy"]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=20, shuffle=True
    )

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = RFR(n_estimators=59, max_depth=8, random_state=20)

    # 10-fold cross-validation on training set
    kf = KFold(n_splits=10, shuffle=True, random_state=20)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring="r2")
    print(f"CV R2  mean: {cv_scores.mean():.4f}  std: {cv_scores.std():.4f}")

    # Train on full training set and evaluate
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    print(f"Train  R2: {r2_score(y_train, y_train_pred):.4f}  "
          f"MAE: {MAE(y_train, y_train_pred):.4f}")
    print(f"Test   R2: {r2_score(y_test, y_test_pred):.4f}  "
          f"MAE: {MAE(y_test, y_test_pred):.4f}")

    # Save model and scaler
    joblib.dump(model, "../models/rf_model.joblib")
    joblib.dump(scaler, "../models/scaler.joblib")
    print("Model saved to ../models/rf_model.joblib")
    print("Scaler saved to ../models/scaler.joblib")


if __name__ == "__main__":
    main()
