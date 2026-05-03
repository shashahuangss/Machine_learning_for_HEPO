import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer


class PhasePredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = None
        self.model = None
        self.feature_names = None
        self.X = None
        self.y = None
        self.X_scaled = None

    def prepare_data(self):
        """Load data from Excel, extract features/target, and apply standard scaling."""
        df = pd.read_excel(self.data_path)

        self.X = df.iloc[:, :36].values
        self.y = df.iloc[:, 36].values
        self.feature_names = df.columns[:36].tolist()

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        return self.X_scaled, self.y

    def cross_validate(self, X, y, params):
        """Run 10-fold stratified cross-validation and save per-fold predictions."""
        rf = RandomForestClassifier(**params, random_state=42)

        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='binary'),
            'recall': make_scorer(recall_score, average='binary'),
            'f1': make_scorer(f1_score, average='binary'),
            'roc_auc': 'roc_auc',
        }

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        cv_results = cross_validate(
            rf, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True,
        )

        # Aggregate metrics
        cv_metrics = {}
        for metric in scoring:
            values = cv_results[f'test_{metric}']
            cv_metrics[f'{metric}_mean'] = np.mean(values)
            cv_metrics[f'{metric}_std'] = np.std(values)

        # Collect per-fold predictions
        all_predictions = []
        folds = list(cv.split(X, y))
        for fold, estimator in enumerate(cv_results['estimator']):
            _, test_index = folds[fold]
            X_test = X[test_index]
            y_test = y[test_index]
            y_pred = estimator.predict(X_test)

            fold_df = pd.DataFrame({
                'Fold': [fold + 1] * len(y_test),
                'True Value': y_test,
                'Predicted Value': y_pred,
            })
            fold_features = pd.DataFrame(X_test, columns=self.feature_names)
            fold_df = pd.concat([fold_df, fold_features], axis=1)
            all_predictions.append(fold_df)

        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_df.to_excel('../results/cross_validation_predictions.xlsx', index=False)

        return cv_metrics

    def train(self, params=None):
        """Prepare data, cross-validate, train final model, and save artifacts."""
        X, y = self.prepare_data()

        if params is None:
            params = {
                'max_depth': None,
                'min_samples_leaf': 2,
                'min_samples_split': 5,
                'n_estimators': 20,
            }

        cv_metrics = self.cross_validate(X, y, params)

        # Train final model on full dataset
        self.model = RandomForestClassifier(**params, random_state=42)
        self.model.fit(X, y)
        self.save_model()

        print("\nCross-validation results:")
        for metric, value in cv_metrics.items():
            print(f"  {metric}: {value:.4f}")

        return cv_metrics

    def save_model(self, model_path='../models/random_forest_model.joblib', scaler_path='../models/standard_scaler.joblib'):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

    def load_model(self, model_path='../models/random_forest_model.joblib', scaler_path='../models/standard_scaler.joblib'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")

    def predict(self, new_data):
        """Scale new data and return predictions with probabilities."""
        if self.model is None or self.scaler is None:
            raise ValueError("Load or train a model first.")
        new_data_scaled = self.scaler.transform(new_data)
        predictions = self.model.predict(new_data_scaled)
        probabilities = self.model.predict_proba(new_data_scaled)[:, 1]
        return predictions, probabilities


def main():
    predictor = PhasePredictor('../data/phase_train_data.xlsx')
    params = {
        'max_depth': None,
        'min_samples_leaf': 2,
        'min_samples_split': 5,
        'n_estimators': 20,
    }
    results = predictor.train(params)

    print("\nFinal metrics:")
    for metric, value in results.items():
        print(f"  {metric}: {value}")


if __name__ == '__main__':
    main()
