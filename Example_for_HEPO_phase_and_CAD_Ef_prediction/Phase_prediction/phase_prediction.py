import pandas as pd
import joblib
import os


class PhasePredictor:
    """Phase prediction using trained Random Forest model."""

    def __init__(self, model_path="../../Phase_prediction/models/random_forest_model.joblib",
                 scaler_path="../../Phase_prediction/models/standard_scaler.joblib"):
        """Load trained model and scaler."""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, input_file):
        """Perform phase prediction on input features."""
        df = pd.read_excel(input_file)

        # Extract features (first 36 columns)
        X = df.iloc[:, :36].values
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        results = df.copy()
        results["Predicted_Phase"] = predictions
        results["Phase_Probability"] = probabilities

        return results

    def save_predictions(self, results, output_file="results/pyrochlore_phase_predictions.xlsx"):
        """Save prediction results."""
        results.to_excel(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        print(f"Total samples: {len(results)}")
        print(f"Phase distribution:\n{results['Predicted_Phase'].value_counts(normalize=True)}")


def filter_high_probability(input_file="results/pyrochlore_phase_predictions.xlsx",
                            threshold=0.95,
                            ef_antisite_dir="../Ef_antisite/data"):
    """
    Filter samples with Phase_Probability > threshold.
    Save full filtered data and first 18 columns (compositions only) to Ef_antisite/data.
    """
    df = pd.read_excel(input_file)

    # Filter by probability threshold
    filtered = df[df["Phase_Probability"] > threshold]
    print(f"\nFiltered {len(filtered)} samples with Phase_Probability > {threshold}")

    # Save full filtered data
    draft_file = os.path.join(ef_antisite_dir, "pyrochlore_phase_probability_0.95_draft.xlsx")
    filtered.to_excel(draft_file, index=False)
    print(f"Full filtered data saved to {draft_file}")

    # Save first 18 columns only (compositions for antisite pair generation)
    compositions = filtered.iloc[:, :18]
    comp_file = os.path.join(ef_antisite_dir, "pyrochlore_phase_probability_0.95.xlsx")
    compositions.to_excel(comp_file, index=False)
    print(f"Compositions (first 18 columns) saved to {comp_file}")


def main():
    predictor = PhasePredictor()

    # Step 1: Predict phase
    input_file = "data/random_pyrochlore_features_phase.xlsx"
    predictions = predictor.predict(input_file)
    predictor.save_predictions(predictions)

    # Step 2: Filter high-probability phases and export to Ef_antisite
    filter_high_probability()


if __name__ == "__main__":
    main()
