# Phase Prediction

A machine-learning pipeline for predicting phase classification using a Random Forest classifier with SHAP-based model interpretability.

## Project Structure

```
Phase_prediction/
├── data/
│   └── phase_train_data.xlsx        # Training dataset (36 features + 1 label)
├── models/
│   ├── random_forest_model.joblib   # Trained Random Forest classifier
│   └── standard_scaler.joblib       # Fitted StandardScaler
├── results/
│   ├── cross_validation_predictions.xlsx   # Per-fold CV predictions
│   ├── SHAP_values.xlsx                    # SHAP values for each sample
│   ├── SHAP_summary_plot.png               # SHAP summary plot (top 10 features)
│   └── model_comparison_accuracy.xlsx      # Classifier comparison metrics
├── scripts/
│   ├── phase_train.py               # Model training & cross-validation
│   ├── phase_shap.py                # SHAP interpretability analysis
│   └── all_model_comparison.py      # Multi-classifier comparison
└── README.md
```

## Usage

All scripts should be run from the `scripts/` directory.

### 1. Classifier model comparison

```bash
cd Phase_prediction/scripts
python all_model_comparison.py
```

This will:
- Compare 8 classifiers: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, Naive Bayes, SVM, Neural Network
- Evaluate each classifier on train/test sets (Accuracy, Precision, Recall, F1, ROC AUC)
- Save comparison results to `results/model_comparison_accuracy.xlsx`

### 2. Train the model using Random Forest classifier

```bash
cd Phase_prediction/scripts
python phase_train.py
```

This will:
- Load data from `data/phase_train_data.xlsx`
- Run 10-fold stratified cross-validation (accuracy, precision, recall, F1, ROC AUC)
- Train the final model on the full dataset
- Save the model and scaler to `models/`
- Save per-fold predictions to `results/`

### 3. Run SHAP analysis

```bash
cd Phase_prediction/scripts
python phase_shap.py
```

This will:
- Load the trained model and scaler from `models/`
- Compute SHAP values using `TreeExplainer`
- Export SHAP values to `results/SHAP_values.xlsx`
- Generate a summary plot saved to `results/SHAP_summary_plot.png`

## Dependencies

- numpy
- pandas
- scikit-learn
- joblib
- matplotlib
- shap
