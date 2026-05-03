# Cation Antisite Formation Energy Prediction

A machine-learning pipeline for predicting cation antisite formation energy using a Random Forest regressor, with Pearson correlation analysis and SHAP-based model interpretability.

## Project Structure

```
Cation_antisite_formation_energy_prediction/
├── data/
│   └── Ef_train_data.xlsx              # Training dataset (62 features + Formation energy)
├── models/
│   ├── rf_model.joblib                 # Trained Random Forest regressor
│   └── scaler.joblib                   # Fitted StandardScaler
├── results/
│   ├── pearson_feature_vs_formation_energy.xlsx       # Feature-target Pearson correlations
│   ├── pearson_feature_correlation_matrix.xlsx        # Feature-feature correlation matrix
│   ├── pearson_feature_correlation_matrix_heatmap.png # Full correlation heatmap
│   ├── pearson_top_features_correlation_heatmap.png   # Top 15 features correlation heatmap
│   ├── SHAP_values.xlsx                               # SHAP values for each sample
│   ├── SHAP_summary_plot.png                          # SHAP summary plot (top 10 features)
│   └── model_comparison_accuracy.xlsx                 # Model comparison metrics
├── scripts/
│   ├── ef_train.py                     # Model training & cross-validation
│   ├── ef_pearson.py                   # Pearson correlation analysis
│   ├── ef_shap.py                      # SHAP interpretability analysis
│   └── all_model_comparison.py         # Multi-model comparison
└── README.md
```

## Usage

All scripts should be run from the `scripts/` directory.

### 1. Model comparison

```bash
cd Cation_antisite_formation_energy_prediction/scripts
python all_model_comparison.py
```

This will:
- Compare 10 regression models: LR, LASSO, Ridge, KRR, KNN, SVM, RF, GBoost, XGBoost, ANN
- Evaluate each model on train/test sets (R², MAE)
- Save comparison results to `results/model_comparison_accuracy.xlsx`

### 2. Train the model with Random Forest regressor 

```bash
cd Cation_antisite_formation_energy_prediction/scripts
python ef_train.py
```

This will:
- Load data from `data/Ef_train_data.xlsx`
- Run 10-fold cross-validation (R², MAE)
- Train the final model on 80% training set
- Save the model and scaler to `models/`

### 3. Pearson correlation analysis

```bash
cd Cation_antisite_formation_energy_prediction/scripts
python ef_pearson.py
```

This will:
- Compute Pearson correlation between each feature and Formation energy
- Compute feature-feature correlation matrix
- Generate correlation heatmaps
- Save all results to `results/`

### 4. SHAP analysis

```bash
cd Cation_antisite_formation_energy_prediction/scripts
python ef_shap.py
```

This will:
- Load the trained model and scaler from `models/`
- Compute SHAP values using `TreeExplainer`
- Export SHAP values to `results/SHAP_values.xlsx`
- Generate a summary plot saved to `results/SHAP_summary_plot.png`

## Dependencies

- numpy
- pandas
- scipy
- scikit-learn
- joblib
- matplotlib
- seaborn
- shap
- xgboost
