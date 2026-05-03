# Machine Learning for High-Entropy Pyrochlore Oxides

This repository contains machine learning models and workflows for predicting phase stability and cation antisite defect formation energy in high-entropy pyrochlore oxides (HEPO).

## Repository Structure

```
├── Phase_prediction/                              # Phase classification model
├── Cation_antisite_formation_energy_prediction/   # Formation energy regression model
└── Example_for_HEPO_phase_and_CAD_Ef_prediction/  # Application example
```

### Phase_prediction

Random Forest classifier for predicting pyrochlore phase stability from composition and derived physical features (36 features → binary classification).

- Multi-classifier comparison (8 models)
- Model training with 10-fold stratified cross-validation
- SHAP interpretability analysis


### Cation_antisite_formation_energy_prediction

Random Forest regressor for predicting cation antisite defect formation energy from composition, physical features, and antisite pair descriptors (62 features → regression).

- Multi-model comparison (10 models)
- Model training with 10-fold cross-validation
- Pearson correlation analysis
- SHAP interpretability analysis

### Example_for_HEPO_phase_and_CAD_Ef_prediction

End-to-end application example demonstrating the full screening workflow:

1. Generate random HEPO compositions
2. Predict phase stability and filter high-probability pyrochlore phases
3. Generate cation antisite pair combinations
4. Predict formation energy for each antisite pair
5. Aggregate results by composition

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

## Usage

Each subdirectory contains its own `README.md` with detailed instructions. All scripts should be run from their respective directories.

