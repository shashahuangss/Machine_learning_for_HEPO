# Example: HEPO Phase and Cation Antisite Formation Energy Prediction

This example demonstrates how to use trained models to screen high-entropy pyrochlore oxides (HEPO) for phase stability and predict cation antisite defect formation energy.

## Project Structure

```
Example_for_HEPO_phase_and_CAD_Ef_prediction/
├── Phase_prediction/
│   ├── data/
│   │   └── random_pyrochlore_features_phase.xlsx       # Generated random compositions (36 features)
│   ├── results/
│   │   └── pyrochlore_phase_predictions.xlsx           # Phase predictions with probabilities
│   ├── generate_phase_predict.py                       # Generate random A/B site compositions + derived features
│   └── phase_prediction.py                             # Phase prediction + filter high-probability phases
├── Ef_antisite/
│   ├── data/
│   │   ├── pyrochlore_phase_probability_0.95_draft.xlsx       # Filtered phases (full data)
│   │   ├── pyrochlore_phase_probability_0.95.xlsx             # Filtered phases (first 18 cols)
│   │   └── pyrochlore_antisite_predictions_feature.xlsx        # Generated antisite pairs (36 cols)
│   ├── results/
│   │   ├── pyrochlore_predictions_with_antisite_energy.xlsx   # Formation energy predictions for each antisite (62 features + Ef)
│   │   └── pyrochlore_aggregated_Ef_antisite_results.xlsx     # Average cation antisite formation energy per composition
│   ├── generate_anti_pair.py                           # Generate A-B antisite pair combinations
│   ├── predict_Ef.py                                   # Compute features + predict formation energy
│   └── calculate_mean.py                               # Aggregate results by composition
└── README.md
```

## Dependencies

This example uses trained models from:
- `Phase_prediction/models/` — Phase classification model (Random Forest)
- `Cation_antisite_formation_energy_prediction/models/` — Formation energy regression model (Random Forest)

## Usage

### Stage 1: Phase Prediction

```bash
cd Example_for_HEPO_phase_and_CAD_Ef_prediction/Phase_prediction
```

**Step 1:** Generate random A/B site compositions with all 36 derived physical features:
```bash
python generate_phase_predict.py
```
Output: `data/random_pyrochlore_features_phase.xlsx`

**Step 2:** Predict phase stability and filter high-probability (>0.95) phases:
```bash
python phase_prediction.py
```
Output:
- `results/pyrochlore_phase_predictions.xlsx` — All predictions
- `../Ef_antisite/data/pyrochlore_phase_probability_0.95_draft.xlsx` — Filtered (full columns)
- `../Ef_antisite/data/pyrochlore_phase_probability_0.95.xlsx` — Filtered (first 18 columns only)

### Stage 2: Antisite Formation Energy Prediction

```bash
cd Example_for_HEPO_phase_and_CAD_Ef_prediction/Ef_antisite
```

**Step 3:** Generate all A-B antisite pair combinations:
```bash
python generate_anti_pair.py
```
Input: `data/pyrochlore_phase_probability_0.95.xlsx` (18 cols)
Output: `data/pyrochlore_antisite_predictions_feature.xlsx` (36 cols: 18 composition + 18 antisite markers)

**Step 4:** Compute physical features and predict formation energy:
```bash
python predict_Ef.py
```
Input: `data/pyrochlore_antisite_predictions_feature.xlsx`
Output: `results/pyrochlore_predictions_with_antisite_energy.xlsx` (62 features + predicted Ef)

**Step 5:** Aggregate results by composition:
```bash
python calculate_mean.py
```
Input: `results/pyrochlore_predictions_with_antisite_energy.xlsx`
Output: `results/pyrochlore_aggregated_Ef_antisite_results.xlsx`

## Workflow Diagram

```
generate_phase_predict.py
    → random_pyrochlore_features_phase.xlsx (10,000 samples × 36 features)
        │
phase_prediction.py
    → pyrochlore_phase_predictions.xlsx
    → pyrochlore_phase_probability_0.95.xlsx (filtered, 18 cols → Ef_antisite/data/)
        │
generate_anti_pair.py
    → pyrochlore_antisite_predictions_feature.xlsx (18 composition + 18 antisite markers)
        │
predict_Ef.py  (auto-computes all 62 features from composition + antisite markers)
    → pyrochlore_predictions_with_antisite_energy.xlsx (62 features + Ef for each antisite pair)
        │
calculate_mean.py
    → pyrochlore_aggregated_Ef_antisite_results.xlsx (final aggregated results)
```

## Notes

- `predict_Ef.py` automatically computes all physical features (radius, mass, VEC, electronegativity, melting point) from element compositions and antisite markers.
- `calculate_mean.py` groups by the first 18 columns (composition) and computes the average cation antisite formation energy across all antisite pairs for each composition.
