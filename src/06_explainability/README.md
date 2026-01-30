# Step 6: Model Explainability

## Overview

This module generates comprehensive visualizations to explain how the Gaussian Process model makes predictions for cryoprotective formulations. Understanding model behavior helps guide experimental design and builds trust in the optimization recommendations.

## Usage

```bash
cd "/path/to/project"
python src/06_explainability/explainability.py
```

## Input

- **Model**: `models/gp_model.pkl`
- **Scaler**: `models/scaler.pkl`
- **Metadata**: `models/model_metadata.json`
- **Feature Importance**: `models/feature_importance.csv`
- **Data**: `data/processed/parsed_formulations.csv`

## Output

All visualizations are saved to `results/explainability/`:

| File | Description |
|------|-------------|
| `feature_importance.png` | Horizontal bar chart of permutation-based feature importance |
| `shap_summary.png` | SHAP beeswarm plot showing individual feature impacts |
| `shap_importance.png` | SHAP-based feature importance ranking |
| `partial_dependence_plots.png` | PDPs showing how viability changes with each ingredient |
| `interaction_contours.png` | 2D contour plots of ingredient pair interactions |
| `acquisition_landscape.png` | EI acquisition function visualization (mean, uncertainty, EI) |
| `uncertainty_analysis.png` | GP uncertainty calibration and distribution analysis |

## Visualization Details

### 1. Feature Importance Bar Chart
Shows which ingredients have the strongest influence on cell viability predictions, based on permutation importance from model training.

**Note**: Feature names are cleaned for display (both `_M` and `_pct` suffixes are removed).

### 2. SHAP Analysis
Uses [SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/) to explain individual predictions:
- **Summary plot**: Shows direction and magnitude of feature effects
- **Importance plot**: Ranks features by mean absolute SHAP value

> **Note**: Requires `shap>=0.42.0`. If not installed, this analysis is skipped.

### 3. Partial Dependence Plots (PDPs)
Shows the marginal effect of each ingredient on predicted viability:
- X-axis: Ingredient concentration (M or %)
- Y-axis: Predicted viability (%)
- Blue line: Mean prediction
- Shaded area: 95% confidence interval from GP uncertainty

### 4. 2D Interaction Contours
Visualizes how pairs of top ingredients interact:
- Color gradient: Predicted viability
- Contour lines: Viability iso-lines
- Helps identify synergistic or antagonistic ingredient combinations

### 5. Acquisition Function Landscape
Three-panel visualization for Bayesian optimization insight:
1. **GP Mean**: Where the model predicts high viability
2. **GP Uncertainty**: Where the model is uncertain (exploration opportunity)
3. **Expected Improvement**: Combined exploration-exploitation score

Red star marks the best observed formulation.

### 6. Uncertainty Analysis
Four-panel analysis of model confidence:
1. **Predicted vs Actual**: Scatter plot colored by uncertainty
2. **Uncertainty Distribution**: Histogram of prediction uncertainties
3. **Error vs Uncertainty**: Calibration check (should correlate positively)
4. **Uncertainty by Viability Range**: Where is the model most/least confident?

## Dependencies

```
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.42.0  # Optional but recommended
scipy>=1.10.0
scikit-learn>=1.3.0
```

## Example Output

After running, you'll see:

```
================================================================================
CryoMN Model Explainability Analysis
================================================================================

📊 Loading model and data...
  Model loaded with 21 features
  Data loaded with 191 formulations

📈 Generating visualizations...

1️⃣ Feature Importance Bar Chart
  ✓ Feature importance chart saved: results/explainability/feature_importance.png

2️⃣ SHAP Values Analysis
  ✓ SHAP summary plot saved: results/explainability/shap_summary.png
  ✓ SHAP importance plot saved: results/explainability/shap_importance.png

...

================================================================================
✅ Explainability Analysis Complete!
================================================================================
```

## Interpretation Guide

### High Importance Features
- **DMSO**: Highest importance (0.29) - key cryoprotectant but toxic at high concentrations
- **HES, Trehalose, Sucrose**: Important sugars for cell membrane protection
- **Glycerol**: Classic CPA with high importance
- **FBS**: Percentage-based serum with protective effects

### Reading the PDPs
- Upward slope: Ingredient increases viability
- Downward slope: Ingredient decreases viability (or becomes toxic)
- Wide confidence interval: High uncertainty in that concentration range

### Using Acquisition Landscape
- High EI regions: Most informative next experiments
- High uncertainty + moderate mean: Exploration opportunities
- High mean + low uncertainty: Exploitation (known good regions)

## Feature Name Handling

The module automatically cleans feature names for display:
- `dmso_M` → `Dmso`
- `fbs_pct` → `Fbs`
- `hyaluronic_acid_pct` → `Hyaluronic Acid`

Both `_M` (molar) and `_pct` (percentage) suffixes are stripped for readability.
