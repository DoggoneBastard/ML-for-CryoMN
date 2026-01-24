# Step 2: Model Training

## Overview

This module trains a Gaussian Process (GP) regression model to predict cell viability from cryoprotective formulation ingredients.

## Usage

```bash
cd "/Users/doggonebastard/Antigravity/ML for CryoMN"
python src/02_model_training/train_gp_model.py
```

## Input

- **File**: `data/processed/parsed_formulations.csv`
- **Features**: Ingredient concentrations (molar)
- **Target**: `viability_percent`

## Output

- **Model**: `models/gp_model.pkl`
- **Scaler**: `models/scaler.pkl`  
- **Metadata**: `models/model_metadata.json`
- **Feature Importance**: `models/feature_importance.csv`

## Model Details

### Kernel

Matérn kernel (ν=2.5) with:
- Constant kernel (amplitude scaling)
- White kernel (noise modeling)

### Training

- 5-fold cross-validation for hyperparameter tuning
- StandardScaler for feature normalization
- Target normalization enabled

### Metrics

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **Uncertainty**: Mean prediction standard deviation

## Programmatic Usage

```python
from src.02_model_training.train_gp_model import load_model

# Load trained model
gp, scaler, metadata = load_model('models/')

# Make predictions
X_new = ...  # New formulation features
X_scaled = scaler.transform(X_new)
y_pred, y_std = gp.predict(X_scaled, return_std=True)

print(f"Predicted viability: {y_pred[0]:.1f}% ± {y_std[0]:.1f}%")
```
