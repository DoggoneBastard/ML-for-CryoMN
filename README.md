# CryoMN ML-Based Cryoprotective Solution Optimization

Machine learning pipeline for optimizing cryoprotective formulations for cryomicroneedle (CryoMN) technology.

## Goals

1. **Minimize DMSO usage** (reduce toxicity)
2. **Maximize cell viability** (maintain therapeutic efficacy)
3. **Limit ingredients** (≤10 components per formulation)

## Approach

**Bayesian Optimization with Gaussian Processes**
- Works well with limited data (~200 samples)
- Provides uncertainty quantification
- Supports iterative refinement with wet lab validation

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Parse formulation data
python src/01_data_parsing/parse_formulations.py

# 2. Train GP model
python src/02_model_training/train_gp_model.py

# 3. Generate optimized formulations
python src/03_optimization/optimize_formulation.py

# 4. Integrate wet lab results (after experiments)
python src/04_validation_loop/update_model.py
```

## Results

| Category | Best Candidate |
|----------|----------------|
| General (≤5% DMSO) | 78.6% viability, 0.5% DMSO |
| DMSO-free | 77.9% viability, 0% DMSO |

See `results/` for full candidate lists.

## Project Structure

```
├── data/
│   ├── raw/                    # Original literature data
│   ├── processed/              # Parsed formulations (200 rows)
│   └── validation/             # Wet lab results template
├── models/                     # Trained GP model
├── results/                    # Optimized candidate formulations
├── src/
│   ├── 01_data_parsing/        # Parse CSV, normalize units
│   ├── 02_model_training/      # Train GP regression model
│   ├── 03_optimization/        # Bayesian optimization
│   └── 04_validation_loop/     # Integrate wet lab feedback
└── requirements.txt
```

## Key Features

- **27 ingredients** tracked (DMSO, trehalose, glycerol, etc.)
- **Synonym merging** (e.g., FBS = FCS = fetal bovine serum)
- **Unit normalization** (all concentrations converted to molar)
- **Uncertainty quantification** (GP provides confidence intervals)
- **Iterative refinement** (model improves with each wet lab validation)
