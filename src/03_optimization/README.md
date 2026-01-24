# Step 3: Bayesian Optimization

## Overview

This module performs multi-objective Bayesian optimization to discover optimal cryoprotective formulations with:
1. **Minimized DMSO usage** (≤5% or DMSO-free)
2. **Maximized viability** (target ≥70%)
3. **Limited ingredients** (≤10 components)

## Usage

```bash
cd "/Users/doggonebastard/Antigravity/ML for CryoMN"
python src/03_optimization/optimize_formulation.py
```

## Input

- **Model**: `models/gp_model.pkl`
- **Data**: `data/processed/parsed_formulations.csv`

## Output

- `results/candidates_general.csv` - Candidates with ≤5% DMSO
- `results/candidates_dmso_free.csv` - DMSO-free candidates
- `*_summary.txt` - Human-readable summaries

## Optimization Approach

### Acquisition Function

**Expected Improvement (EI)** - Balances exploration and exploitation:
- Explores uncertain regions (high variance)
- Exploits promising regions (high mean)

### Constraints

| Constraint | Value |
|------------|-------|
| Max DMSO | 5% (general), 0.5% (DMSO-free) |
| Max ingredients | 10 |
| Min viability | 70% (target) |

### Algorithm

1. Load trained GP model
2. Initialize optimizer with constraints
3. Use Differential Evolution for global search
4. Apply Expected Improvement acquisition
5. Generate top-N candidate formulations
6. Export results with uncertainty estimates

## Output Format

```csv
rank,predicted_viability,uncertainty,dmso_percent,n_ingredients,dmso_M,trehalose_M,...
1,85.2,12.3,0.0,5,0.0,0.5,...
```

## Programmatic Usage

```python
from src.03_optimization.optimize_formulation import FormulationOptimizer, OptimizationConfig

config = OptimizationConfig(max_dmso_percent=0.0, n_candidates=50)
optimizer = FormulationOptimizer(gp, scaler, feature_names, config)
candidates = optimizer.generate_low_dmso_candidates(X, y)
```
