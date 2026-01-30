# Step 3: Candidate Generation via Random Sampling

## Overview

This module generates candidate cryoprotective formulations using **random sampling + GP prediction**. It provides a fast way to explore the formulation space, though it uses pure exploitation (no exploration-exploitation balance).

> **Note**: For proper Bayesian optimization with acquisition-guided search, see [`05_bo_optimization`](../05_bo_optimization/README.md).

## Usage

```bash
cd "/path/to/project"
python src/03_optimization/optimize_formulation.py
```

## Input

- **Model**: `models/gp_model.pkl`
- **Data**: `data/processed/parsed_formulations.csv`

## Output

- `results/candidates_general.csv` - Candidates with ≤5% DMSO
- `results/candidates_dmso_free.csv` - DMSO-free candidates
- `*_summary.txt` - Human-readable summaries

## Algorithm

1. Load trained GP model
2. Generate large pool of random formulations (50× target count)
3. Filter by constraints (max DMSO, max ingredients)
4. Use GP to predict viability for each candidate
5. Rank by predicted viability (highest mean)
6. Select top-N candidates

### Constraints

| Constraint | Value |
|------------|-------|
| Max DMSO | 5% (general), 0.5% (DMSO-free) |
| Max ingredients | 10 |
| Min viability | 70% (target) |

## Comparison with Proper BO

| Aspect | This Module (03) | Proper BO (05) |
|--------|------------------|----------------|
| **Method** | Random sampling | Differential Evolution |
| **Selection** | Highest predicted mean | Highest Expected Improvement |
| **Exploration** | None (pure exploitation) | Balanced via uncertainty |
| **Speed** | Fast (~seconds) | Slower (~minutes) |
| **Best for** | Quick generation | Most informative experiments |

### Why the difference matters

- **This module** always suggests what the model thinks will work best *right now*
- **Proper BO** suggests what would be most *informative* to test, including uncertain regions that might reveal better formulations

## Output Format

The output CSV includes both molar and percentage-based features:

```csv
rank,predicted_viability,uncertainty,dmso_percent,n_ingredients,dmso_M,trehalose_M,fbs_pct,hsa_pct,...
1,85.2,12.3,0.0,5,0.0,0.5,20.0,0.0,...
```

**Column naming convention:**
- `{ingredient}_M` - Molar concentration
- `{ingredient}_pct` - Percentage concentration

## Programmatic Usage

```python
from src.03_optimization.optimize_formulation import FormulationOptimizer, OptimizationConfig

config = OptimizationConfig(max_dmso_percent=0.0, n_candidates=50)
optimizer = FormulationOptimizer(gp, scaler, feature_names, config)
candidates = optimizer.generate_low_dmso_candidates(X, y)
```

## Latest Results

| Category | Top Candidate | Predicted Viability |
|----------|---------------|---------------------|
| General (≤5% DMSO) | 3-ingredient DMSO-free | 72.3% ± 23.5% |
| DMSO-free | 2-ingredient formulation | 81.6% ± 20.6% |
