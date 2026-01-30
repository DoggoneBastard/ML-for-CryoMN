# Step 5: Bayesian Optimization with Differential Evolution

## Overview

This module performs **proper Bayesian optimization** using Differential Evolution (DE) to maximize the Expected Improvement acquisition function. This provides better exploration-exploitation balance compared to random sampling.

## Usage

```bash
cd "/path/to/project"
python src/05_bo_optimization/bo_optimizer.py
```

## Input

- **Model**: `models/gp_model.pkl`
- **Data**: `data/processed/parsed_formulations.csv`

## Output

- `results/bo_candidates_general.csv` - Candidates with ≤5% DMSO
- `results/bo_candidates_dmso_free.csv` - DMSO-free candidates
- `*_summary.txt` - Human-readable summaries

## How It Works

### Algorithm

1. Load trained GP model
2. For each candidate to generate:
   - Run Differential Evolution to find `x* = argmax(EI(x))`
   - DE explores the entire search space globally
   - Constraint violations are penalized in the objective
3. Rank candidates by Expected Improvement value
4. Export with predictions and uncertainty estimates

### Expected Improvement (EI)

EI balances **exploration** and **exploitation**:

```
EI(x) = (μ(x) - y_best - ξ) · Φ(Z) + σ(x) · φ(Z)
```

Where:
- `μ(x)` = GP predicted mean
- `σ(x)` = GP predicted uncertainty
- `y_best` = best observed viability
- `ξ` = exploration parameter

High EI means: either high predicted viability OR high uncertainty (unexplored region).

### Constraints

| Constraint | Value |
|------------|-------|
| Max DMSO | 5% (general), 0.5% (DMSO-free) |
| Max viscous polymers | 2.0% (Hyaluronic Acid, Methylcellulose) |
| Max ingredients | 10 |

## Output Format

Results include both molar and percentage-based ingredients:

```csv
rank,expected_improvement,predicted_viability,uncertainty,dmso_M,trehalose_M,fbs_pct,...
1,0.806,63.5,25.0,0.0,0.3,20.0,...
```

**Formulation display** correctly shows units:
- Molar ingredients: `0.50M trehalose`
- Percentage ingredients: `20.0% FBS`

## Comparison: Random Sampling vs DE-based BO

| Aspect | `03_optimization` | `05_bo_optimization` |
|--------|-------------------|----------------------|
| **Search** | Random sampling | Differential Evolution |
| **Acquisition** | Sorts by mean only | Maximizes EI |
| **Exploration** | Pure exploitation | Balanced |
| **Speed** | Fast (~seconds) | Slower (~minutes) |
| **Quality** | May miss optima | Finds acquisition maxima |

## When to Use Which?

- **`03_optimization`**: Quick candidate generation, initial exploration, when speed matters
- **`05_bo_optimization`**: Serious optimization, when you want the most informative experiments

## Latest Results

| Category | Top EI | Predicted Viability |
|----------|--------|---------------------|
| General (≤5% DMSO) | 0.842 | 63.9% ± 25.1% |
| DMSO-free | 0.840 | 63.9% ± 25.1% |
