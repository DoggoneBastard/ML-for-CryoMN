# Step 5: Bayesian Optimization with Differential Evolution

## Overview

This module performs **proper Bayesian optimization** using Differential Evolution (DE) to maximize the Expected Improvement acquisition function. It uses **batch-mode BO with local penalization** to generate diverse candidates, preventing convergence to the same optimum.

## Usage

```bash
cd "/path/to/project"
python src/05_bo_optimization/bo_optimizer.py
```

## Input

- **Model**: `models/composite_model.pkl` (preferred) or `models/gp_model.pkl` (fallback)
- **Data**: `data/processed/parsed_formulations.csv`

The script auto-detects which model to use and prints the selection:
- `>>> Using COMPOSITE model (literature prior + wet lab correction)` — after running `04_validation_loop`
- `>>> Using STANDARD GP model (literature-only)` — before any validation data is added

## Output

- `results/bo_candidates_general.csv` - Candidates with ≤5% DMSO
- `results/bo_candidates_dmso_free.csv` - DMSO-free candidates
- `*_summary.txt` - Human-readable summaries

## How It Works

### Algorithm

1. Load trained model (composite if available, otherwise standard GP)
2. Compute `y_best` from model predictions on observed data
3. For each candidate (sequentially):
   - Run Differential Evolution to find `x* = argmax(EI(x) - penalty(x))`
   - DE explores the entire search space globally
   - **Batch diversity**: Gaussian penalty repels DE away from previously found candidates
   - Constraint violations (DMSO, ingredient count) are penalized
4. Recalculate pure EI (without penalty) for accurate reporting
5. Rank candidates by predicted viability
6. Export with predictions and uncertainty estimates

### Batch Diversity (Local Penalization)

To prevent all candidates from converging to the same optimum, each DE run adds a Gaussian repulsion centered on previously found candidates:

```
penalty(x) = Σ_i  strength · exp(-0.5 · ||x - x_i||² / r²)
```

Where `strength` and `r` (radius) control how strongly candidates repel each other. This ensures each new candidate explores a different region of formulation space.

### Expected Improvement (EI)

EI balances **exploration** and **exploitation**:

```
EI(x) = (μ(x) - y_best - ξ) · Φ(Z) + σ(x) · φ(Z)
```

Where:
- `μ(x)` = GP predicted mean
- `σ(x)` = GP predicted uncertainty
- `y_best` = best model-predicted viability (composite) or observed (standard)
- `ξ` = exploration parameter

High EI means: either high predicted viability OR high uncertainty (unexplored region).

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_ingredients` | 10 | Max non-zero ingredients per formulation |
| `max_dmso_percent` | 5.0 | Max DMSO (general), 0.5% (DMSO-free) |
| `n_candidates` | 20 | Number of diverse candidates to generate |
| `xi` | 0.01 | EI exploration parameter |
| `de_maxiter` | 100 | DE iterations per candidate |
| `de_popsize` | 15 | DE population size |
| `diversity_penalty` | 5.0 | Strength of batch diversity repulsion |
| `diversity_radius` | 0.3 | Fraction of feature range for penalty radius |

## Comparison: Random Sampling vs DE-based BO

| Aspect | `03_optimization` | `05_bo_optimization` |
|--------|-------------------|----------------------|
| **Search** | Random sampling | Differential Evolution |
| **Acquisition** | Sorts by mean only | Maximizes EI |
| **Exploration** | Pure exploitation | Balanced |
| **Diversity** | Naturally diverse (random) | Batch-mode penalization |
| **Speed** | Fast (~seconds) | Slower (~minutes) |
| **Quality** | May miss optima | Finds acquisition maxima |

## When to Use Which?

- **`03_optimization`**: Quick candidate generation, initial exploration, when speed matters
- **`05_bo_optimization`**: Serious optimization, when you want the most diverse and informative experiments
