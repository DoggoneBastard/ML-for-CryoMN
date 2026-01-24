# Step 4: Validation Loop

## Overview

This module integrates wet lab validation results to iteratively refine the GP model.

## Workflow

```
┌─────────────────┐
│  Train Model    │  ← Literature data
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Optimize      │  → Candidate formulations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Wet Lab       │  → Validation results
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Update Model   │  ← Combined data
└────────┬────────┘
         │
         └──────────→ (Repeat)
```

## Usage

### First Time

```bash
cd "/Users/doggonebastard/Antigravity/ML for CryoMN"
python src/04_validation_loop/update_model.py
```

This creates a validation template at `data/validation/validation_template.csv`.

### After Wet Lab Experiments

1. Copy template to `data/validation/validation_results.csv`
2. Fill in experimental viability values
3. Run the script again

## Validation CSV Format

```csv
experiment_id,experiment_date,viability_measured,notes,dmso,trehalose,glycerol,...
EXP001,2026-01-25,85.5,"Test batch 1",0.0,0.3,0.5,...
EXP002,2026-01-26,72.3,"Repeat with higher trehalose",0.0,0.5,0.5,...
```

## Output

- Updated model in `models/iteration_N/`
- Main model updated in `models/`
- Iteration history in `data/validation/iteration_history.json`

## Iteration Tracking

Each iteration is logged with:
- Timestamp
- Number of validation samples
- Validation RMSE
- Model performance metrics
