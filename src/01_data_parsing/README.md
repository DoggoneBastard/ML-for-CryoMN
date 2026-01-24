# Step 1: Data Parsing

## Overview

This module parses cryoprotective solution formulation data from literature-derived CSV files. It extracts ingredients, normalizes concentrations to molar units, and handles ingredient synonym merging.

## Usage

```bash
cd "/Users/doggonebastard/Antigravity/ML for CryoMN"
python src/01_data_parsing/parse_formulations.py
```

## Input

- **File**: `data/raw/Cryopreservative Data 2026.csv`
- **Format**: CSV with columns:
  1. `All ingredients in cryoprotective solution` - Free-text formulation
  2. `DMSO usage` - DMSO percentage
  3. `Cooling rate` - Freezing protocol
  4. `Viability` - Cell viability post-thaw

## Output

- **File**: `data/processed/parsed_formulations.csv`
- **Format**: Structured CSV with columns:
  - `formulation_id` - Unique identifier
  - `viability_percent` - Extracted viability value
  - `dmso_percent` - DMSO percentage
  - `source_doi` - Literature source
  - `{ingredient}_M` - Molar concentration for each ingredient

## Features

### Ingredient Synonym Mapping

The parser merges equivalent ingredient names:

| Canonical Name | Synonyms |
|----------------|----------|
| `dmso` | DMSO, Me2SO, dimethyl sulfoxide |
| `ethylene_glycol` | EG, ethylene glycol |
| `propylene_glycol` | 1,2-propanediol, PROH, PG |
| `fbs` | FBS, FCS, fetal bovine serum |
| `hsa` | HSA, human albumin, BSA |
| `hes` | HES, hydroxyethyl starch, HES450 |

### Unit Conversion

Concentrations are converted to molar (M):
- `%` → M (using molecular weight and density)
- `mM` → M (÷ 1000)
- `mg/mL` → M (using molecular weight)

### Culture Media Exclusion

The following are excluded as separate variables:
- DMEM, α-MEM, PBS, HBSS, saline
- Culture media supplements

### Duplicate Detection

The script identifies identical formulations with different viabilities and prompts for user resolution.

## Example Output

```csv
formulation_id,viability_percent,dmso_percent,source_doi,dmso_M,trehalose_M,glycerol_M,...
1,82.5,10.0,10.1038/srep09596,1.409,0.0,0.0,...
2,77.0,0.0,10.1002/term.2175,0.0,0.3,0.2,...
```
