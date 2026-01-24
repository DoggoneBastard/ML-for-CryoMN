#!/usr/bin/env python3
"""
CryoMN Formulation Data Parser

Parses cryoprotective solution formulation data from literature-derived CSV.
Extracts ingredients, normalizes concentrations to molar units, and handles
ingredient synonym merging.

Author: CryoMN ML Project
Date: 2026-01-24
"""

import pandas as pd
import numpy as np
import re
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# =============================================================================
# INGREDIENT SYNONYM MAPPINGS
# =============================================================================

INGREDIENT_SYNONYMS = {
    # Permeating CPAs
    'propylene_glycol': ['1,2-propanediol', 'propylene glycol', 'proh', 'pg', '1,2-propane diol'],
    'dmso': ['dmso', 'me2so', 'dimethyl sulfoxide', 'dimethylsulfoxide'],
    'ethylene_glycol': ['eg', 'ethylene glycol', 'ethyleneglycol'],
    'glycerol': ['glycerol', 'glycerin'],
    
    # Sugars and sugar alcohols
    'trehalose': ['trehalose'],
    'sucrose': ['sucrose'],
    'glucose': ['glucose', 'dextrose'],
    'mannitol': ['mannitol'],
    'maltose': ['maltose'],
    'raffinose': ['raffinose'],
    
    # Polymers
    'peg': ['peg', 'polyethylene glycol'],
    'pvp': ['pvp', 'polyvinylpyrrolidone'],
    'pva': ['pva', 'polyvinyl alcohol'],
    'hes': ['hes', 'hydroxyethyl starch', 'hes450', 'hydroxychyl starch', 'hydroxyethylstarch'],
    'dextran': ['dextran', 'dextran-40', 'dextran40'],
    'ficoll': ['ficoll', 'ficoll 70', 'ficoll70'],
    'pentaisomaltose': ['pentaisomaltose'],
    
    # Proteins and sera
    'fbs': ['fbs', 'fcs', 'fetal bovine serum', 'fetal calf serum'],
    'human_serum': ['hs', 'human serum'],
    'hsa': ['hsa', 'human albumin', 'human serum albumin', 'has', 'albumin', 'bsa', 'bovine serum albumin'],
    
    # Amino acids and compatible solutes
    'proline': ['proline', 'l-proline'],
    'ectoin': ['ectoin', 'ectoine'],
    'isoleucine': ['isoleucine', 'l-isoleucine'],
    'taurine': ['taurine'],
    'betaine': ['betaine'],
    
    # Other additives
    'methylcellulose': ['mc', 'methylcellulose'],
    'hyaluronic_acid': ['ha', 'hmw-ha', 'hyaluronic acid'],
    'creatine': ['creatine'],
    'acetamide': ['acetamide'],
    'sericin': ['sericin'],
    
    # Polyampholytes
    'cooh_pll': ['cooh-pll', 'polyampholyte'],
    'peg_pa': ['peg-pa', 'peg−pa'],
}

# Build reverse lookup
SYNONYM_TO_CANONICAL = {}
for canonical, synonyms in INGREDIENT_SYNONYMS.items():
    for syn in synonyms:
        SYNONYM_TO_CANONICAL[syn.lower()] = canonical

# =============================================================================
# MOLECULAR WEIGHTS (g/mol) for unit conversion
# =============================================================================

MOLECULAR_WEIGHTS = {
    'dmso': 78.13,
    'ethylene_glycol': 62.07,
    'propylene_glycol': 76.09,
    'glycerol': 92.09,
    'trehalose': 342.3,
    'sucrose': 342.3,
    'glucose': 180.16,
    'mannitol': 182.17,
    'maltose': 342.3,
    'raffinose': 504.42,
    'proline': 115.13,
    'ectoin': 142.16,
    'isoleucine': 131.17,
    'taurine': 125.15,
    'betaine': 117.15,
    'creatine': 131.13,
    'acetamide': 59.07,
}

# Density (g/mL) for % v/v conversions
DENSITIES = {
    'dmso': 1.10,
    'ethylene_glycol': 1.11,
    'propylene_glycol': 1.036,
    'glycerol': 1.26,
}

# =============================================================================
# CULTURE MEDIA TO EXCLUDE (not counted as separate ingredients)
# =============================================================================

CULTURE_MEDIA = [
    'dmem', 'dmem/f12', 'dmem/ham', "dulbecco's modified eagle medium",
    'α-mem', 'a-mem', 'alpha-mem', 'mem', 'minimum essential medium',
    'pbs', 'phosphate buffer saline', 'phosphate buffered saline',
    'dpbs', 'hbss', 'saline', 'nacl', '0.9% nacl',
    'mesencult', 'l-15', 'l15', 'medium 199', 'knockout dmem',
    'plasmalyte', 'plasmalyte a', 'lactated ringer', 'complete culture medium',
    'culture medium', 'basal medium', 'electroporation buffer',
    'e8 medium', 'cssc medium',
]


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def extract_concentration(text: str, ingredient: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract concentration value and unit for an ingredient from text.
    
    Returns:
        Tuple of (value, unit) or (None, None) if not found
    """
    text_lower = text.lower()
    ingredient_lower = ingredient.lower()
    
    # Common patterns for concentration extraction
    patterns = [
        # "10% DMSO", "10 % DMSO"
        rf'(\d+\.?\d*)\s*%\s*(?:v/v\s*)?{re.escape(ingredient_lower)}',
        rf'{re.escape(ingredient_lower)}\s*(\d+\.?\d*)\s*%',
        
        # "10% (v/v) DMSO"
        rf'(\d+\.?\d*)\s*%\s*\(?v/v\)?\s*{re.escape(ingredient_lower)}',
        
        # "0.5M trehalose", "0.5 M trehalose"
        rf'(\d+\.?\d*)\s*[mM]\s*{re.escape(ingredient_lower)}',
        rf'{re.escape(ingredient_lower)}\s*(\d+\.?\d*)\s*[mM](?!\w)',
        
        # "500 mM trehalose"
        rf'(\d+\.?\d*)\s*mM\s*{re.escape(ingredient_lower)}',
        rf'{re.escape(ingredient_lower)}\s*(\d+\.?\d*)\s*mM',
        
        # "30 mmol/L trehalose"
        rf'(\d+\.?\d*)\s*(?:mmol/[lL]|mM)\s*{re.escape(ingredient_lower)}',
        
        # "1.5M EG"
        rf'(\d+\.?\d*)\s*M\s*{re.escape(ingredient_lower)}',
        
        # Generic number before ingredient
        rf'(\d+\.?\d*)\s*(?:%|M|mM|mg/ml|wt%)\s*{re.escape(ingredient_lower)}',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            # Determine unit from pattern
            if '%' in pattern or 'wt%' in text_lower:
                return value, '%'
            elif 'mM' in pattern or 'mmol' in pattern.lower():
                return value, 'mM'
            elif 'M' in pattern:
                return value, 'M'
            elif 'mg/ml' in pattern:
                return value, 'mg/ml'
    
    return None, None


def convert_to_molar(value: float, unit: str, ingredient: str) -> Optional[float]:
    """
    Convert concentration to molar (M).
    
    Args:
        value: Numeric concentration value
        unit: Unit string ('%', 'mM', 'M', 'mg/ml')
        ingredient: Canonical ingredient name
        
    Returns:
        Concentration in molar (M), or None if conversion not possible
    """
    if value is None or unit is None:
        return None
    
    canonical = SYNONYM_TO_CANONICAL.get(ingredient.lower(), ingredient.lower())
    
    if unit == 'M':
        return value
    
    if unit == 'mM':
        return value / 1000.0
    
    if unit == '%':
        # For % v/v with known density and MW
        if canonical in DENSITIES and canonical in MOLECULAR_WEIGHTS:
            density = DENSITIES[canonical]
            mw = MOLECULAR_WEIGHTS[canonical]
            # % v/v -> g/mL -> mol/L
            return (value / 100.0) * density * 1000 / mw
        # For % w/v with known MW
        elif canonical in MOLECULAR_WEIGHTS:
            mw = MOLECULAR_WEIGHTS[canonical]
            # % w/v = g/100mL -> g/L / MW = M
            return (value * 10) / mw
        else:
            # Can't convert, keep as percentage (will need manual review)
            return None
    
    if unit == 'mg/ml':
        if canonical in MOLECULAR_WEIGHTS:
            mw = MOLECULAR_WEIGHTS[canonical]
            # mg/mL = g/L -> M
            return value / mw
    
    return None


def parse_formulation_text(text: str) -> Dict[str, float]:
    """
    Parse a formulation text to extract all ingredients and their concentrations.
    
    Args:
        text: Free-text formulation description
        
    Returns:
        Dictionary mapping canonical ingredient names to molar concentrations
    """
    if pd.isna(text) or not text.strip():
        return {}
    
    text_lower = text.lower()
    ingredients = {}
    
    # Check for each known ingredient
    for canonical, synonyms in INGREDIENT_SYNONYMS.items():
        for syn in synonyms:
            if syn.lower() in text_lower:
                value, unit = extract_concentration(text, syn)
                if value is not None:
                    molar = convert_to_molar(value, unit, canonical)
                    if molar is not None:
                        ingredients[canonical] = molar
                    else:
                        # Store raw value with unit indicator as negative (for review)
                        # Actually, let's store as-is and mark for review
                        ingredients[canonical] = value  # Store raw for now
                    break
                else:
                    # Ingredient mentioned but no concentration found
                    # Check if it's a qualitative mention (just presence)
                    pass
    
    return ingredients


def extract_viability(text: str) -> Optional[float]:
    """
    Extract viability percentage from viability column text.
    
    Args:
        text: Viability text (e.g., "82.5 ± 8.3%", "~80%", ">90%")
        
    Returns:
        Viability as float percentage, or None if not extractable
    """
    if pd.isna(text) or not text.strip():
        return None
    
    text = str(text).strip()
    
    # Skip non-mentioned values
    if 'not mentioned' in text.lower() or 'not reported' in text.lower():
        return None
    
    # Handle "compared to standard DMSO" cases - mark for review
    if 'compared to' in text.lower():
        # Extract the percentage if present
        match = re.search(r'(\d+\.?\d*)\s*%', text)
        if match:
            return float(match.group(1))
        return None
    
    # Handle ranges like "~50-65%"
    range_match = re.search(r'~?(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*%', text)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return (low + high) / 2
    
    # Handle "82.5 ± 8.3%" or "(82.5±8.3)%"
    match = re.search(r'\(?(\d+\.?\d*)\s*[±+/-]+\s*\d+\.?\d*\)?%?', text)
    if match:
        return float(match.group(1))
    
    # Handle ">80%" or "<10%"
    match = re.search(r'[<>≤≥~]?\s*(\d+\.?\d*)\s*%', text)
    if match:
        return float(match.group(1))
    
    # Handle plain percentage
    match = re.search(r'(\d+\.?\d*)\s*%', text)
    if match:
        return float(match.group(1))
    
    # Handle just a number that looks like percentage
    match = re.search(r'^(\d+\.?\d*)$', text.strip())
    if match:
        val = float(match.group(1))
        if 0 <= val <= 100:
            return val
    
    return None


def extract_dmso_percentage(dmso_col: str, formulation_text: str) -> float:
    """
    Extract DMSO percentage from the dedicated DMSO column or formulation text.
    
    Args:
        dmso_col: Value from "DMSO usage" column
        formulation_text: Full formulation text
        
    Returns:
        DMSO percentage as float, defaults to 0 if not found
    """
    # First try the dedicated DMSO column
    if pd.notna(dmso_col):
        text = str(dmso_col).strip()
        match = re.search(r'(\d+\.?\d*)\s*%?', text)
        if match:
            return float(match.group(1))
    
    # Fall back to parsing from formulation text
    ingredients = parse_formulation_text(formulation_text)
    if 'dmso' in ingredients:
        # Convert back to percentage if in molar
        molar = ingredients['dmso']
        # DMSO: MW = 78.13, density = 1.10
        # M -> % v/v
        percentage = molar * 78.13 / (1.10 * 10)
        return percentage
    
    return 0.0


def is_culture_media(text: str) -> bool:
    """Check if text refers to culture media only."""
    text_lower = text.lower().strip()
    for media in CULTURE_MEDIA:
        if media in text_lower:
            return True
    return False


def parse_csv(input_path: str) -> pd.DataFrame:
    """
    Parse the raw CSV file and extract structured formulation data.
    
    Args:
        input_path: Path to raw CSV file
        
    Returns:
        DataFrame with parsed formulation data
    """
    # Read CSV
    df = pd.read_csv(input_path, encoding='utf-8', on_bad_lines='skip')
    
    # Get column names
    formulation_col = df.columns[0]  # "All ingredients in cryoprotective solution"
    dmso_col = df.columns[1]  # "DMSO usage"
    viability_col = df.columns[3]  # "Viability"
    source_col = df.columns[8] if len(df.columns) > 8 else None  # "Source (DOI link)"
    
    # Collect all unique ingredients across all formulations
    all_ingredients = set()
    parsed_rows = []
    
    print("Parsing formulations...")
    
    for idx, row in df.iterrows():
        formulation_text = str(row[formulation_col]) if pd.notna(row[formulation_col]) else ''
        
        # Skip blank rows
        if not formulation_text.strip():
            continue
        
        # Extract viability
        viability = extract_viability(row[viability_col] if len(df.columns) > 3 else None)
        
        # Skip rows without viability data (can't train on them)
        if viability is None:
            continue
        
        # Parse ingredients
        ingredients = parse_formulation_text(formulation_text)
        
        # Also get DMSO from dedicated column for accuracy
        dmso_pct = extract_dmso_percentage(
            row[dmso_col] if len(df.columns) > 1 else None,
            formulation_text
        )
        
        # Convert DMSO % to molar
        if dmso_pct > 0:
            dmso_molar = (dmso_pct / 100.0) * 1.10 * 1000 / 78.13
            ingredients['dmso'] = dmso_molar
        
        # Track all ingredients
        all_ingredients.update(ingredients.keys())
        
        # Get source DOI
        source = row[source_col] if source_col and pd.notna(row[source_col]) else ''
        
        parsed_rows.append({
            'formulation_id': idx + 1,
            'original_text': formulation_text[:200],  # Truncate for readability
            'ingredients': ingredients,
            'viability_percent': viability,
            'dmso_percent': dmso_pct,
            'source_doi': source,
        })
    
    print(f"Parsed {len(parsed_rows)} formulations with viability data")
    print(f"Found {len(all_ingredients)} unique ingredients")
    
    # Create output DataFrame with one column per ingredient
    output_data = []
    for row in parsed_rows:
        row_data = {
            'formulation_id': row['formulation_id'],
            'viability_percent': row['viability_percent'],
            'dmso_percent': row['dmso_percent'],
            'source_doi': row['source_doi'],
        }
        
        # Add each ingredient column
        for ingredient in sorted(all_ingredients):
            col_name = f'{ingredient}_M'
            row_data[col_name] = row['ingredients'].get(ingredient, 0.0)
        
        output_data.append(row_data)
    
    output_df = pd.DataFrame(output_data)
    
    return output_df


def find_duplicate_formulations(df: pd.DataFrame) -> List[Tuple[int, int, float, float]]:
    """
    Find formulations with identical ingredient profiles but different viabilities.
    
    Returns:
        List of tuples: (row_idx1, row_idx2, viability1, viability2)
    """
    # Get ingredient columns
    ingredient_cols = [c for c in df.columns if c.endswith('_M')]
    
    duplicates = []
    seen = {}
    
    for idx, row in df.iterrows():
        # Create a hashable key from ingredient values
        key = tuple(round(row[col], 6) for col in ingredient_cols)
        
        if key in seen:
            prev_idx, prev_viability = seen[key]
            curr_viability = row['viability_percent']
            if abs(prev_viability - curr_viability) > 1.0:  # More than 1% difference
                duplicates.append((prev_idx, idx, prev_viability, curr_viability))
        else:
            seen[key] = (idx, row['viability_percent'])
    
    return duplicates


def auto_resolve_duplicates(df: pd.DataFrame, duplicates: List[Tuple], interactive: bool = False) -> pd.DataFrame:
    """
    Automatically resolve duplicate formulations by averaging viabilities.
    
    For identical formulations with different viabilities, this function:
    1. Groups all duplicates together
    2. Averages their viability values
    3. Keeps one representative row with the averaged viability
    
    Args:
        df: Parsed DataFrame
        duplicates: List of duplicate pairs
        interactive: If True, prompt for each duplicate (only for small numbers)
        
    Returns:
        DataFrame with duplicates resolved
    """
    if not duplicates:
        print("No duplicate formulations with different viabilities found.")
        return df
    
    print(f"\nFound {len(duplicates)} pairs of formulations with identical ingredients but different viabilities.")
    
    # Get ingredient columns
    ingredient_cols = [c for c in df.columns if c.endswith('_M')]
    
    # Group all rows by their ingredient profile
    profile_groups = defaultdict(list)
    for idx, row in df.iterrows():
        key = tuple(round(row[col], 6) for col in ingredient_cols)
        profile_groups[key].append(idx)
    
    # Find groups with multiple entries and different viabilities
    rows_to_drop = []
    rows_to_update = {}
    
    for key, indices in profile_groups.items():
        if len(indices) > 1:
            viabilities = [df.loc[idx, 'viability_percent'] for idx in indices]
            sources = [df.loc[idx, 'source_doi'] for idx in indices if pd.notna(df.loc[idx, 'source_doi'])]
            
            # Calculate average viability
            avg_viability = np.mean(viabilities)
            
            # Keep first row, update with average, drop others
            keep_idx = indices[0]
            rows_to_update[keep_idx] = {
                'viability_percent': avg_viability,
                'source_doi': '; '.join(set(str(s) for s in sources if s))[:500]  # Combine sources
            }
            rows_to_drop.extend(indices[1:])
    
    # Apply updates
    for idx, updates in rows_to_update.items():
        for col, val in updates.items():
            df.loc[idx, col] = val
    
    # Drop duplicate rows
    if rows_to_drop:
        df = df.drop(rows_to_drop).reset_index(drop=True)
        print(f"Averaged {len(rows_to_update)} duplicate groups, dropped {len(rows_to_drop)} redundant rows.")
        print(f"{len(df)} unique formulations remaining.")
    
    return df




def main():
    """Main entry point for parsing script."""
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    input_path = os.path.join(project_root, 'data', 'raw', 'Cryopreservative Data 2026.csv')
    output_path = os.path.join(project_root, 'data', 'processed', 'parsed_formulations.csv')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("=" * 80)
    print("CryoMN Formulation Data Parser")
    print("=" * 80)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    print()
    
    # Parse CSV
    df = parse_csv(input_path)
    
    # Find and resolve duplicates (automatically average viabilities)
    duplicates = find_duplicate_formulations(df)
    if duplicates:
        df = auto_resolve_duplicates(df, duplicates)
    
    # Save output
    df.to_csv(output_path, index=False)
    print(f"\nSaved parsed data to: {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total formulations: {len(df)}")
    print(f"Viability range: {df['viability_percent'].min():.1f}% - {df['viability_percent'].max():.1f}%")
    print(f"Mean viability: {df['viability_percent'].mean():.1f}%")
    print(f"Formulations with 0% DMSO: {(df['dmso_percent'] == 0).sum()}")
    print(f"Formulations with <5% DMSO: {(df['dmso_percent'] < 5).sum()}")
    
    # List ingredients
    ingredient_cols = [c for c in df.columns if c.endswith('_M')]
    print(f"\nIngredients detected ({len(ingredient_cols)}):")
    for col in sorted(ingredient_cols):
        non_zero = (df[col] > 0).sum()
        if non_zero > 0:
            print(f"  - {col.replace('_M', '')}: {non_zero} formulations")


if __name__ == '__main__':
    main()
