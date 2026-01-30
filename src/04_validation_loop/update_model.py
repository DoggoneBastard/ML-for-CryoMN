#!/usr/bin/env python3
"""
CryoMN Validation Loop

Integrates wet lab validation results to refine the GP model.
Supports iterative experiment-model-optimize cycles.

Author: CryoMN ML Project
Date: 2026-01-24
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import shutil
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler


# =============================================================================
# VALIDATION DATA HANDLING
# =============================================================================

def create_validation_template(feature_names: List[str], output_path: str):
    """
    Create a template CSV for entering wet lab validation results.
    
    Args:
        feature_names: List of ingredient feature names
        output_path: Path to save template
    """
    columns = ['experiment_id', 'experiment_date', 'viability_measured', 'notes']
    columns.extend([name.replace('_M', '').replace('_pct', '') for name in feature_names])
    
    template_df = pd.DataFrame(columns=columns)
    
    # Add example rows
    template_df.loc[0] = ['EXP001', '2026-01-25', 85.5, 'Example entry'] + [0.0] * len(feature_names)
    
    template_df.to_csv(output_path, index=False)
    print(f"Validation template created: {output_path}")
    return template_df


def load_validation_results(validation_path: str, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load validated formulation results.
    
    Args:
        validation_path: Path to validation CSV
        feature_names: List of feature names (with _M suffix)
        
    Returns:
        Tuple of (X features, y viability)
    """
    df = pd.read_csv(validation_path)
    
    # Map column names
    X_data = []
    y_data = []
    
    for _, row in df.iterrows():
        if pd.isna(row['viability_measured']):
            continue
        
        x = np.zeros(len(feature_names))
        for i, name in enumerate(feature_names):
            col_name = name.replace('_M', '').replace('_pct', '')
            if col_name in row and pd.notna(row[col_name]):
                x[i] = float(row[col_name])
        
        X_data.append(x)
        y_data.append(float(row['viability_measured']))
    
    if len(X_data) == 0:
        return np.array([]).reshape(0, len(feature_names)), np.array([])
    
    return np.array(X_data), np.array(y_data)


# =============================================================================
# MODEL UPDATE
# =============================================================================

def update_model(original_model_dir: str, validation_data: Tuple[np.ndarray, np.ndarray],
                 original_data: Tuple[np.ndarray, np.ndarray], output_dir: str) -> Dict:
    """
    Update GP model with new validation data.
    
    Args:
        original_model_dir: Path to original model directory
        validation_data: Tuple of (X_val, y_val) from wet lab
        original_data: Tuple of (X_orig, y_orig) from literature
        output_dir: Directory to save updated model
        
    Returns:
        Dictionary with update statistics
    """
    X_val, y_val = validation_data
    X_orig, y_orig = original_data
    
    # Combine datasets
    X_combined = np.vstack([X_orig, X_val])
    y_combined = np.concatenate([y_orig, y_val])
    
    print(f"Original data: {len(X_orig)} samples")
    print(f"Validation data: {len(X_val)} samples")
    print(f"Combined data: {len(X_combined)} samples")
    
    # Load original scaler and retrain
    scaler_path = os.path.join(original_model_dir, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        original_scaler = pickle.load(f)
    
    # Create new scaler on combined data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Create and train new GP
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5, length_scale_bounds=(1e-5, 1e5)) + WhiteKernel(noise_level=1.0)
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        random_state=42,
        alpha=1e-6,
        normalize_y=True,
    )
    
    print("Training updated model...")
    gp.fit(X_scaled, y_combined)
    
    # Evaluate on validation data
    X_val_scaled = scaler.transform(X_val)
    y_val_pred = gp.predict(X_val_scaled)
    val_rmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    
    print(f"Optimized kernel: {gp.kernel_}")
    print(f"Validation RMSE: {val_rmse:.2f}")
    
    # Save updated model
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'gp_model.pkl'), 'wb') as f:
        pickle.dump(gp, f)
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Load and update metadata
    metadata_path = os.path.join(original_model_dir, 'model_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    metadata['updated_at'] = datetime.now().isoformat()
    metadata['n_validation_samples'] = len(X_val)
    metadata['n_total_samples'] = len(X_combined)
    metadata['validation_rmse'] = val_rmse
    
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Updated model saved to: {output_dir}")
    
    return {
        'n_original': len(X_orig),
        'n_validation': len(X_val),
        'n_total': len(X_combined),
        'validation_rmse': val_rmse,
    }


# =============================================================================
# ITERATION TRACKING
# =============================================================================

def get_iteration_number(project_dir: str) -> int:
    """Get current iteration number from history."""
    history_path = os.path.join(project_dir, 'data', 'validation', 'iteration_history.json')
    
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        return len(history.get('iterations', []))
    return 0


def save_iteration(project_dir: str, iteration_data: Dict):
    """Save iteration information to history."""
    history_path = os.path.join(project_dir, 'data', 'validation', 'iteration_history.json')
    
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = {'iterations': []}
    
    iteration_data['timestamp'] = datetime.now().isoformat()
    history['iterations'].append(iteration_data)
    
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Iteration {len(history['iterations'])} logged")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for validation loop."""
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    model_dir = os.path.join(project_root, 'models')
    data_dir = os.path.join(project_root, 'data')
    validation_dir = os.path.join(data_dir, 'validation')
    os.makedirs(validation_dir, exist_ok=True)
    
    print("=" * 80)
    print("CryoMN Validation Loop")
    print("=" * 80)
    
    # Load model metadata
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    feature_names = metadata['feature_names']
    print(f"\nModel has {len(feature_names)} features")
    
    # Check for validation data
    validation_path = os.path.join(validation_dir, 'validation_results.csv')
    template_path = os.path.join(validation_dir, 'validation_template.csv')
    
    if not os.path.exists(template_path):
        print("\nCreating validation template...")
        create_validation_template(feature_names, template_path)
    
    if not os.path.exists(validation_path):
        print("\n" + "=" * 80)
        print("No validation results found.")
        print(f"\nTo add wet lab results:")
        print(f"  1. Copy: {template_path}")
        print(f"  2. To:   {validation_path}")
        print(f"  3. Fill in your experimental results")
        print(f"  4. Run this script again")
        print("=" * 80)
        return
    
    # Load validation data
    print(f"\nLoading validation results from: {validation_path}")
    X_val, y_val = load_validation_results(validation_path, feature_names)
    
    if len(X_val) == 0:
        print("No valid validation entries found. Please add results to the CSV.")
        return
    
    print(f"Found {len(X_val)} validation experiments")
    print(f"Viability range: {y_val.min():.1f}% - {y_val.max():.1f}%")
    
    # Load original training data
    parsed_path = os.path.join(data_dir, 'processed', 'parsed_formulations.csv')
    df_orig = pd.read_csv(parsed_path)
    df_orig = df_orig[df_orig['viability_percent'] <= 100].copy()
    
    X_orig = df_orig[feature_names].values
    y_orig = df_orig['viability_percent'].values
    
    # Get iteration number
    iteration = get_iteration_number(project_root) + 1
    print(f"\n--- Iteration {iteration} ---")
    
    # Update model
    updated_model_dir = os.path.join(model_dir, f'iteration_{iteration}')
    stats = update_model(
        model_dir,
        (X_val, y_val),
        (X_orig, y_orig),
        updated_model_dir
    )
    
    # Also update main model directory
    print("\nUpdating main model...")
    for filename in ['gp_model.pkl', 'scaler.pkl', 'model_metadata.json']:
        src = os.path.join(updated_model_dir, filename)
        dst = os.path.join(model_dir, filename)
        shutil.copy(src, dst)
    
    # Save iteration history
    save_iteration(project_root, {
        'iteration': iteration,
        'n_validation_samples': stats['n_validation'],
        'validation_rmse': stats['validation_rmse'],
    })
    
    print("\n" + "=" * 80)
    print("Validation Loop Complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Run optimization: python src/03_optimization/optimize_formulation.py")
    print(f"  2. Test top candidates in wet lab")
    print(f"  3. Add results to: {validation_path}")
    print(f"  4. Run this script again for next iteration")


if __name__ == '__main__':
    main()
