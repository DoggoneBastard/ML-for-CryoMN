#!/usr/bin/env python3
"""
CryoMN Bayesian Optimization with Differential Evolution

Proper Bayesian optimization using DE to maximize acquisition functions.
This provides better exploration-exploitation balance compared to random sampling.

Author: CryoMN ML Project
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from typing import Tuple, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution
from scipy.stats import norm


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BOConfig:
    """Configuration for Bayesian optimization with DE."""
    max_ingredients: int = 10
    max_dmso_percent: float = 5.0  # Set to 0.5 for DMSO-free
    min_viability: float = 70.0
    n_candidates: int = 20
    xi: float = 0.01  # EI exploration parameter
    de_maxiter: int = 100  # DE iterations per candidate
    de_popsize: int = 15  # DE population size
    random_seed: int = 42


# =============================================================================
# ACQUISITION FUNCTIONS
# =============================================================================

def expected_improvement(x: np.ndarray, gp: GaussianProcessRegressor, 
                         scaler: StandardScaler, y_best: float, 
                         xi: float = 0.01) -> float:
    """
    Calculate Expected Improvement for a single point.
    
    Args:
        x: Formulation vector (unscaled)
        gp: Trained Gaussian Process
        scaler: Feature scaler
        y_best: Best observed viability
        xi: Exploration-exploitation trade-off
        
    Returns:
        Negative EI (for minimization)
    """
    x_scaled = scaler.transform(x.reshape(1, -1))
    mean, std = gp.predict(x_scaled, return_std=True)
    mean, std = mean[0], std[0]
    
    # Handle zero variance
    if std < 1e-9:
        return 0.0
    
    z = (mean - y_best - xi) / std
    ei = (mean - y_best - xi) * norm.cdf(z) + std * norm.pdf(z)
    
    # Return negative for minimization (DE minimizes)
    return -ei


# =============================================================================
# CONSTRAINT HANDLING
# =============================================================================

def count_nonzero(x: np.ndarray, threshold: float = 1e-6) -> int:
    """Count non-zero ingredients."""
    return np.sum(np.abs(x) > threshold)


def ingredient_constraint(x: np.ndarray, max_ingredients: int) -> float:
    """Constraint: n_ingredients <= max_ingredients. Returns >=0 if satisfied."""
    return max_ingredients - count_nonzero(x)


def dmso_constraint(x: np.ndarray, dmso_index: int, max_dmso_molar: float) -> float:
    """Constraint: DMSO <= max. Returns >=0 if satisfied."""
    if dmso_index < 0:
        return 1.0  # No DMSO feature, constraint satisfied
    return max_dmso_molar - x[dmso_index]


# =============================================================================
# DE-BASED OPTIMIZER
# =============================================================================

class BayesianOptimizer:
    """
    Bayesian Optimizer using Differential Evolution.
    
    Uses DE to find formulations that maximize Expected Improvement,
    providing proper exploration-exploitation balance.
    """
    
    def __init__(self, gp: GaussianProcessRegressor, scaler: StandardScaler,
                 feature_names: List[str], config: BOConfig = None):
        """
        Initialize optimizer.
        
        Args:
            gp: Trained Gaussian Process model
            scaler: Feature scaler
            feature_names: List of feature names
            config: Optimization configuration
        """
        self.gp = gp
        self.scaler = scaler
        self.feature_names = feature_names
        self.config = config or BOConfig()
        
        # Find DMSO index
        self.dmso_index = -1
        for i, name in enumerate(feature_names):
            if 'dmso' in name.lower():
                self.dmso_index = i
                break
        
        # Calculate max DMSO in molar (5% v/v ≈ 0.70 M)
        self.max_dmso_molar = (self.config.max_dmso_percent / 100.0) * 1.10 * 1000 / 78.13
        
        # Set feature bounds
        self.bounds = self._get_feature_bounds()
        
        np.random.seed(self.config.random_seed)
    
    def _get_feature_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for each feature based on typical concentration ranges."""
        bounds = []
        for name in self.feature_names:
            name_lower = name.lower()
            if 'dmso' in name_lower:
                bounds.append((0.0, self.max_dmso_molar))
            elif any(x in name_lower for x in ['ethylene_glycol', 'glycerol', 'propylene_glycol']):
                bounds.append((0.0, 2.5))
            elif any(x in name_lower for x in ['trehalose', 'sucrose', 'raffinose']):
                bounds.append((0.0, 1.0))
            elif any(x in name_lower for x in ['proline', 'betaine', 'ectoin', 'taurine', 'isoleucine']):
                bounds.append((0.0, 0.5))
            elif any(x in name_lower for x in ['fbs', 'human_serum']):
                bounds.append((0.0, 90.0))
            elif any(x in name_lower for x in ['hyaluronic_acid', 'methylcellulose']):
                bounds.append((0.0, 2.0))
            else:
                bounds.append((0.0, 10.0))
        return bounds
    
    def _sparsify(self, x: np.ndarray) -> np.ndarray:
        """
        Enforce max ingredients by zeroing smallest components.
        """
        x_sparse = x.copy()
        n_ing = count_nonzero(x_sparse)
        if n_ing > self.config.max_ingredients:
            nonzero_idx = np.where(np.abs(x_sparse) > 1e-6)[0]
            sorted_idx = nonzero_idx[np.argsort(np.abs(x_sparse[nonzero_idx]))]
            for idx in sorted_idx[:n_ing - self.config.max_ingredients]:
                x_sparse[idx] = 0.0
        return x_sparse
    
    def _objective_with_penalty(self, x: np.ndarray, y_best: float) -> float:
        """
        Objective function for DE: negative EI + constraint penalties.
        """
        # Sparsify first to enforce ingredient constraint
        x_sparse = self._sparsify(x)
        
        # Calculate EI on sparsified formulation
        neg_ei = expected_improvement(x_sparse, self.gp, self.scaler, y_best, self.config.xi)
        
        # Soft penalty for DMSO (in case it slightly exceeds)
        penalty = 0.0
        if self.dmso_index >= 0 and x_sparse[self.dmso_index] > self.max_dmso_molar:
            penalty += (x_sparse[self.dmso_index] - self.max_dmso_molar) * 50.0
        
        return neg_ei + penalty
    
    def _run_de_single(self, y_best: float, seed: int) -> Tuple[np.ndarray, float]:
        """
        Run a single DE optimization to find one candidate.
        
        Returns:
            Tuple of (best formulation, EI value)
        """
        result = differential_evolution(
            func=lambda x: self._objective_with_penalty(x, y_best),
            bounds=self.bounds,
            maxiter=self.config.de_maxiter,
            popsize=self.config.de_popsize,
            seed=seed,
            polish=True,  # Use L-BFGS-B to polish the result
            disp=False,
        )
        
        return result.x, -result.fun  # Return positive EI
    
    def optimize(self, X_observed: np.ndarray, y_observed: np.ndarray,
                 n_candidates: int = None) -> pd.DataFrame:
        """
        Generate optimized candidates using DE-based acquisition maximization.
        
        Args:
            X_observed: Observed formulation features
            y_observed: Observed viability values
            n_candidates: Number of candidates to generate
            
        Returns:
            DataFrame with candidate formulations ranked by EI
        """
        if n_candidates is None:
            n_candidates = self.config.n_candidates
        
        y_best = np.max(y_observed)
        print(f"Best observed viability: {y_best:.1f}%")
        print(f"Running DE optimization for {n_candidates} candidates...")
        
        candidates = []
        
        for i in range(n_candidates):
            seed = self.config.random_seed + i
            x_opt, ei_value = self._run_de_single(y_best, seed)
            
            # Enforce constraints by clipping
            if self.dmso_index >= 0:
                x_opt[self.dmso_index] = min(x_opt[self.dmso_index], self.max_dmso_molar)
            
            # Sparsify: zero out smallest components if too many ingredients
            n_ing = count_nonzero(x_opt)
            if n_ing > self.config.max_ingredients:
                # Zero out smallest components
                nonzero_idx = np.where(np.abs(x_opt) > 1e-6)[0]
                sorted_idx = nonzero_idx[np.argsort(np.abs(x_opt[nonzero_idx]))]
                for idx in sorted_idx[:n_ing - self.config.max_ingredients]:
                    x_opt[idx] = 0.0
            
            # Get final prediction
            x_scaled = self.scaler.transform(x_opt.reshape(1, -1))
            pred_mean, pred_std = self.gp.predict(x_scaled, return_std=True)
            
            # Calculate DMSO percentage
            dmso_molar = x_opt[self.dmso_index] if self.dmso_index >= 0 else 0
            dmso_percent = dmso_molar * 78.13 / (1.10 * 10)
            
            candidates.append({
                'formulation': x_opt.copy(),
                'ei_value': ei_value,
                'predicted_viability': pred_mean[0],
                'uncertainty': pred_std[0],
                'dmso_percent': dmso_percent,
                'n_ingredients': count_nonzero(x_opt),
            })
            
            if (i + 1) % 5 == 0:
                print(f"  Generated {i + 1}/{n_candidates} candidates...")
        
        # Sort by EI value (higher is better - more informative to test)
        candidates.sort(key=lambda c: c['ei_value'], reverse=True)
        
        # Build output DataFrame
        output_data = []
        for rank, c in enumerate(candidates, 1):
            row = {
                'rank': rank,
                'expected_improvement': c['ei_value'],
                'predicted_viability': c['predicted_viability'],
                'uncertainty': c['uncertainty'],
                'dmso_percent': c['dmso_percent'],
                'n_ingredients': c['n_ingredients'],
            }
            
            # Add ingredient concentrations
            x = c['formulation']
            for j, name in enumerate(self.feature_names):
                if x[j] > 1e-6:
                    row[name] = x[j]
            
            output_data.append(row)
        
        return pd.DataFrame(output_data)
    
    def generate_dmso_free_candidates(self, X_observed: np.ndarray,
                                       y_observed: np.ndarray,
                                       n_candidates: int = 20) -> pd.DataFrame:
        """Generate DMSO-free candidates."""
        # Temporarily set DMSO bound to near-zero
        original_max = self.max_dmso_molar
        self.max_dmso_molar = 0.07  # ~0.5% DMSO
        
        if self.dmso_index >= 0:
            original_bound = self.bounds[self.dmso_index]
            self.bounds[self.dmso_index] = (0.0, 0.07)
        
        try:
            candidates = self.optimize(X_observed, y_observed, n_candidates)
        finally:
            self.max_dmso_molar = original_max
            if self.dmso_index >= 0:
                self.bounds[self.dmso_index] = original_bound
        
        return candidates


# =============================================================================
# RESULTS EXPORT
# =============================================================================

def format_formulation(row: pd.Series, feature_names: List[str]) -> str:
    """Format a formulation as human-readable string."""
    parts = []
    for name in feature_names:
        if name in row and row[name] > 1e-6:
            # Handle both _M (molar) and _pct (percentage) suffixes
            if name.endswith('_pct'):
                clean_name = name.replace('_pct', '')
                conc = row[name]
                parts.append(f"{conc:.1f}% {clean_name}")
            else:
                clean_name = name.replace('_M', '')
                conc = row[name]
                if conc >= 1.0:
                    parts.append(f"{conc:.2f}M {clean_name}")
                elif conc >= 0.001:
                    parts.append(f"{conc*1000:.1f}mM {clean_name}")
                else:
                    parts.append(f"{conc*1e6:.1f}µM {clean_name}")
    return ' + '.join(parts)


def export_candidates(candidates_df: pd.DataFrame, feature_names: List[str],
                      output_path: str):
    """Export candidate formulations to CSV and summary."""
    candidates_df.to_csv(output_path, index=False)
    
    summary_path = output_path.replace('.csv', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CryoMN Bayesian Optimization Candidates (DE-based)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 80 + "\n\n")
        
        for _, row in candidates_df.iterrows():
            f.write(f"Rank {int(row['rank'])}: {format_formulation(row, feature_names)}\n")
            f.write(f"  Expected Improvement: {row['expected_improvement']:.4f}\n")
            f.write(f"  Predicted viability: {row['predicted_viability']:.1f}% ± {row['uncertainty']:.1f}%\n")
            f.write(f"  DMSO: {row['dmso_percent']:.1f}%\n")
            f.write(f"  Ingredients: {int(row['n_ingredients'])}\n\n")
    
    print(f"Candidates saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for DE-based Bayesian optimization."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    model_dir = os.path.join(project_root, 'models')
    data_path = os.path.join(project_root, 'data', 'processed', 'parsed_formulations.csv')
    output_dir = os.path.join(project_root, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("CryoMN Bayesian Optimization with Differential Evolution")
    print("=" * 80)
    
    # Load model
    print("\nLoading trained model...")
    with open(os.path.join(model_dir, 'gp_model.pkl'), 'rb') as f:
        gp = pickle.load(f)
    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    feature_names = metadata['feature_names']
    print(f"Model loaded with {len(feature_names)} features")
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    df = df[df['viability_percent'] <= 100].copy()
    
    X = df[feature_names].values
    y = df['viability_percent'].values
    print(f"Loaded {len(df)} formulations")
    
    # Initialize optimizer
    config = BOConfig(
        max_ingredients=10,
        max_dmso_percent=5.0,
        n_candidates=20,
    )
    
    optimizer = BayesianOptimizer(gp, scaler, feature_names, config)
    
    # Generate candidates
    print("\n" + "-" * 40)
    print("Generating Candidates via DE")
    print("-" * 40)
    
    print("\n1. General optimization (≤5% DMSO)...")
    general_candidates = optimizer.optimize(X, y, n_candidates=20)
    
    print("\n2. DMSO-free optimization...")
    dmso_free_candidates = optimizer.generate_dmso_free_candidates(X, y, n_candidates=20)
    
    # Export results
    print("\n" + "-" * 40)
    print("Exporting Results")
    print("-" * 40)
    
    export_candidates(
        general_candidates,
        feature_names,
        os.path.join(output_dir, 'bo_candidates_general.csv')
    )
    
    export_candidates(
        dmso_free_candidates,
        feature_names,
        os.path.join(output_dir, 'bo_candidates_dmso_free.csv')
    )
    
    # Print top candidates
    print("\n" + "=" * 80)
    print("Top 20 General Candidates (by Expected Improvement)")
    print("=" * 80)
    for _, row in general_candidates.head(20).iterrows():
        print(f"\nRank {int(row['rank'])}: EI = {row['expected_improvement']:.4f}")
        print(f"  Predicted: {row['predicted_viability']:.1f}% ± {row['uncertainty']:.1f}%")
        print(f"  DMSO: {row['dmso_percent']:.1f}%, Ingredients: {int(row['n_ingredients'])}")
    
    print("\n" + "=" * 80)
    print("Top 20 DMSO-Free Candidates")
    print("=" * 80)
    for _, row in dmso_free_candidates.head(20).iterrows():
        print(f"\nRank {int(row['rank'])}: EI = {row['expected_improvement']:.4f}")
        print(f"  Predicted: {row['predicted_viability']:.1f}% ± {row['uncertainty']:.1f}%")
        print(f"  DMSO: {row['dmso_percent']:.1f}%, Ingredients: {int(row['n_ingredients'])}")
    
    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
