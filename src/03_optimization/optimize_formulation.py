#!/usr/bin/env python3
"""
CryoMN Multi-Objective Bayesian Optimization

Optimizes cryoprotective formulations to minimize DMSO usage while
maximizing cell viability using Bayesian optimization.

Author: CryoMN ML Project
Date: 2026-01-24
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
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm

# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for Bayesian optimization."""
    max_ingredients: int = 10  # Maximum non-zero ingredients per formulation
    max_dmso_percent: float = 5.0  # Maximum DMSO percentage
    min_viability: float = 70.0  # Minimum target viability
    n_candidates: int = 20  # Number of candidate formulations to generate
    acquisition: str = 'ei'  # Acquisition function: 'ei', 'ucb', 'poi'
    exploration_weight: float = 0.1  # UCB exploration weight (kappa)
    random_seed: int = 42


# =============================================================================
# ACQUISITION FUNCTIONS
# =============================================================================

def expected_improvement(mean: np.ndarray, std: np.ndarray, 
                         y_best: float, xi: float = 0.01) -> np.ndarray:
    """
    Calculate Expected Improvement acquisition function.
    
    Args:
        mean: Predicted mean values
        std: Predicted standard deviations
        y_best: Best observed value so far
        xi: Exploration-exploitation trade-off parameter
        
    Returns:
        EI values
    """
    # Handle zero variance
    std = np.maximum(std, 1e-9)
    
    z = (mean - y_best - xi) / std
    ei = (mean - y_best - xi) * norm.cdf(z) + std * norm.pdf(z)
    
    return ei


def upper_confidence_bound(mean: np.ndarray, std: np.ndarray, 
                            kappa: float = 2.0) -> np.ndarray:
    """
    Calculate Upper Confidence Bound acquisition function.
    
    Args:
        mean: Predicted mean values
        std: Predicted standard deviations
        kappa: Exploration weight
        
    Returns:
        UCB values
    """
    return mean + kappa * std


def probability_of_improvement(mean: np.ndarray, std: np.ndarray,
                                y_best: float, xi: float = 0.01) -> np.ndarray:
    """
    Calculate Probability of Improvement.
    
    Args:
        mean: Predicted mean values
        std: Predicted standard deviations
        y_best: Best observed value
        xi: Margin
        
    Returns:
        POI values
    """
    std = np.maximum(std, 1e-9)
    z = (mean - y_best - xi) / std
    return norm.cdf(z)


# =============================================================================
# CONSTRAINT HANDLING
# =============================================================================

def count_ingredients(x: np.ndarray, threshold: float = 1e-6) -> int:
    """Count non-zero ingredients in formulation."""
    return np.sum(np.abs(x) > threshold)


def dmso_penalty(x: np.ndarray, dmso_index: int, max_dmso_molar: float) -> float:
    """
    Calculate penalty for exceeding DMSO limit.
    
    Args:
        x: Formulation vector
        dmso_index: Index of DMSO in feature vector
        max_dmso_molar: Maximum allowed DMSO concentration (molar)
        
    Returns:
        Penalty value (0 if within limit)
    """
    if dmso_index < 0 or dmso_index >= len(x):
        return 0.0
    
    dmso_conc = x[dmso_index]
    if dmso_conc > max_dmso_molar:
        return (dmso_conc - max_dmso_molar) * 100  # Strong penalty
    return 0.0


def ingredient_count_penalty(x: np.ndarray, max_ingredients: int) -> float:
    """
    Calculate penalty for exceeding ingredient count.
    
    Args:
        x: Formulation vector
        max_ingredients: Maximum allowed ingredients
        
    Returns:
        Penalty value
    """
    count = count_ingredients(x)
    if count > max_ingredients:
        return (count - max_ingredients) * 10
    return 0.0


# =============================================================================
# OPTIMIZATION CORE
# =============================================================================

class FormulationOptimizer:
    """
    Multi-objective Bayesian optimizer for cryoprotective formulations.
    
    Objectives:
    1. Maximize viability (primary)
    2. Minimize DMSO usage (secondary)
    """
    
    def __init__(self, gp: GaussianProcessRegressor, scaler: StandardScaler,
                 feature_names: List[str], config: OptimizationConfig = None):
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
        self.config = config or OptimizationConfig()
        
        # Find DMSO index
        self.dmso_index = -1
        for i, name in enumerate(feature_names):
            if 'dmso' in name.lower():
                self.dmso_index = i
                break
        
        # Calculate max DMSO in molar (5% v/v ≈ 0.70 M)
        self.max_dmso_molar = (self.config.max_dmso_percent / 100.0) * 1.10 * 1000 / 78.13
        
        # Set feature bounds based on training data
        self.bounds = self._get_feature_bounds()
        
        np.random.seed(self.config.random_seed)
    
    def _get_feature_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for each feature based on typical concentration ranges."""
        bounds = []
        for name in self.feature_names:
            name_lower = name.lower()
            if 'dmso' in name_lower:
                # DMSO: 0 to max allowed
                bounds.append((0.0, self.max_dmso_molar))
            elif any(x in name_lower for x in ['ethylene_glycol', 'glycerol', 'propylene_glycol']):
                # Permeating CPAs: 0 to 2.5 M
                bounds.append((0.0, 2.5))
            elif any(x in name_lower for x in ['trehalose', 'sucrose', 'raffinose']):
                # Sugars: 0 to 1 M
                bounds.append((0.0, 1.0))
            elif any(x in name_lower for x in ['proline', 'betaine', 'ectoin', 'taurine', 'isoleucine']):
                # Amino acids: 0 to 0.5 M
                bounds.append((0.0, 0.5))
            elif any(x in name_lower for x in ['fbs', 'human_serum']):
                # Sera: 0 to 90% (normalized value)
                bounds.append((0.0, 90.0))
            else:
                # Other: 0 to 10 (generic bound)
                bounds.append((0.0, 10.0))
        
        return bounds
    
    def _objective(self, x: np.ndarray, y_best: float) -> float:
        """
        Combined objective function for optimization.
        
        Maximizes: viability - penalties
        """
        x_reshaped = x.reshape(1, -1)
        x_scaled = self.scaler.transform(x_reshaped)
        
        # Get prediction
        mean, std = self.gp.predict(x_scaled, return_std=True)
        mean = mean[0]
        std = std[0]
        
        # Calculate acquisition value (to maximize)
        if self.config.acquisition == 'ei':
            acq = expected_improvement(mean, std, y_best)
        elif self.config.acquisition == 'ucb':
            acq = upper_confidence_bound(mean, std, self.config.exploration_weight)
        else:
            acq = probability_of_improvement(mean, std, y_best)
        
        # Apply penalties
        penalty = 0.0
        penalty += dmso_penalty(x, self.dmso_index, self.max_dmso_molar)
        penalty += ingredient_count_penalty(x, self.config.max_ingredients)
        
        # Return negative (for minimization)
        return -(acq - penalty)
    
    def _generate_random_candidate(self) -> np.ndarray:
        """Generate a random candidate formulation."""
        x = np.zeros(len(self.feature_names))
        
        # Select random subset of ingredients
        n_ingredients = np.random.randint(2, self.config.max_ingredients + 1)
        selected_indices = np.random.choice(
            len(self.feature_names), 
            size=n_ingredients, 
            replace=False
        )
        
        # Assign random concentrations
        for idx in selected_indices:
            low, high = self.bounds[idx]
            x[idx] = np.random.uniform(low, high)
        
        return x
    
    def optimize(self, X_observed: np.ndarray, y_observed: np.ndarray,
                 n_candidates: int = None) -> pd.DataFrame:
        """
        Generate optimized candidate formulations using random sampling + GP prediction.
        
        Args:
            X_observed: Observed formulation features
            y_observed: Observed viability values
            n_candidates: Number of candidates to generate
            
        Returns:
            DataFrame with candidate formulations
        """
        if n_candidates is None:
            n_candidates = self.config.n_candidates
        
        y_best = np.max(y_observed)
        print(f"Best observed viability: {y_best:.1f}%")
        
        # Generate many random candidates and select the best
        n_samples = n_candidates * 50  # Over-sample
        candidates = []
        
        print(f"Generating {n_samples} random formulations...")
        
        for i in range(n_samples):
            x = self._generate_random_candidate()
            
            # Skip if too many ingredients
            n_ing = count_ingredients(x)
            if n_ing > self.config.max_ingredients or n_ing < 1:
                continue
            
            # Check DMSO constraint
            if self.dmso_index >= 0:
                dmso_molar = x[self.dmso_index]
                if dmso_molar > self.max_dmso_molar:
                    continue
            
            # Predict viability
            x_scaled = self.scaler.transform(x.reshape(1, -1))
            pred_mean, pred_std = self.gp.predict(x_scaled, return_std=True)
            
            # Calculate DMSO percentage
            dmso_molar = x[self.dmso_index] if self.dmso_index >= 0 else 0
            dmso_percent = dmso_molar * 78.13 / (1.10 * 10)
            
            candidate = {
                'predicted_viability': pred_mean[0],
                'uncertainty': pred_std[0],
                'dmso_percent': dmso_percent,
                'n_ingredients': n_ing,
                'formulation': x.copy(),
            }
            
            candidates.append(candidate)
        
        print(f"Generated {len(candidates)} valid candidates")
        
        if len(candidates) == 0:
            # Fallback: generate at least some candidates without constraints
            print("Warning: No valid candidates, relaxing constraints...")
            for i in range(n_candidates):
                x = self._generate_random_candidate()
                x_scaled = self.scaler.transform(x.reshape(1, -1))
                pred_mean, pred_std = self.gp.predict(x_scaled, return_std=True)
                dmso_molar = x[self.dmso_index] if self.dmso_index >= 0 else 0
                dmso_percent = dmso_molar * 78.13 / (1.10 * 10)
                
                candidates.append({
                    'predicted_viability': pred_mean[0],
                    'uncertainty': pred_std[0],
                    'dmso_percent': dmso_percent,
                    'n_ingredients': count_ingredients(x),
                    'formulation': x.copy(),
                })
        
        # Sort by predicted viability and select top candidates
        candidates.sort(key=lambda c: c['predicted_viability'], reverse=True)
        top_candidates = candidates[:n_candidates]
        
        # Build output DataFrame
        output_data = []
        for rank, c in enumerate(top_candidates, 1):
            row = {
                'rank': rank,
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
        
        candidates_df = pd.DataFrame(output_data)
        return candidates_df
    
    def generate_low_dmso_candidates(self, X_observed: np.ndarray, 
                                      y_observed: np.ndarray,
                                      n_candidates: int = 10) -> pd.DataFrame:
        """
        Generate candidates with zero or minimal DMSO.
        
        Args:
            X_observed: Observed formulations
            y_observed: Observed viabilities
            n_candidates: Number of candidates
            
        Returns:
            DataFrame with DMSO-free/low candidates
        """
        # Temporarily set max DMSO to very low
        original_max = self.max_dmso_molar
        self.max_dmso_molar = 0.07  # ~0.5% DMSO
        
        # Force DMSO bound to near-zero
        if self.dmso_index >= 0:
            original_bound = self.bounds[self.dmso_index]
            self.bounds[self.dmso_index] = (0.0, 0.07)
        
        try:
            candidates = self.optimize(X_observed, y_observed, n_candidates)
        finally:
            # Restore original settings
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
            clean_name = name.replace('_M', '')
            conc = row[name]
            
            # Format concentration appropriately
            if conc >= 1.0:
                parts.append(f"{conc:.2f}M {clean_name}")
            elif conc >= 0.001:
                parts.append(f"{conc*1000:.1f}mM {clean_name}")
            else:
                parts.append(f"{conc*1e6:.1f}µM {clean_name}")
    
    return ' + '.join(parts)


def export_candidates(candidates_df: pd.DataFrame, feature_names: List[str],
                      output_path: str):
    """Export candidate formulations to CSV and human-readable format."""
    # Save full CSV
    candidates_df.to_csv(output_path, index=False)
    
    # Create human-readable summary
    summary_path = output_path.replace('.csv', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CryoMN Optimized Formulation Candidates\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 80 + "\n\n")
        
        for _, row in candidates_df.head(10).iterrows():
            f.write(f"Rank {int(row['rank'])}: {format_formulation(row, feature_names)}\n")
            f.write(f"  Predicted viability: {row['predicted_viability']:.1f}% ± {row['uncertainty']:.1f}%\n")
            f.write(f"  DMSO: {row['dmso_percent']:.1f}%\n")
            f.write(f"  Ingredients: {int(row['n_ingredients'])}\n")
            f.write("\n")
    
    print(f"Candidates saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for optimization."""
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    model_dir = os.path.join(project_root, 'models')
    data_path = os.path.join(project_root, 'data', 'processed', 'parsed_formulations.csv')
    output_dir = os.path.join(project_root, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("CryoMN Multi-Objective Bayesian Optimization")
    print("=" * 80)
    
    # Load model
    print("\nLoading trained model...")
    model_path = os.path.join(model_dir, 'gp_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    
    with open(model_path, 'rb') as f:
        gp = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    feature_names = metadata['feature_names']
    print(f"Model loaded with {len(feature_names)} features")
    
    # Load observed data
    print(f"\nLoading observed data from: {data_path}")
    df = pd.read_csv(data_path)
    df = df[df['viability_percent'] <= 100].copy()
    
    # Prepare features
    ingredient_cols = [c for c in df.columns if c.endswith('_M')]
    active_ingredients = [c for c in ingredient_cols if (df[c] > 0).sum() >= 3]
    
    X = df[active_ingredients].values
    y = df['viability_percent'].values
    
    # Ensure feature alignment
    if active_ingredients != feature_names:
        print("Warning: Feature alignment needed")
        # Re-extract using model's feature names
        X = df[feature_names].values
    
    print(f"Loaded {len(df)} formulations")
    
    # Initialize optimizer
    config = OptimizationConfig(
        max_ingredients=10,
        max_dmso_percent=5.0,
        min_viability=70.0,
        n_candidates=20,
        acquisition='ei',
    )
    
    optimizer = FormulationOptimizer(gp, scaler, feature_names, config)
    
    # Generate candidates
    print("\n" + "-" * 40)
    print("Generating Optimized Candidates")
    print("-" * 40)
    
    print("\n1. General optimization (up to 5% DMSO allowed)...")
    general_candidates = optimizer.optimize(X, y, n_candidates=20)
    
    print("\n2. DMSO-free optimization...")
    dmso_free_candidates = optimizer.generate_low_dmso_candidates(X, y, n_candidates=15)
    
    # Export results
    print("\n" + "-" * 40)
    print("Exporting Results")
    print("-" * 40)
    
    export_candidates(
        general_candidates, 
        feature_names,
        os.path.join(output_dir, 'candidates_general.csv')
    )
    
    export_candidates(
        dmso_free_candidates,
        feature_names,
        os.path.join(output_dir, 'candidates_dmso_free.csv')``
    )
    
    # Print top candidates
    print("\n" + "=" * 80)
    print("Top 5 General Candidates")
    print("=" * 80)
    for _, row in general_candidates.head(5).iterrows():
        print(f"\nRank {int(row['rank'])}: Viability = {row['predicted_viability']:.1f}% ± {row['uncertainty']:.1f}%")
        print(f"  DMSO: {row['dmso_percent']:.1f}%, Ingredients: {int(row['n_ingredients'])}")
    
    print("\n" + "=" * 80)
    print("Top 5 DMSO-Free Candidates")
    print("=" * 80)
    for _, row in dmso_free_candidates.head(5).iterrows():
        print(f"\nRank {int(row['rank'])}: Viability = {row['predicted_viability']:.1f}% ± {row['uncertainty']:.1f}%")
        print(f"  DMSO: {row['dmso_percent']:.1f}%, Ingredients: {int(row['n_ingredients'])}")
    
    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
