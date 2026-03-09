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
import sys
import json
import pickle
from typing import Tuple, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm

# Add validation loop to path for CompositeGP deserialization
_script_dir = os.path.dirname(os.path.abspath(__file__))
_validation_dir = os.path.join(os.path.dirname(_script_dir), '04_validation_loop')
if _validation_dir not in sys.path:
    sys.path.insert(0, _validation_dir)
from update_model_weighted_prior import CompositeGP  # noqa: E402
from iteration_metadata import (  # noqa: E402
    derive_iteration_dir,
    load_iteration_history,
    method_uses_composite,
    normalize_model_method,
    stamp_model_metadata,
    write_metadata_with_notice,
)

_main_module = sys.modules.get('__main__')
if _main_module is not None and not hasattr(_main_module, 'CompositeGP'):
    setattr(_main_module, 'CompositeGP', CompositeGP)

# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================

class ModelResolutionError(RuntimeError):
    """Raised when the active model cannot be resolved safely."""


@dataclass
class IterationCandidate:
    """One validated iteration that can be loaded safely."""
    iteration: int
    model_method: str
    iteration_dir: str
    is_composite_model: bool
    metadata: Dict
    directory: str


def _try_load_json(path: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Read JSON and return a user-facing error instead of throwing."""
    if not os.path.exists(path):
        return None, f"{path} is missing."
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return None, f"{path} is malformed JSON: {exc}"
    if not isinstance(data, dict):
        return None, f"{path} does not contain a JSON object."
    return data, None


def _load_history_entries(project_root: str) -> Tuple[List[Dict], Optional[str]]:
    """Load iteration history while preserving parse errors for conflict reporting."""
    history_path = os.path.join(project_root, 'data', 'validation', 'iteration_history.json')
    if not os.path.exists(history_path):
        return [], None
    try:
        return load_iteration_history(project_root), None
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return [], f"{history_path} could not be read: {exc}"


def _normalize_history_method(entry: Dict) -> str:
    """Resolve old and new history fields to one normalized method label."""
    return normalize_model_method(
        entry.get('model_method') or entry.get('method'),
        entry.get('is_composite_model'),
    )


def _build_iteration_candidate(model_dir: str, entry: Dict) -> Tuple[Optional[IterationCandidate], Optional[str]]:
    """Validate one history entry against the on-disk iteration artifacts."""
    iteration = entry.get('iteration')
    if not isinstance(iteration, int) or iteration <= 0:
        return None, f"Invalid history entry without a positive integer iteration: {entry!r}"

    model_method = _normalize_history_method(entry)
    iteration_dir = entry.get('iteration_dir') or derive_iteration_dir(iteration, model_method)
    directory = os.path.join(model_dir, iteration_dir)
    if not os.path.isdir(directory):
        return None, f"Iteration {iteration} is recorded, but {directory} does not exist."

    metadata_path = os.path.join(directory, 'model_metadata.json')
    metadata, metadata_error = _try_load_json(metadata_path)
    if metadata_error:
        return None, f"Iteration {iteration} is unusable: {metadata_error}"

    metadata_method = normalize_model_method(
        metadata.get('model_method') or metadata.get('weighting_method'),
        metadata.get('is_composite_model'),
    )
    metadata_iteration_dir = metadata.get('iteration_dir')
    metadata_iteration = metadata.get('iteration')

    if metadata_iteration not in (None, iteration):
        return None, (
            f"Iteration {iteration} metadata points to iteration {metadata_iteration}, "
            "so the record is inconsistent."
        )
    if metadata_iteration_dir not in (None, iteration_dir):
        return None, (
            f"Iteration {iteration} metadata points to {metadata_iteration_dir} instead of "
            f"{iteration_dir}."
        )
    if metadata.get('model_method') is not None and metadata_method != model_method:
        return None, (
            f"Iteration {iteration} metadata says {metadata_method}, but history says "
            f"{model_method}."
        )

    is_composite_model = metadata.get('is_composite_model')
    if is_composite_model is None:
        is_composite_model = method_uses_composite(model_method)

    composite_path = os.path.join(directory, 'composite_model.pkl')
    model_path = os.path.join(directory, 'gp_model.pkl')
    scaler_path = os.path.join(directory, 'scaler.pkl')

    if is_composite_model:
        if not os.path.exists(composite_path):
            return None, (
                f"Iteration {iteration} is marked composite, but {composite_path} is missing."
            )
    else:
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, (
                f"Iteration {iteration} is marked standard, but gp_model.pkl/scaler.pkl are missing."
            )

    stamped_metadata = stamp_model_metadata(
        metadata,
        iteration=iteration,
        model_method=model_method,
        iteration_dir=iteration_dir,
        is_composite_model=is_composite_model,
    )
    if 'feature_names' not in stamped_metadata:
        return None, f"Iteration {iteration} metadata is missing feature_names."

    return IterationCandidate(
        iteration=iteration,
        model_method=model_method,
        iteration_dir=iteration_dir,
        is_composite_model=is_composite_model,
        metadata=stamped_metadata,
        directory=directory,
    ), None


def _collect_iteration_candidates(project_root: str, model_dir: str) -> Tuple[List[IterationCandidate], List[str]]:
    """Validate recorded iterations and collect user-facing conflict messages."""
    history_entries, history_error = _load_history_entries(project_root)
    issues: List[str] = []
    if history_error:
        issues.append(history_error)
        return [], issues

    if not history_entries:
        return [], issues

    seen_iterations = set()
    candidates: List[IterationCandidate] = []
    raw_iterations: List[int] = []

    for entry in history_entries:
        iteration = entry.get('iteration')
        if isinstance(iteration, int) and iteration > 0:
            raw_iterations.append(iteration)
        if iteration in seen_iterations:
            issues.append(f"Iteration {iteration} appears multiple times in iteration history.")
            continue
        seen_iterations.add(iteration)

        candidate, issue = _build_iteration_candidate(model_dir, entry)
        if issue:
            issues.append(issue)
            continue
        candidates.append(candidate)

    if raw_iterations:
        highest_recorded = max(raw_iterations)
        highest_valid = max((candidate.iteration for candidate in candidates), default=None)
        if highest_valid == highest_recorded:
            issues = [issue for issue in issues if not issue.startswith('Iteration ')]
        else:
            issues.append(
                f"Latest recorded iteration {highest_recorded} is not loadable, so active "
                "metadata cannot be trusted automatically."
            )

    return candidates, issues


def _describe_candidate(candidate: IterationCandidate) -> str:
    """Human-readable description for prompts and logs."""
    model_kind = 'COMPOSITE' if candidate.is_composite_model else 'STANDARD'
    return (
        f"iteration {candidate.iteration} [{model_kind}; {candidate.model_method}; "
        f"{candidate.iteration_dir}]"
    )


def _prompt_for_iteration_choice(candidates: List[IterationCandidate], issues: List[str]) -> IterationCandidate:
    """Ask the user which recorded iteration should be used after a conflict."""
    print(">>> Active model metadata conflict detected.")
    for issue in issues:
        print(f">>> {issue}")

    if not candidates:
        raise ModelResolutionError(
            "No valid recorded iterations are available. Restore the metadata/history manually."
        )

    print(">>> Available valid iterations:")
    for candidate in sorted(candidates, key=lambda item: item.iteration):
        print(f">>>   {candidate.iteration}: {_describe_candidate(candidate)}")

    try:
        raw_choice = input("Enter the iteration number to use: ").strip()
    except EOFError as exc:
        raise ModelResolutionError(
            "Interactive input is unavailable, so the metadata conflict cannot be resolved safely."
        ) from exc

    if not raw_choice or not raw_choice.lstrip('-').isdigit():
        raise ModelResolutionError(f"Iteration selection '{raw_choice}' is nonsensical.")

    chosen_iteration = int(raw_choice)
    selected = next((candidate for candidate in candidates if candidate.iteration == chosen_iteration), None)
    if selected is None:
        raise ModelResolutionError(
            f"Iteration {chosen_iteration} is nonsensical: no such valid iteration exists."
        )
    return selected


def _load_model_from_candidate(candidate: IterationCandidate):
    """Load one fully validated iteration."""
    if candidate.is_composite_model:
        composite_path = os.path.join(candidate.directory, 'composite_model.pkl')
        with open(composite_path, 'rb') as f:
            gp = pickle.load(f)
        print(f">>> Using COMPOSITE model from {_describe_candidate(candidate)}")
        return gp, None, candidate.metadata, True

    model_path = os.path.join(candidate.directory, 'gp_model.pkl')
    scaler_path = os.path.join(candidate.directory, 'scaler.pkl')
    with open(model_path, 'rb') as f:
        gp = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f">>> Using STANDARD GP model from {_describe_candidate(candidate)}")
    return gp, scaler, candidate.metadata, False


def _load_root_model_without_history(model_dir: str):
    """Use the active root model only when there is no iteration history to consult."""
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    metadata, metadata_error = _try_load_json(metadata_path)
    if metadata_error:
        raise ModelResolutionError(metadata_error)

    if 'feature_names' not in metadata:
        raise ModelResolutionError(f"{metadata_path} is missing feature_names.")

    wants_composite = bool(metadata.get('is_composite_model', False))
    composite_path = os.path.join(model_dir, 'composite_model.pkl')
    model_path = os.path.join(model_dir, 'gp_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')

    if wants_composite:
        if not os.path.exists(composite_path):
            raise ModelResolutionError(
                "Root metadata selects a composite model, but composite_model.pkl is missing. "
                "No automatic fallback will be used."
            )
        with open(composite_path, 'rb') as f:
            gp = pickle.load(f)
        print(">>> Using COMPOSITE model from root metadata (no iteration history found)")
        return gp, None, metadata, True

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise ModelResolutionError(
            "Root metadata selects a standard model, but gp_model.pkl/scaler.pkl are missing."
        )

    with open(model_path, 'rb') as f:
        gp = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(">>> Using STANDARD GP model from root metadata (no iteration history found)")
    return gp, scaler, metadata, False


def load_active_model(project_root: str, model_dir: str):
    """Load the active model with strict iteration-aware conflict resolution."""
    candidates, issues = _collect_iteration_candidates(project_root, model_dir)
    if not candidates:
        if issues:
            raise ModelResolutionError(" ".join(issues))
        return _load_root_model_without_history(model_dir)

    latest_candidate = max(candidates, key=lambda item: item.iteration)
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    root_metadata, metadata_error = _try_load_json(metadata_path)

    conflicts = list(issues)
    if metadata_error:
        conflicts.append(metadata_error)
    else:
        if root_metadata.get('iteration') is None:
            conflicts.append("Root model metadata is missing the iteration field.")
        if root_metadata.get('iteration_dir') is None:
            conflicts.append("Root model metadata is missing the iteration_dir field.")
        if root_metadata.get('model_method') is None:
            conflicts.append("Root model metadata is missing the model_method field.")

        if not conflicts:
            root_method = normalize_model_method(
                root_metadata.get('model_method') or root_metadata.get('weighting_method'),
                root_metadata.get('is_composite_model'),
            )
            if root_metadata.get('iteration') != latest_candidate.iteration:
                conflicts.append(
                    f"Root metadata points to iteration {root_metadata.get('iteration')}, "
                    f"but the latest valid iteration is {latest_candidate.iteration}."
                )
            if root_metadata.get('iteration_dir') != latest_candidate.iteration_dir:
                conflicts.append(
                    f"Root metadata points to {root_metadata.get('iteration_dir')}, but the "
                    f"latest valid directory is {latest_candidate.iteration_dir}."
                )
            if root_method != latest_candidate.model_method:
                conflicts.append(
                    f"Root metadata says {root_method}, but the latest valid iteration says "
                    f"{latest_candidate.model_method}."
                )
            if bool(root_metadata.get('is_composite_model')) != latest_candidate.is_composite_model:
                conflicts.append(
                    "Root metadata disagrees with the latest valid iteration on whether the "
                    "active model is composite."
                )

    if conflicts:
        selected_candidate = _prompt_for_iteration_choice(candidates, conflicts)
        write_metadata_with_notice(
            metadata_path,
            selected_candidate.metadata,
            selected_candidate.iteration,
            selected_candidate.model_method,
            reason='conflict resolution',
        )
        return _load_model_from_candidate(selected_candidate)

    return _load_model_from_candidate(latest_candidate)

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
    
    def __init__(self, gp, scaler: StandardScaler,
                 feature_names: List[str], config: OptimizationConfig = None,
                 is_composite: bool = False):
        """
        Initialize optimizer.
        
        Args:
            gp: Trained Gaussian Process model (or CompositeGP)
            scaler: Feature scaler (unused if is_composite)
            feature_names: List of feature names
            config: Optimization configuration
            is_composite: If True, model handles scaling internally
        """
        self.gp = gp
        self.scaler = scaler
        self.feature_names = feature_names
        self.config = config or OptimizationConfig()
        self.is_composite = is_composite
        
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
        
        # Get prediction
        if self.is_composite:
            mean, std = self.gp.predict(x_reshaped, return_std=True)
        else:
            x_scaled = self.scaler.transform(x_reshaped)
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
        
        if self.is_composite:
            # When using composite model, compute y_best from model predictions
            # Raw y_observed may contain literature values (up to 100%) that the
            # composite model would predict much lower, making EI near-zero everywhere
            y_pred = self.gp.predict(X_observed)
            y_best = np.max(y_pred)
            print(f"Best model-predicted viability: {y_best:.1f}% (raw observed max: {np.max(y_observed):.1f}%)")
        else:
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
            x_reshaped = x.reshape(1, -1)
            if self.is_composite:
                pred_mean, pred_std = self.gp.predict(x_reshaped, return_std=True)
            else:
                x_scaled = self.scaler.transform(x_reshaped)
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
                x_reshaped = x.reshape(1, -1)
                if self.is_composite:
                    pred_mean, pred_std = self.gp.predict(x_reshaped, return_std=True)
                else:
                    x_scaled = self.scaler.transform(x_reshaped)
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
                                      n_candidates: int = 20) -> pd.DataFrame:
        """
        Generate candidates with very low DMSO (<0.5% v/v).
        
        Args:
            X_observed: Observed formulations
            y_observed: Observed viabilities
            n_candidates: Number of candidates
            
        Returns:
            DataFrame with low-DMSO candidates
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
            # Handle both _M (molar) and _pct (percentage) suffixes
            if name.endswith('_pct'):
                clean_name = name.replace('_pct', '')
                conc = row[name]
                parts.append(f"{conc:.1f}% {clean_name}")
            else:
                clean_name = name.replace('_M', '')
                conc = row[name]
                # Format concentration appropriately for molar units
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
        
        for _, row in candidates_df.iterrows():
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
    
    print("\nLoading trained model...")
    try:
        gp, scaler, metadata, is_composite = load_active_model(project_root, model_dir)
    except ModelResolutionError as exc:
        print(f"ERROR: {exc}")
        return
    feature_names = metadata['feature_names']
    print(f"Model loaded with {len(feature_names)} features")
    
    # Load observed data
    print(f"\nLoading observed data from: {data_path}")
    df = pd.read_csv(data_path)
    df = df[df['viability_percent'] <= 100].copy()
    
    # Prepare features (handle both _M and _pct columns)
    ingredient_cols = [c for c in df.columns if c.endswith('_M') or c.endswith('_pct')]
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
    
    optimizer = FormulationOptimizer(gp, scaler, feature_names, config, is_composite=is_composite)
    
    # Generate candidates
    print("\n" + "-" * 40)
    print("Generating Optimized Candidates")
    print("-" * 40)
    
    print("\n1. General optimization (up to 5% DMSO allowed)...")
    general_candidates = optimizer.optimize(X, y, n_candidates=20)
    
    print("\n2. Low-DMSO optimization (<0.5% DMSO)...")
    dmso_free_candidates = optimizer.generate_low_dmso_candidates(X, y, n_candidates=20)
    
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
        os.path.join(output_dir, 'candidates_dmso_free.csv')
    )
    
    # Print top candidates
    print("\n" + "=" * 80)
    print("Top 20 General Candidates")
    print("=" * 80)
    for _, row in general_candidates.head(20).iterrows():
        print(f"\nRank {int(row['rank'])}: Viability = {row['predicted_viability']:.1f}% ± {row['uncertainty']:.1f}%")
        print(f"  DMSO: {row['dmso_percent']:.1f}%, Ingredients: {int(row['n_ingredients'])}")
    
    print("\n" + "=" * 80)
    print("Top 20 Low-DMSO Candidates (<0.5% DMSO)")
    print("=" * 80)
    for _, row in dmso_free_candidates.head(20).iterrows():
        print(f"\nRank {int(row['rank'])}: Viability = {row['predicted_viability']:.1f}% ± {row['uncertainty']:.1f}%")
        print(f"  DMSO: {row['dmso_percent']:.1f}%, Ingredients: {int(row['n_ingredients'])}")
    
    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
