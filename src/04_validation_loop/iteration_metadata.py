#!/usr/bin/env python3
"""
Shared helpers for iteration-aware model metadata and history management.
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional


STANDARD_METHOD = 'standard'
WEIGHTED_SIMPLE_METHOD = 'weighted_simple'
PRIOR_MEAN_METHOD = 'prior_mean_correction'

METHOD_ALIASES = {
    'standard': STANDARD_METHOD,
    'standard_gp': STANDARD_METHOD,
    'literature_only': STANDARD_METHOD,
    'weighted_simple': WEIGHTED_SIMPLE_METHOD,
    'sample_duplication': WEIGHTED_SIMPLE_METHOD,
    'prior_mean_correction': PRIOR_MEAN_METHOD,
}


def normalize_model_method(method: Optional[str], is_composite_model: Optional[bool] = None) -> str:
    """Normalize stored method names across old and new metadata/history formats."""
    if method:
        normalized = METHOD_ALIASES.get(str(method).strip().lower())
        if normalized:
            return normalized
    if is_composite_model:
        return PRIOR_MEAN_METHOD
    return STANDARD_METHOD


def method_uses_composite(model_method: str) -> bool:
    """Return True when the method requires the composite model artifacts."""
    return normalize_model_method(model_method) == PRIOR_MEAN_METHOD


def derive_iteration_dir(iteration: int, model_method: Optional[str]) -> str:
    """Map an iteration number and method to the expected model subdirectory."""
    normalized = normalize_model_method(model_method)
    if normalized == PRIOR_MEAN_METHOD:
        return f'iteration_{iteration}_prior_mean'
    if normalized == WEIGHTED_SIMPLE_METHOD:
        return f'iteration_{iteration}_weighted_simple'
    return f'iteration_{iteration}'


def stamp_model_metadata(
    metadata: Dict,
    iteration: int,
    model_method: str,
    iteration_dir: Optional[str] = None,
    is_composite_model: Optional[bool] = None,
) -> Dict:
    """Attach explicit iteration identity to model metadata."""
    normalized_method = normalize_model_method(model_method, is_composite_model)
    stamped = dict(metadata)
    stamped['iteration'] = iteration
    stamped['model_method'] = normalized_method
    stamped['iteration_dir'] = iteration_dir or derive_iteration_dir(iteration, normalized_method)
    stamped['is_composite_model'] = method_uses_composite(normalized_method)
    return stamped


def load_iteration_history(project_dir: str) -> List[Dict]:
    """Load the iteration history list, returning an empty list when absent."""
    history_path = os.path.join(project_dir, 'data', 'validation', 'iteration_history.json')
    if not os.path.exists(history_path):
        return []
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history.get('iterations', [])


def append_iteration_history(project_dir: str, iteration_data: Dict):
    """Append one iteration entry to the history file."""
    history_path = os.path.join(project_dir, 'data', 'validation', 'iteration_history.json')
    history = {'iterations': load_iteration_history(project_dir)}
    record = dict(iteration_data)
    record['timestamp'] = datetime.now().isoformat()
    history['iterations'].append(record)

    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Iteration {record['iteration']} logged")


def announce_metadata_overwrite(target_path: str, iteration: int, model_method: str, reason: str):
    """Print a consistent notice before replacing active metadata."""
    print(
        f">>> Overwriting metadata at {target_path} due to {reason}: "
        f"iteration {iteration} ({normalize_model_method(model_method)})"
    )


def write_metadata_with_notice(
    target_path: str,
    metadata: Dict,
    iteration: int,
    model_method: str,
    reason: str,
):
    """Write model metadata and announce the overwrite before and after."""
    announce_metadata_overwrite(target_path, iteration, model_method, reason)
    with open(target_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(
        f">>> Metadata overwritten due to {reason}. "
        f"Active iteration is now {iteration} ({normalize_model_method(model_method)})."
    )


def activate_iteration_artifacts(
    source_dir: str,
    target_dir: str,
    filenames: List[str],
    iteration: int,
    model_method: str,
    reason: str,
):
    """Copy a trained iteration into the active models directory."""
    metadata_target = os.path.join(target_dir, 'model_metadata.json')
    announce_metadata_overwrite(metadata_target, iteration, model_method, reason)

    for filename in filenames:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(target_dir, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)

    print(
        f">>> Metadata overwritten due to {reason}. "
        f"Active iteration is now {iteration} ({normalize_model_method(model_method)})."
    )
