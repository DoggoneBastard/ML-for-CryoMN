#!/usr/bin/env python3
"""
Shared prediction calibration utilities.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np


DEFAULT_BIAS_SHIFT_PERCENT = 0.0
DEFAULT_UNCERTAINTY_SCALE = 1.0


def calibration_from_metadata(metadata: Mapping[str, object] | None) -> Dict[str, float]:
    """Extract calibration parameters from model metadata with safe defaults."""
    if metadata is None:
        return {
            "bias_shift_percent": DEFAULT_BIAS_SHIFT_PERCENT,
            "uncertainty_scale": DEFAULT_UNCERTAINTY_SCALE,
        }

    bias_shift = float(metadata.get("bias_shift_percent", DEFAULT_BIAS_SHIFT_PERCENT))
    uncertainty_scale = float(metadata.get("uncertainty_scale", DEFAULT_UNCERTAINTY_SCALE))
    if not np.isfinite(uncertainty_scale) or uncertainty_scale <= 0.0:
        uncertainty_scale = DEFAULT_UNCERTAINTY_SCALE

    return {
        "bias_shift_percent": bias_shift,
        "uncertainty_scale": uncertainty_scale,
    }


def apply_prediction_calibration(
    mean: np.ndarray,
    std: np.ndarray,
    metadata: Mapping[str, object] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply metadata-driven post-hoc calibration to predictions.

    Mean shift is additive in viability percentage points.
    Uncertainty scaling is multiplicative and clipped non-negative.
    """
    calibrated = calibration_from_metadata(metadata)
    mean_arr = np.asarray(mean, dtype=float) + calibrated["bias_shift_percent"]
    std_arr = np.asarray(std, dtype=float) * calibrated["uncertainty_scale"]
    std_arr = np.maximum(std_arr, 0.0)
    return mean_arr, std_arr

