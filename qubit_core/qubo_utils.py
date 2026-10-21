"""
QuBit - QUBO Model Utilities

Core mathematical functions for building QUBO formulations
of the mixed-precision quantization problem.
"""

import numpy as np


def calculate_precision_loss(
    bit: int,
    original_bit_width: int,
    sensitivity: float,
    alpha: float,
) -> float:
    """
    Compute the precision loss (ΔAcc) for a single layer, following the
    exponential sensitivity model from the paper.

    Formula:
        ΔAcc_i = S_i * (exp(-α * b_i / B) - exp(-α)) / (1 - exp(-α))

    where:
        S_i  = layer sensitivity (Fisher information)
        b_i  = assigned bit-width for this layer
        B    = original model bit-width (e.g., 16)
        α    = steepness hyperparameter

    Args:
        bit: Assigned bit-width for this layer.
        original_bit_width: Original model precision (typically 16).
        sensitivity: Layer sensitivity score (e.g., Fisher information).
        alpha: Steepness parameter controlling the loss curve shape.

    Returns:
        Precision loss value (non-negative float).
    """
    denominator = 1.0 - np.exp(-alpha)
    if np.isclose(denominator, 0):
        return np.inf
    numerator = np.exp(-alpha * (bit / original_bit_width)) - np.exp(-alpha)
    return sensitivity * (numerator / denominator)
