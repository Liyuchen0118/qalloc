"""
QuBit - Evaluation Metrics

Provides a unified scoring function to evaluate the quality of a
bit-width allocation plan using accuracy and memory convergence scores.
"""

import numpy as np
from typing import List, Dict

from .qubo_utils import calculate_precision_loss


def calculate_final_score(
    final_assigned_bits: List[int],
    sensitivities: np.ndarray,
    layer_params: np.ndarray,
    target_bit_widths: List[int],
    original_bit_width: int,
    alpha: float,
    target_avg_bit_width: float,
    w_acc: float = 0.7,
    w_mem: float = 0.3,
    strictness: float = 1.0,
) -> Dict[str, float]:
    """
    Compute a 100-point composite score for a given bit-width allocation.

    The score combines two sub-scores:
      - Accuracy Score: how close the allocation's precision loss is to
        the theoretical minimum (all layers at max bit-width).
      - Memory Score: how close the weighted average bit-width is to the
        target average bit-width.

    Args:
        final_assigned_bits: List of assigned bit-widths, one per layer.
        sensitivities: Array of layer sensitivity values (Fisher info).
        layer_params: Array of parameter counts per layer.
        target_bit_widths: List of candidate bit-width options.
        original_bit_width: Original model precision (e.g., 16).
        alpha: Steepness parameter for precision loss.
        target_avg_bit_width: Desired weighted-average bit-width.
        w_acc: Weight for the accuracy score component (default 0.7).
        w_mem: Weight for the memory score component (default 0.3).
        strictness: Exponent applied to the normalized accuracy score
                    (higher = more penalty for suboptimal allocations).

    Returns:
        Dictionary containing:
          - "Accuracy Score": 0-100 accuracy sub-score
          - "Memory Score": 0-100 memory convergence sub-score
          - "Final Score": weighted composite score
          - "Total Accuracy Loss": raw precision loss sum
          - "Final Average Bit-width": actual weighted-average bit-width
    """
    num_layers = len(layer_params)

    # Compute worst-case (all min-bit) and best-case (all max-bit) losses
    loss_max = sum(
        calculate_precision_loss(
            min(target_bit_widths), original_bit_width, sensitivities[i], alpha
        )
        for i in range(num_layers)
    )
    loss_min = sum(
        calculate_precision_loss(
            max(target_bit_widths), original_bit_width, sensitivities[i], alpha
        )
        for i in range(num_layers)
    )

    # Current allocation's precision loss
    current_loss = sum(
        calculate_precision_loss(
            final_assigned_bits[i], original_bit_width, sensitivities[i], alpha
        )
        for i in range(num_layers)
    )

    # Accuracy score (0-100)
    if (loss_max - loss_min) > 1e-6:
        normalized_loss = (current_loss - loss_min) / (loss_max - loss_min)
        score_acc = 100.0 * (1 - normalized_loss) ** strictness
    else:
        score_acc = 100.0

    # Memory convergence score (0-100)
    actual_avg_bit_width = np.average(final_assigned_bits, weights=layer_params)
    delta_b = abs(actual_avg_bit_width - target_avg_bit_width)
    delta_b_max = max(
        abs(max(target_bit_widths) - target_avg_bit_width),
        abs(min(target_bit_widths) - target_avg_bit_width),
    )
    if delta_b_max > 1e-6:
        score_mem = 100.0 * max(0, 1 - (delta_b / delta_b_max))
    else:
        score_mem = 100.0

    # Composite score
    final_score = w_acc * score_acc + w_mem * score_mem

    return {
        "Accuracy Score": score_acc,
        "Memory Score": score_mem,
        "Final Score": final_score,
        "Total Accuracy Loss": current_loss,
        "Final Average Bit-width": actual_avg_bit_width,
    }
