"""
QuBit - Base QUBO Generation

Generates the QUBO matrix for the base method using {2, 3, 4, 8} candidate
bit-widths with algebraic encoding (2 binary variables per layer, no auxiliary
variables needed).

Encoding:  b = 2 + 2*y0 + 1*y1 + 3*y0*y1
    (y0, y1) = (0,0) -> 2-bit
    (y0, y1) = (0,1) -> 3-bit
    (y0, y1) = (1,0) -> 4-bit
    (y0, y1) = (1,1) -> 8-bit

Usage:
    python generate_qubo.py --model_name qwen2.5-7b
"""

import argparse
import os
import sys

import kaiwu as kw
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from qubit_core import load_and_flatten_json, calculate_precision_loss


def build_base_qubo(model_name: str, model_info_root: str = "../../model_info"):
    """Build and export the base QUBO matrix for a given model."""
    model_info_path = os.path.join(model_info_root, model_name)

    print(f"--- Step 1: Loading pre-computed layer metrics from '{model_info_path}' ---")

    layer_params = load_and_flatten_json(os.path.join(model_info_path, "LayerParams.json"))
    sensitivities = load_and_flatten_json(os.path.join(model_info_path, "fisher.json"))

    NUM_LAYERS = len(layer_params)
    print(f"Successfully loaded metrics for {NUM_LAYERS} layers.")

    # --- Step 2: Build QUBO model (algebraic penalty formulation) ---
    TARGET_BIT_WIDTHS = [2, 3, 4, 8]
    ORIGINAL_BIT_WIDTH = 16
    ALPHA = 10.0
    COMPRESSION_RATIO = 0.25

    total_params = np.sum(layer_params, dtype=np.float64)
    layer_weights = layer_params / total_params
    target_avg_bit_width = ORIGINAL_BIT_WIDTH * COMPRESSION_RATIO

    print(f"\nCandidate bit-widths: {TARGET_BIT_WIDTHS}")
    print(f"Target average bit-width: {target_avg_bit_width:.2f} bits")

    precision_loss_matrix = np.zeros((NUM_LAYERS, len(TARGET_BIT_WIDTHS)))
    for i in range(NUM_LAYERS):
        for j, bit in enumerate(TARGET_BIT_WIDTHS):
            precision_loss_matrix[i, j] = calculate_precision_loss(
                bit, ORIGINAL_BIT_WIDTH, sensitivities[i], ALPHA
            )

    qubo_model = kw.qubo.QuboModel()
    y = kw.qubo.ndarray((NUM_LAYERS, 2), "y", kw.qubo.Binary)

    all_terms = []
    CONSTRAINT_PENALTY_WEIGHT = 2000.0

    for i in range(NUM_LAYERS):
        y0, y1 = y[i, 0], y[i, 1]
        loss_2, loss_3, loss_4, loss_8 = precision_loss_matrix[i]

        # 1. Precision loss polynomial
        layer_precision_loss = (
            (loss_4 - loss_2) * y0
            + (loss_3 - loss_2) * y1
            + (loss_2 - loss_3 - loss_4 + loss_8) * y0 * y1
            + loss_2
        )
        all_terms.append(layer_precision_loss)

        # 2. Memory constraint algebraic penalty
        wi = layer_weights[i]
        target = target_avg_bit_width
        b_const, b_y0, b_y1, b_y0y1 = 2, 2, 1, 3
        A = wi * (b_const - target)
        B = wi * b_y0
        C = wi * b_y1
        D = wi * b_y0y1
        const_coeff = A**2
        y0_coeff = B**2 + 2 * A * B
        y1_coeff = C**2 + 2 * A * C
        y0y1_coeff = D**2 + 2 * A * D + 2 * B * C + 2 * B * D + 2 * C * D
        layer_memory_penalty = (
            y0_coeff * y0 + y1_coeff * y1 + y0y1_coeff * y0 * y1 + const_coeff
        )
        all_terms.append(CONSTRAINT_PENALTY_WEIGHT * layer_memory_penalty)

    final_objective = kw.qubo.quicksum(all_terms)
    qubo_model.set_objective(final_objective)

    print("\n--- Step 2: QUBO model construction complete ---")

    # --- Step 3: Export QUBO matrix as 8-bit precision CSV ---
    original_qubo_mat = qubo_model.get_qubo_matrix()

    print(f"\nOriginal QUBO matrix range: [{original_qubo_mat.min():.2f}, {original_qubo_mat.max():.2f}]")
    print(f"   Total variables: {original_qubo_mat.shape[0]}")
    if original_qubo_mat.shape[0] == NUM_LAYERS * 2:
        print("✅ Variable count correct — no auxiliary variables introduced!")

    print("\nAdjusting QUBO matrix to 8-bit precision space [-128, 127]...")
    adjusted_qubo_mat = kw.qubo.adjust_qubo_matrix_precision(original_qubo_mat, bit_width=8)
    print(f"Adjusted QUBO matrix range: [{adjusted_qubo_mat.min()}, {adjusted_qubo_mat.max()}]")

    output_csv_path = f"{model_name}_base_qubo_8bit.csv"
    pd.DataFrame(adjusted_qubo_mat).to_csv(output_csv_path, index=False, header=False)

    print(f"\n✅ Successfully generated 8-bit QUBO matrix!")
    print(f"   Saved to: {os.path.abspath(output_csv_path)}")
    print("\nNext step: Upload this CSV to the quantum cloud platform for solving.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate base QUBO matrix for QuBit method")
    parser.add_argument("--model_name", type=str, default="qwen2.5-7b", help="Model name")
    parser.add_argument("--model_info_root", type=str, default="../../model_info",
                        help="Root directory of model_info")
    args = parser.parse_args()
    build_base_qubo(args.model_name, args.model_info_root)
