"""
QuBit - Base Result Decoder

Decodes the 0/1 solution vector returned from the quantum solver
for the base {2, 3, 4, 8} candidate set and evaluates the allocation.

Decoding:
    (y0, y1) = (0,0) -> 2-bit
    (y0, y1) = (0,1) -> 3-bit
    (y0, y1) = (1,0) -> 4-bit
    (y0, y1) = (1,1) -> 8-bit

Usage:
    python decode_result.py --model_name llama-7b --solution_file solution.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from qubit_core import load_and_flatten_json_with_names, calculate_final_score

# Decoding map: (y0, y1) -> bit-width
DECODE_MAP = {
    (0, 0): 2,
    (0, 1): 3,
    (1, 0): 4,
    (1, 1): 8,
}


def decode_base_solution(solution_vector, num_layers):
    """Decode a 0/1 solution vector into per-layer bit-width assignments."""
    bits = []
    for i in range(num_layers):
        y0 = solution_vector[i * 2]
        y1 = solution_vector[i * 2 + 1]
        bits.append(DECODE_MAP.get((y0, y1), 8))
    return bits


def main():
    parser = argparse.ArgumentParser(description="Decode base QUBO solution")
    parser.add_argument("--model_name", type=str, default="llama-7b")
    parser.add_argument("--model_info_root", type=str, default="../../model_info")
    parser.add_argument("--solution_file", type=str, default=None,
                        help="Path to JSON file containing the solution vector")
    args = parser.parse_args()

    model_info_path = os.path.join(args.model_info_root, args.model_name)

    print(f"--- Loading context data for '{args.model_name}' ---")

    layer_params, layer_names = load_and_flatten_json_with_names(
        os.path.join(model_info_path, "LayerParams.json")
    )
    sensitivities, _ = load_and_flatten_json_with_names(
        os.path.join(model_info_path, "fisher.json")
    )

    NUM_LAYERS = len(layer_params)
    TARGET_BIT_WIDTHS = [2, 3, 4, 8]
    ORIGINAL_BIT_WIDTH = 16
    ALPHA = 10.0
    COMPRESSION_RATIO = 0.25
    target_avg_bit_width = ORIGINAL_BIT_WIDTH * COMPRESSION_RATIO

    # Load solution vector
    if args.solution_file:
        with open(args.solution_file, "r") as f:
            full_solution_vector = json.load(f)
    else:
        print("ERROR: Please provide --solution_file with the JSON solution vector.")
        print("       Example: python decode_result.py --solution_file solution.json")
        return

    # Decode
    num_decision_vars = NUM_LAYERS * 2
    if len(full_solution_vector) < num_decision_vars:
        print(f"ERROR: Solution vector length ({len(full_solution_vector)}) < "
              f"expected ({num_decision_vars})!")
        return

    solution_vector = full_solution_vector[:num_decision_vars]
    final_assigned_bits = decode_base_solution(solution_vector, NUM_LAYERS)

    # Print allocation
    print("\n" + "=" * 60)
    print("           Final Quantization Allocation")
    print("=" * 60)
    print(f"{'Layer Name':<45} {'Bit-width':<12}")
    print("-" * 60)
    for i in range(NUM_LAYERS):
        print(f"{layer_names[i]:<45} {final_assigned_bits[i]}-bit")
    print("-" * 60)

    # Evaluate
    scores = calculate_final_score(
        final_assigned_bits=final_assigned_bits,
        sensitivities=sensitivities,
        layer_params=layer_params,
        target_bit_widths=TARGET_BIT_WIDTHS,
        original_bit_width=ORIGINAL_BIT_WIDTH,
        alpha=ALPHA,
        target_avg_bit_width=target_avg_bit_width,
    )

    unique_bits, counts = np.unique(final_assigned_bits, return_counts=True)
    print("\n--- Allocation Summary ---")
    for bit, count in zip(unique_bits, counts):
        print(f"  - {bit}-bit assigned to {count} layers")

    print("\n--- Bit-width Allocation List ---")
    print(final_assigned_bits)

    print("\n--- Performance Metrics ---")
    print(f"Total Accuracy Loss (ΔAcc): {scores['Total Accuracy Loss']:.4f}")
    print(f"Final Average Bit-width: {scores['Final Average Bit-width']:.2f} bits "
          f"(target: {target_avg_bit_width:.2f} bits)")

    print("\n--- Composite Score (100-point scale) ---")
    print(f"Accuracy Score:  {scores['Accuracy Score']:.2f} / 100")
    print(f"Memory Score:    {scores['Memory Score']:.2f} / 100")
    print(f"Final Score:     {scores['Final Score']:.2f} / 100")


if __name__ == "__main__":
    main()
