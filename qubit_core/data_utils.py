"""
QuBit - Shared Data Utilities

Provides common data loading and processing functions used across
all QUBO formulation, baseline, and evaluation scripts.
"""

import json
import numpy as np
from typing import Tuple, List

# Canonical ordering of sub-layer weight matrices within each Transformer block.
# This ordering must be consistent across QUBO generation and result decoding.
PARAM_ORDER = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def load_and_flatten_json(file_path: str) -> np.ndarray:
    """
    Load a JSON file containing per-layer metrics and flatten into a 1-D array.

    The JSON file is expected to have the structure:
        { "layer_idx": { "param_name": value, ... }, ... }
    or wrapped in a single top-level key:
        { "some_key": { "layer_idx": { ... }, ... } }

    Values are flattened in order of PARAM_ORDER within each layer,
    and layers are sorted by their integer index.

    Args:
        file_path: Path to the JSON file.

    Returns:
        1-D numpy array of flattened values.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    top_level_keys = list(json_data.keys())
    if len(top_level_keys) == 1:
        data_to_process = json_data[top_level_keys[0]]
    else:
        data_to_process = json_data

    flat_data = []
    for layer_idx in sorted(data_to_process.keys(), key=int):
        layer_block = data_to_process[layer_idx]
        for param_name in PARAM_ORDER:
            if param_name in layer_block:
                flat_data.append(layer_block[param_name])
    return np.array(flat_data)


def load_and_flatten_json_with_names(
    file_path: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Same as load_and_flatten_json, but also returns human-readable layer names.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Tuple of (values_array, names_list) where names_list contains
        strings like "layers.0.self_attn.q_proj".
    """
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    top_level_keys = list(json_data.keys())
    if len(top_level_keys) == 1:
        data_to_process = json_data[top_level_keys[0]]
    else:
        data_to_process = json_data

    flat_data = []
    flat_names = []
    for layer_idx in sorted(data_to_process.keys(), key=int):
        layer_block = data_to_process[layer_idx]
        for param_name in PARAM_ORDER:
            if param_name in layer_block:
                flat_data.append(layer_block[param_name])
                flat_names.append(f"layers.{layer_idx}.{param_name}")
    return np.array(flat_data), flat_names
