"""Export verified 196-bit, adjacent-GGUF-type int8 hardware subproblems.

Each independent block covers two transformer layers and one tensor shape
class. Exactly half of its tensors are upgraded. This fixes the *global* byte
cost at the corresponding midpoint budget, but restricts the feasible set;
the matrices are not exact QUBOs for the unrestricted global-budget problem.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

__author__ = "Yuchen Li"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = ROOT / "experiments/20261021/tables/qwen25_7b_costs.json"
DEFAULT_OUTPUT = ROOT / "experiments/20261022/int8_adjacent"
CASES = (
    ("B2", "Q3_K", "Q4_K", 1, 243_712),
    ("B4", "Q4_K", "Q5_K", 3, 229_376),
)
GROUPS = (
    ("kv", 1, ("self_attn.k_proj", "self_attn.v_proj")),
    ("qo", 7, ("self_attn.q_proj", "self_attn.o_proj")),
    ("mlp", 37, ("mlp.down_proj", "mlp.gate_proj", "mlp.up_proj")),
)
NAME_PATTERN = re.compile(r"^layers\.(\d+)\.(self_attn\.[a-z]+_proj|mlp\.[a-z]+_proj)$")


@dataclass(frozen=True)
class Variable:
    name: str
    group: str
    weight: int
    base_bytes: int
    upgrade_bytes: int
    base_loss: Decimal
    upgrade_loss: Decimal

    @property
    def gain(self) -> Decimal:
        return self.base_loss - self.upgrade_loss


@dataclass(frozen=True)
class Block:
    group: str
    layer_pair: tuple[int, int]
    indices: tuple[int, ...]
    quota: int
    gain_center: Decimal
    penalty: Decimal


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _round_int8(value: Decimal) -> int:
    result = int(value.to_integral_value(rounding=ROUND_HALF_EVEN))
    if not -128 <= result <= 127:
        raise ValueError(f"Coefficient {value} exceeds signed int8")
    return result


def _matrix_energy(matrix: list[list[int]], bits: tuple[int, ...]) -> int:
    selected = [index for index, bit in enumerate(bits) if bit]
    return sum(matrix[i][j] for i in selected for j in selected if i <= j)


def _ising_4x_energy(matrix: list[list[int]], bits: tuple[int, ...]) -> int:
    n = len(bits)
    spins = [2 * bit - 1 for bit in bits]
    fields = [2 * matrix[i][i] + sum(matrix[min(i, j)][max(i, j)]
                                       for j in range(n) if j != i) for i in range(n)]
    constant = 2 * sum(matrix[i][i] for i in range(n)) + sum(
        matrix[i][j] for i in range(n) for j in range(i + 1, n)
    )
    return (constant + sum(h * s for h, s in zip(fields, spins))
            + sum(matrix[i][j] * spins[i] * spins[j]
                  for i in range(n) for j in range(i + 1, n)))


def _build_variables(raw: dict, lower_type: str, upper_type: str,
                     base_unit: int) -> tuple[list[Variable], list[tuple[str, tuple[int, int], tuple[int, ...]]]]:
    tensors = raw["tensors"]
    by_name = {tensor["name"]: tensor for tensor in tensors}
    if len(tensors) != 196 or len(by_name) != 196:
        raise ValueError("Expected 196 distinct target tensors")
    layer_numbers = set()
    for name in by_name:
        match = NAME_PATTERN.fullmatch(name)
        if match is None:
            raise ValueError(f"Unexpected tensor name: {name}")
        layer_numbers.add(int(match.group(1)))
    if layer_numbers != set(range(28)):
        raise ValueError("Expected transformer layers 0 through 27")

    variables: list[Variable] = []
    blocks: list[tuple[str, tuple[int, int], tuple[int, ...]]] = []
    for group, weight, suffixes in GROUPS:
        for start in range(0, 28, 2):
            indices = []
            for layer in (start, start + 1):
                for suffix in suffixes:
                    name = f"layers.{layer}.{suffix}"
                    tensor = by_name[name]
                    low = tensor["choices"][lower_type]
                    high = tensor["choices"][upper_type]
                    increment = high["bytes"] - low["bytes"]
                    if increment != base_unit * weight:
                        raise ValueError(f"Unexpected measured GGUF cost for {name}: {increment}")
                    variable = Variable(
                        name, group, weight, low["bytes"], high["bytes"],
                        Decimal(str(low["loss"])), Decimal(str(high["loss"])),
                    )
                    if variable.gain <= 0:
                        raise ValueError(f"Upgrade proxy gain must be positive: {name}")
                    indices.append(len(variables))
                    variables.append(variable)
            blocks.append((group, (start, start + 1), tuple(indices)))
    if {variable.name for variable in variables} != set(by_name):
        raise ValueError("Hardware blocks do not cover the entire target inventory")
    return variables, blocks


def _raw_block_coefficients(variables: list[Variable], indices: tuple[int, ...],
                            center: Decimal, penalty: Decimal) -> tuple[list[Decimal], Decimal]:
    quota = len(indices) // 2
    diagonal = [-(variables[index].gain - center)
                + penalty * (1 - 2 * quota) for index in indices]
    return diagonal, 2 * penalty


def _build_blocks(variables: list[Variable], block_layout: list[tuple[str, tuple[int, int], tuple[int, ...]]]
                  ) -> tuple[list[Block], Decimal]:
    initial: list[Block] = []
    for group, layer_pair, indices in block_layout:
        gains = [variables[index].gain for index in indices]
        center = (min(gains) + max(gains)) / 2
        penalty = max(abs(gain - center) for gain in gains) + Decimal(1)
        initial.append(Block(group, layer_pair, indices, len(indices) // 2, center, penalty))

    max_coefficient = max(
        max(*(abs(value) for value in _raw_block_coefficients(
            variables, block.indices, block.gain_center, block.penalty)[0]),
            _raw_block_coefficients(variables, block.indices, block.gain_center, block.penalty)[1])
        for block in initial
    )
    initial_scale = max_coefficient / Decimal(127)
    # An extremely small block otherwise loses its whole quota penalty at int8.
    penalty_floor = 2 * initial_scale
    blocks = [
        Block(block.group, block.layer_pair, block.indices, block.quota,
              block.gain_center, max(block.penalty, penalty_floor))
        for block in initial
    ]
    maximum = max(
        max(*(abs(value) for value in _raw_block_coefficients(
            variables, block.indices, block.gain_center, block.penalty)[0]),
            _raw_block_coefficients(variables, block.indices, block.gain_center, block.penalty)[1])
        for block in blocks
    )
    return blocks, maximum / Decimal(127)


def _best_unrestricted(variables: list[Variable], capacity_units: int) -> dict:
    """Exact two-choice MILP optimum via three cost classes and positive gains."""
    groups = {weight: sorted((variable.gain for variable in variables if variable.weight == weight),
                             reverse=True) for weight in (1, 7, 37)}
    prefixes: dict[int, list[Decimal]] = {}
    for weight, gains in groups.items():
        prefix = [Decimal(0)]
        for gain in gains:
            prefix.append(prefix[-1] + gain)
        prefixes[weight] = prefix
    best_gain = Decimal("-Infinity")
    best_counts = None
    for k1 in range(len(groups[1]) + 1):
        for k7 in range(len(groups[7]) + 1):
            residual = capacity_units - k1 - 7 * k7
            if residual < 0:
                continue
            k37 = min(len(groups[37]), residual // 37)
            gain = prefixes[1][k1] + prefixes[7][k7] + prefixes[37][k37]
            if gain > best_gain:
                best_gain = gain
                best_counts = (k1, k7, k37)
    if best_counts is None:
        raise ValueError("No feasible two-choice allocation")
    return {
        "method": "exact count enumeration of the two-choice global-budget MILP",
        "upgrade_counts_by_cost_weight_1_7_37": list(best_counts),
        "upgrade_cost_units": best_counts[0] + 7 * best_counts[1] + 37 * best_counts[2],
        "proxy_gain": str(best_gain),
    }


def generate_case(table_path: Path, output_dir: Path, case: tuple[str, str, str, int, int]
                  ) -> tuple[Path, Path]:
    label, lower_type, upper_type, budget_index, base_unit = case
    raw = json.loads(table_path.read_text(encoding="utf-8"), parse_float=Decimal)
    variables, layout = _build_variables(raw, lower_type, upper_type, base_unit)
    blocks, scale = _build_blocks(variables, layout)
    n = len(variables)
    if n != 196 or len(blocks) != 42:
        raise AssertionError("Expected 196 bits in 42 two-layer blocks")
    budget = raw["budgets"][budget_index]
    fixed_bytes = raw["metadata"]["fixed_non_target_bytes"]
    full_budget = raw["metadata"]["uniform_full_file_budget_bytes"][budget_index]
    baseline_bytes = sum(variable.base_bytes for variable in variables)
    capacity_units, remainder = divmod(budget - baseline_bytes, base_unit)
    if remainder or capacity_units != 1778 or full_budget != budget + fixed_bytes:
        raise ValueError("Midpoint budget or fixed GGUF byte accounting has changed")
    if sum(block.quota * variables[block.indices[0]].weight for block in blocks) != capacity_units:
        raise AssertionError("Block quotas do not reproduce the global budget")

    matrix = [[0] * n for _ in range(n)]
    block_checks = []
    selected_representative = [0] * n
    proxy_gain_best = Decimal(0)
    proxy_gain_worst = Decimal(0)
    proxy_gain_representative = Decimal(0)
    restricted_gain = Decimal(0)
    ground_state_degeneracy = 1
    for block in blocks:
        indices = block.indices
        diagonal, off_diagonal = _raw_block_coefficients(
            variables, indices, block.gain_center, block.penalty)
        for local_index, global_index in enumerate(indices):
            matrix[global_index][global_index] = _round_int8(diagonal[local_index] / scale)
            for following in indices[local_index + 1:]:
                matrix[global_index][following] = _round_int8(off_diagonal / scale)
        local_matrix = [[matrix[i][j] for j in indices] for i in indices]
        states = []
        for bits in itertools.product((0, 1), repeat=len(indices)):
            qubo_energy = _matrix_energy(local_matrix, bits)
            if _ising_4x_energy(local_matrix, bits) != 4 * qubo_energy:
                raise AssertionError("QUBO-to-Ising energy identity failed")
            gain = sum((variables[index].gain for index, bit in zip(indices, bits) if bit), Decimal(0))
            states.append((qubo_energy, bits, gain))
        minimum = min(energy for energy, _, _ in states)
        optima = [state for state in states if state[0] == minimum]
        if any(sum(bits) != block.quota for _, bits, _ in optima):
            raise ValueError(f"{label} {block.group} {block.layer_pair}: int8 ground state violates its quota")
        representative = min(optima, key=lambda item: item[1])
        for index, bit in zip(indices, representative[1]):
            selected_representative[index] = bit
        proxy_gain_best += max(state[2] for state in optima)
        proxy_gain_worst += min(state[2] for state in optima)
        proxy_gain_representative += representative[2]
        true_block_gain = sum(sorted((variables[index].gain for index in indices),
                                     reverse=True)[:block.quota], Decimal(0))
        restricted_gain += true_block_gain
        ground_state_degeneracy *= len(optima)
        block_checks.append({
            "group": block.group,
            "layer_pair": list(block.layer_pair),
            "indices": list(indices),
            "quota": block.quota,
            "gain_center": str(block.gain_center),
            "raw_penalty": str(block.penalty),
            "exhaustive_binary_states": len(states),
            "int8_ground_state_count": len(optima),
            "all_int8_ground_states_satisfy_quota": True,
            "restricted_exact_proxy_gain": str(true_block_gain),
            "int8_ground_state_proxy_gain_min": str(min(state[2] for state in optima)),
            "int8_ground_state_proxy_gain_max": str(max(state[2] for state in optima)),
        })

    fields_4x = [2 * matrix[i][i] + sum(matrix[min(i, j)][max(i, j)]
                                           for j in range(n) if i != j) for i in range(n)]
    couplings = [matrix[i][j] for i in range(n) for j in range(i + 1, n) if matrix[i][j]]
    if max(abs(value) for value in fields_4x) > 127 or max(abs(value) for value in couplings) > 127:
        raise ValueError("QUBO-to-Ising 4x fields/couplings exceed signed int8")
    if any(value < -128 or value > 127 for row in matrix for value in row):
        raise AssertionError("Export matrix is not signed int8")

    representative_units = sum(variable.weight * bit for variable, bit in zip(variables, selected_representative))
    if representative_units != capacity_units:
        raise AssertionError("Representative state exceeds the measured-byte budget")
    unrestricted = _best_unrestricted(variables, capacity_units)
    base_loss = sum((variable.base_loss for variable in variables), Decimal(0))
    baseline_bits = [0] * n
    for block in blocks:
        for index in block.indices[:block.quota]:
            baseline_bits[index] = 1
    if sum(variable.weight * bit for variable, bit in zip(variables, baseline_bits)) != capacity_units:
        raise AssertionError("Neutral baseline does not meet the measured-byte budget")

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / f"qwen25_7b_{label.lower()}_{lower_type.lower()}_{upper_type.lower()}_196bit_int8.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(matrix)
    metadata = {
        "author": __author__,
        "purpose": "Hardware-oriented signed-int8 adjacent-type allocation subproblem",
        "scope_warning": "This 196-bit block-quota QUBO is a restricted feasible subset of the global-budget problem, not an exact full global-budget QUBO; no physical solve or acceptance is implied.",
        "hardware_acceptance": "unverified",
        "case": label,
        "model": raw["metadata"].get("model"),
        "types": {"zero": lower_type, "one": upper_type},
        "source_cost_table": str(table_path.resolve()),
        "source_cost_table_sha256": _sha256(table_path),
        "matrix_file": matrix_path.name,
        "matrix_sha256": _sha256(matrix_path),
        "matrix_format": "Headerless square upper-triangular signed-int8 CSV; E = x.T @ Q @ x",
        "decision_bit_count": n,
        "auxiliary_bit_count": 0,
        "block_count": len(blocks),
        "nonzero_qubo_couplings": len(couplings),
        "budget_target_bytes": budget,
        "fixed_non_target_bytes": fixed_bytes,
        "budget_full_file_bytes": full_budget,
        "base_target_bytes": baseline_bytes,
        "upgrade_unit_bytes": base_unit,
        "upgrade_capacity_units": capacity_units,
        "baseline": {
            "construction": "For each two-layer block, upgrade every tensor in the first layer and keep the second layer at the base type.",
            "upgrade_counts_by_group": {group: sum(
                baseline_bits[i] for i, variable in enumerate(variables) if variable.group == group)
                for group, _, _ in GROUPS},
            "target_bytes": baseline_bytes + base_unit * capacity_units,
            "full_file_bytes": baseline_bytes + base_unit * capacity_units + fixed_bytes,
        },
        "formulation": {
            "raw_block_qubo": "-sum_i((gain_i - center_b)*x_i) + penalty_b*(sum_i(x_i)-quota_b)^2; one block per shape group and two transformer layers",
            "centered_byte_residual": "increment_bytes_b*(sum_i(x_i)-quota_b)",
            "gain_definition": "measured lower-type squared-weight-error minus measured upper-type squared-weight-error",
            "gain_center_rule": "midpoint of the minimum and maximum measured gains in the block",
            "penalty_rule": "max_i abs(gain_i-center_b)+1, floored at twice the initial global int8 scale",
            "global_int8_scale": str(scale),
            "rounding_rule": "round-half-even(raw QUBO coefficient/global_int8_scale)",
            "interpretation": "Proxy objective matches the original only within each fixed-quota block; quota constraints remove cross-block byte trades.",
        },
        "variables": [{
            "index": index, "tensor_name": variable.name, "group": variable.group,
            "zero_type": lower_type, "one_type": upper_type,
            "zero_bytes": variable.base_bytes, "one_bytes": variable.upgrade_bytes,
            "upgrade_bytes": variable.upgrade_bytes - variable.base_bytes,
            "zero_proxy_loss": str(variable.base_loss),
            "one_proxy_loss": str(variable.upgrade_loss),
            "proxy_gain": str(variable.gain),
        } for index, variable in enumerate(variables)],
        "blocks": block_checks,
        "ising_for_4x_qubo_energy": {
            "spin_mapping": "s_i = 2*x_i - 1",
            "fields": fields_4x,
            "coupling_rule": "J_ij = upper-triangular Q_ij",
            "max_abs_field": max(abs(value) for value in fields_4x),
            "max_abs_coupling": max(abs(value) for value in couplings),
            "all_fields_and_couplings_signed_int8": True,
            "constant": 2 * sum(matrix[i][i] for i in range(n)) + sum(couplings),
        },
        "exact_local_checks": {
            "states_enumerated": sum(block["exhaustive_binary_states"] for block in block_checks),
            "all_ground_states_within_measured_budget": True,
            "ground_state_degeneracy": str(ground_state_degeneracy),
            "representative_bits": selected_representative,
            "representative_target_bytes": budget,
            "representative_full_file_bytes": full_budget,
            "representative_upgrade_counts_by_group": {
                group: sum(selected_representative[i] for i, variable in enumerate(variables)
                           if variable.group == group)
                for group, _, _ in GROUPS
            },
            "representative_proxy_loss": str(base_loss - proxy_gain_representative),
            "worst_ground_state_proxy_loss": str(base_loss - proxy_gain_worst),
            "best_ground_state_proxy_loss": str(base_loss - proxy_gain_best),
            "restricted_exact_proxy_loss": str(base_loss - restricted_gain),
            "unrestricted_two_choice_proxy_loss": str(base_loss - Decimal(unrestricted["proxy_gain"])),
            "unrestricted_two_choice_optimum": unrestricted,
        },
    }
    metadata_path = matrix_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return matrix_path, metadata_path


def decode_vector(metadata: dict, vector: list[int]) -> dict:
    """Decode a returned 196-bit state using measured GGUF bytes and proxy loss."""
    variables = metadata["variables"]
    if len(vector) != len(variables) or any(type(bit) is not int or bit not in (0, 1) for bit in vector):
        raise ValueError(f"Expected {len(variables)} binary values")
    target_bytes = sum(variable["one_bytes"] if bit else variable["zero_bytes"]
                       for variable, bit in zip(variables, vector))
    proxy_loss = sum((Decimal(variable["one_proxy_loss"] if bit else variable["zero_proxy_loss"])
                      for variable, bit in zip(variables, vector)), Decimal(0))
    block_violations = [{
        "group": block["group"], "layer_pair": block["layer_pair"],
        "selected_upgrades": sum(vector[index] for index in block["indices"]),
        "required_upgrades": block["quota"],
    } for block in metadata["blocks"]
        if sum(vector[index] for index in block["indices"]) != block["quota"]]
    return {
        "author": __author__,
        "target_bytes": target_bytes,
        "full_file_bytes": target_bytes + metadata["fixed_non_target_bytes"],
        "budget_target_bytes": metadata["budget_target_bytes"],
        "within_global_budget": target_bytes <= metadata["budget_target_bytes"],
        "satisfies_block_quotas": not block_violations,
        "block_violations": block_violations,
        "proxy_loss": str(proxy_loss),
        "allocation": {variable["tensor_name"]: variable["one_type"] if bit else variable["zero_type"]
                       for variable, bit in zip(variables, vector)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for case in CASES:
        for path in generate_case(args.table, args.output_dir, case):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
