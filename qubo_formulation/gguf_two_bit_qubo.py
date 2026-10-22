"""Export exact two-decision-bit GGUF allocation QUBOs.

The type mapping per tensor is 00=Q3_K, 01=Q4_K, 10=Q5_K, 11=Q6_K,
where the first displayed bit is b1 and the second is b0. This construction
requires c(Q3_K) + c(Q6_K) == c(Q4_K) + c(Q5_K) for every tensor. It exports
the historical headerless upper-triangle CSV format, energy x.T @ Q @ x.
These unrounded coefficients are a mathematical reference, not an int8 upload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal, getcontext
from functools import reduce
from pathlib import Path
from typing import Any

__author__ = "Yuchen Li"
getcontext().prec = 80

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiments/20261022/q6_data/qwen25_7b_q3_q4_q5_q6_costs.json"
DEFAULT_OUTPUT = ROOT / "experiments/20261022/two_bit_qubo"
TYPE_ORDER = ("Q3_K", "Q4_K", "Q5_K", "Q6_K")
TYPE_BITS = {"Q3_K": (0, 0), "Q4_K": (1, 0), "Q5_K": (0, 1), "Q6_K": (1, 1)}
MAX_VARIABLES = 1000


@dataclass(frozen=True)
class Choice:
    tensor_type: str
    byte_count: int
    loss: Decimal


@dataclass(frozen=True)
class Tensor:
    name: str
    choices: tuple[Choice, ...]

    def choice(self, tensor_type: str) -> Choice:
        return self.choices[TYPE_ORDER.index(tensor_type)]


@dataclass(frozen=True)
class Problem:
    tensors: tuple[Tensor, ...]
    budgets: tuple[int, ...]
    metadata: dict[str, Any]

    @property
    def minimum_bytes(self) -> int:
        return sum(tensor.choice("Q3_K").byte_count for tensor in self.tensors)


@dataclass(frozen=True)
class Variable:
    kind: str
    tensor_index: int | None
    tensor_name: str | None
    bit: str | None
    budget_weight_units: int
    loss_linear: Decimal


@dataclass(frozen=True)
class QuboSpec:
    problem: Problem
    budget_index: int
    budget_bytes: int
    fixed_bytes: int
    byte_unit: int
    capacity_units: int
    base_loss: Decimal
    loss_interactions: tuple[Decimal, ...]
    budget_penalty: int
    variables: tuple[Variable, ...]

    @property
    def offset(self) -> Decimal:
        return self.base_loss + Decimal(self.budget_penalty * self.capacity_units**2)


def load_problem(path: Path) -> Problem:
    """Read a measured four-type cost table without the older Q8-only loader."""
    raw = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
    if not isinstance(raw, dict) or not isinstance(raw.get("tensors"), list) or not raw["tensors"]:
        raise ValueError("Input must contain a non-empty tensors array")
    if not isinstance(raw.get("budgets"), list) or not raw["budgets"]:
        raise ValueError("Input must contain a non-empty budgets array")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")

    tensors: list[Tensor] = []
    names: set[str] = set()
    for item in raw["tensors"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
            raise ValueError("Each tensor needs a non-empty name")
        name = item["name"]
        if name in names:
            raise ValueError(f"Duplicate tensor name: {name}")
        names.add(name)
        choices_raw = item.get("choices")
        if not isinstance(choices_raw, dict) or set(choices_raw) != set(TYPE_ORDER):
            raise ValueError(f"{name}: expected exactly {TYPE_ORDER}")
        choices: list[Choice] = []
        for tensor_type in TYPE_ORDER:
            value = choices_raw[tensor_type]
            if not isinstance(value, dict):
                raise ValueError(f"{name}/{tensor_type}: expected bytes and loss")
            byte_count = value.get("bytes")
            loss_raw = value.get("loss")
            if type(byte_count) is not int or byte_count <= 0:
                raise ValueError(f"{name}/{tensor_type}: bytes must be positive integer")
            if isinstance(loss_raw, bool) or not isinstance(loss_raw, (int, float, Decimal)):
                raise ValueError(f"{name}/{tensor_type}: loss must be numeric")
            loss = Decimal(str(loss_raw))
            if not loss.is_finite() or loss < 0:
                raise ValueError(f"{name}/{tensor_type}: loss must be finite and nonnegative")
            choices.append(Choice(tensor_type, byte_count, loss))
        costs = [choice.byte_count for choice in choices]
        if costs != sorted(costs) or len(set(costs)) != 4:
            raise ValueError(f"{name}: Q3/Q4/Q5/Q6 byte costs must strictly increase")
        if costs[0] + costs[3] != costs[1] + costs[2]:
            raise ValueError(f"{name}: actual byte costs violate c3+c6=c4+c5")
        tensors.append(Tensor(name, tuple(choices)))

    budgets = raw["budgets"]
    if any(type(value) is not int or value < 0 for value in budgets):
        raise ValueError("Budgets must be nonnegative integer target-byte counts")
    return Problem(tuple(tensors), tuple(budgets), metadata)


def bounded_binary_weights(capacity: int) -> tuple[int, ...]:
    """Cover every integer slack from zero through capacity."""
    if capacity < 0:
        raise ValueError("Capacity must be nonnegative")
    if capacity == 0:
        return ()
    weights = [1 << index for index in range(capacity.bit_length() - 1)]
    weights.append(capacity - sum(weights))
    return tuple(weights)


def byte_unit_for(problem: Problem) -> int:
    deltas = [
        choice.byte_count - tensor.choice("Q3_K").byte_count
        for tensor in problem.tensors for choice in tensor.choices
    ]
    deltas.extend(budget - problem.minimum_bytes for budget in problem.budgets)
    return reduce(math.gcd, deltas, 0) or 1


def build_spec(problem: Problem, budget_index: int, max_variables: int = MAX_VARIABLES) -> QuboSpec:
    """Construct an exact global-budget QUBO with 2N decision bits."""
    budget = problem.budgets[budget_index]
    if budget < problem.minimum_bytes:
        raise ValueError("Target budget is below the all-Q3_K cost")
    fixed_bytes = problem.metadata.get("fixed_non_target_bytes", 0)
    if type(fixed_bytes) is not int or fixed_bytes < 0:
        raise ValueError("fixed_non_target_bytes must be nonnegative integer")
    full_budgets = problem.metadata.get("uniform_full_file_budget_bytes")
    if full_budgets is not None and full_budgets[budget_index] != budget + fixed_bytes:
        raise ValueError("Target and fixed bytes do not match the full-file budget")
    unit = byte_unit_for(problem)
    capacity = (budget - problem.minimum_bytes) // unit
    variables: list[Variable] = []
    interactions: list[Decimal] = []
    base_loss = Decimal(0)
    for index, tensor in enumerate(problem.tensors):
        if tuple(choice.tensor_type for choice in tensor.choices) != TYPE_ORDER:
            raise ValueError(f"{tensor.name}: expected choices in {TYPE_ORDER} order")
        q3, q4, q5, q6 = tensor.choices
        if q3.byte_count + q6.byte_count != q4.byte_count + q5.byte_count:
            raise ValueError(f"{tensor.name}: actual byte costs violate c3+c6=c4+c5")
        if not (q3.byte_count < q4.byte_count < q5.byte_count < q6.byte_count):
            raise ValueError(f"{tensor.name}: Q3/Q4/Q5/Q6 byte costs must strictly increase")
        base_loss += q3.loss
        variables.extend((
            Variable("decision", index, tensor.name, "b0", (q4.byte_count - q3.byte_count) // unit,
                     q4.loss - q3.loss),
            Variable("decision", index, tensor.name, "b1", (q5.byte_count - q3.byte_count) // unit,
                     q5.loss - q3.loss),
        ))
        interactions.append(q6.loss - q5.loss - q4.loss + q3.loss)
    variables.extend(Variable("slack", None, None, None, weight, Decimal(0))
                     for weight in bounded_binary_weights(capacity))
    if len(variables) > max_variables:
        raise ValueError(f"QUBO requires {len(variables)} variables; limit is {max_variables}")
    return QuboSpec(problem, budget_index, budget, fixed_bytes, unit, capacity, base_loss,
                    tuple(interactions), math.floor(base_loss) + 1, tuple(variables))


def coefficient(spec: QuboSpec, row: int, column: int) -> Decimal:
    """Return one upper-triangle entry of the raw exact Q matrix."""
    if row > column:
        return Decimal(0)
    left, right = spec.variables[row], spec.variables[column]
    a, b = left.budget_weight_units, right.budget_weight_units
    if row == column:
        return left.loss_linear + spec.budget_penalty * (a * a - 2 * spec.capacity_units * a)
    result = Decimal(2 * spec.budget_penalty * a * b)
    if left.kind == right.kind == "decision" and left.tensor_index == right.tensor_index:
        result += spec.loss_interactions[left.tensor_index]
    return result


def encode_allocation(spec: QuboSpec, allocation: dict[str, str]) -> list[int]:
    """Encode a feasible allocation and exact unused-budget slack."""
    if set(allocation) != {tensor.name for tensor in spec.problem.tensors}:
        raise ValueError("Allocation tensor names do not match the source table")
    vector = [0] * len(spec.variables)
    used_bytes = 0
    for index, tensor in enumerate(spec.problem.tensors):
        tensor_type = allocation[tensor.name]
        if tensor_type not in TYPE_BITS:
            raise ValueError(f"Unknown GGUF type: {tensor_type}")
        vector[2 * index:2 * index + 2] = TYPE_BITS[tensor_type]
        used_bytes += tensor.choice(tensor_type).byte_count
    remaining = spec.budget_bytes - used_bytes
    if remaining < 0 or remaining % spec.byte_unit:
        raise ValueError("Allocation exceeds budget or has non-integral slack")
    remaining //= spec.byte_unit
    start = 2 * len(spec.problem.tensors)
    for index in sorted(range(start, len(vector)),
                        key=lambda item: spec.variables[item].budget_weight_units, reverse=True):
        weight = spec.variables[index].budget_weight_units
        if weight <= remaining:
            vector[index] = 1
            remaining -= weight
    if remaining:
        raise AssertionError("Bounded binary slack could not encode unused capacity")
    return vector


def inspect_vector(spec: QuboSpec, vector: list[int]) -> dict[str, Any]:
    """Decode the four types and recompute actual byte use and proxy loss."""
    if len(vector) != len(spec.variables) or any(bit not in (0, 1) for bit in vector):
        raise ValueError(f"Expected {len(spec.variables)} binary bits")
    allocation: dict[str, str] = {}
    target_bytes = 0
    proxy_loss = Decimal(0)
    type_for_bits = {bits: tensor_type for tensor_type, bits in TYPE_BITS.items()}
    for index, tensor in enumerate(spec.problem.tensors):
        tensor_type = type_for_bits[tuple(vector[2 * index:2 * index + 2])]
        choice = tensor.choice(tensor_type)
        allocation[tensor.name] = tensor_type
        target_bytes += choice.byte_count
        proxy_loss += choice.loss
    slack_units = sum(variable.budget_weight_units * bit for variable, bit in zip(spec.variables, vector)
                      if variable.kind == "slack")
    residual = sum(variable.budget_weight_units * bit for variable, bit in zip(spec.variables, vector))
    residual -= spec.capacity_units
    return {
        "author": __author__, "allocation": allocation, "target_bytes": target_bytes,
        "full_file_bytes": target_bytes + spec.fixed_bytes, "budget_bytes": spec.budget_bytes,
        "within_budget": target_bytes <= spec.budget_bytes,
        "qubo_constraints_satisfied": residual == 0,
        "proxy_loss": str(proxy_loss), "slack_units": slack_units,
        "budget_residual_units": residual,
    }


def direct_energy(spec: QuboSpec, vector: list[int]) -> Decimal:
    """Evaluate the semantic loss plus exact budget penalty."""
    decoded = inspect_vector(spec, vector)
    return Decimal(decoded["proxy_loss"]) + spec.budget_penalty * decoded["budget_residual_units"]**2


def matrix_energy(matrix_path: Path, vector: list[int]) -> Decimal:
    """Read the exported CSV and evaluate x.T @ Q @ x exactly."""
    selected = {index for index, bit in enumerate(vector) if bit}
    energy = Decimal(0)
    with matrix_path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        for index, row in enumerate(rows):
            if len(row) != len(vector):
                raise ValueError(f"Matrix row {index} has {len(row)} columns")
            if index in selected:
                energy += sum((Decimal(row[column]) for column in selected if column >= index), Decimal(0))
        if index + 1 != len(vector):
            raise ValueError("Matrix row count does not match variable count")
    return energy


def export_matrix(spec: QuboSpec, path: Path) -> dict[str, Any]:
    """Write a headerless, upper-triangular QBoson CSV without rounding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    minimum = Decimal(0)
    maximum = Decimal(0)
    nonzero = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row_index in range(len(spec.variables)):
            row = []
            for column_index in range(len(spec.variables)):
                value = coefficient(spec, row_index, column_index)
                if value:
                    nonzero += 1
                    minimum = min(minimum, value)
                    maximum = max(maximum, value)
                row.append(str(value))
            writer.writerow(row)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "nonzero_upper_triangle": nonzero,
        "minimum_coefficient": str(minimum), "maximum_coefficient": str(maximum),
    }


def export_metadata(spec: QuboSpec, source_path: Path, matrix_path: Path,
                    statistics: dict[str, Any], reference: dict[str, Any] | None = None) -> Path:
    """Write a variable map and verify allocation/energy round trips."""
    checks = {}
    baseline = {tensor.name: "Q3_K" for tensor in spec.problem.tensors}
    allocations = [("all_q3", baseline)]
    if reference is not None:
        allocations.append(("milp_reference", reference["allocation"]))
    for label, allocation in allocations:
        vector = encode_allocation(spec, allocation)
        decoded = inspect_vector(spec, vector)
        value = matrix_energy(matrix_path, vector)
        if value + spec.offset != direct_energy(spec, vector):
            raise AssertionError(f"{label}: matrix energy differs from semantic objective")
        if label == "milp_reference" and abs(Decimal(str(reference["objective"])) -
                                             Decimal(decoded["proxy_loss"])) > Decimal("0.000001"):
            raise AssertionError("MILP reference proxy loss differs from cost table")
        checks[label] = {"target_bytes": decoded["target_bytes"],
                         "proxy_loss": decoded["proxy_loss"], "matrix_energy": str(value)}
        if label == "milp_reference":
            checks[label]["status"] = reference.get("status")
            checks[label]["feasible"] = reference.get("feasible")
    metadata = {
        "author": __author__, "model": spec.problem.metadata.get("model"),
        "source_cost_table": str(source_path.resolve()),
        "source_cost_table_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "budget_index": spec.budget_index, "budget_target_bytes": spec.budget_bytes,
        "budget_full_file_bytes": spec.budget_bytes + spec.fixed_bytes,
        "fixed_non_target_bytes": spec.fixed_bytes,
        "minimum_target_bytes": spec.problem.minimum_bytes,
        "exact_byte_unit": spec.byte_unit, "increment_capacity_units": spec.capacity_units,
        "encoding": "(b1,b0): 00=Q3_K, 01=Q4_K, 10=Q5_K, 11=Q6_K",
        "byte_identity": "For every tensor c(Q3_K)+c(Q6_K)=c(Q4_K)+c(Q5_K)",
        "variable_count": len(spec.variables),
        "decision_variable_count": 2 * len(spec.problem.tensors),
        "slack_variable_count": len(spec.variables) - 2 * len(spec.problem.tensors),
        "budget_penalty": spec.budget_penalty,
        "base_loss": str(spec.base_loss), "energy_offset": str(spec.offset),
        "matrix_format": "headerless square upper-triangular CSV; energy = x.T @ Q @ x",
        "precision": "exact unrounded decimal coefficients; not scaled to hardware int8",
        "hardware_acceptance": "not certified for SPQC-1000 signed-int8 coefficient precision",
        "matrix_file": matrix_path.name, "matrix_statistics": statistics, "checks": checks,
        "variables": [
            {"index": index, "kind": variable.kind, "tensor_index": variable.tensor_index,
             "tensor_name": variable.tensor_name, "bit": variable.bit,
             "budget_weight_units": variable.budget_weight_units,
             "increment_bytes": variable.budget_weight_units * spec.byte_unit if variable.kind == "decision" else 0,
             "loss_linear": str(variable.loss_linear)}
            for index, variable in enumerate(spec.variables)
        ],
        "per_tensor_loss_b0_b1": [str(value) for value in spec.loss_interactions],
    }
    output = matrix_path.with_suffix(".json")
    output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def load_references(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["budget_bytes"]: row["methods"]["milp"] for row in data["results"]}


def generate(input_path: Path, output_dir: Path, references_path: Path | None = None,
             max_variables: int = MAX_VARIABLES) -> list[Path]:
    """Export one checked matrix and JSON variable map per target budget."""
    problem = load_problem(input_path)
    references = load_references(references_path)
    paths: list[Path] = []
    stem = input_path.stem.removesuffix("_costs")
    for index, budget in enumerate(problem.budgets):
        spec = build_spec(problem, index, max_variables=max_variables)
        matrix_path = output_dir / f"{stem}_budget_{index + 1:02d}_two_bit_qubo.csv"
        statistics = export_matrix(spec, matrix_path)
        metadata_path = export_metadata(spec, input_path, matrix_path, statistics, references.get(budget))
        paths.extend((matrix_path, metadata_path))
        print(f"B{index + 1}: {len(spec.variables)} variables -> {matrix_path}")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--max-variables", type=int, default=MAX_VARIABLES)
    args = parser.parse_args(argv)
    generate(args.input, args.output_dir, args.references, args.max_variables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
