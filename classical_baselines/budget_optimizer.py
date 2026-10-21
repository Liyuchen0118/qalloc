"""Budgeted mixed-type tensor allocation baselines.

Input JSON format::

    {
      "metadata": {"model": "Qwen2.5-7B"},
      "budgets": [1000, 1250],
      "tensors": [
        {"name": "layers.0.self_attn.q_proj", "choices": {
          "Q3_K": {"bytes": 300, "loss": 0.12},
          "Q4_K": {"bytes": 400, "loss": 0.04},
          "Q5_K": {"bytes": 500, "loss": 0.01},
          "Q8_0": {"bytes": 800, "loss": 0.0}
        }}
      ]
    }

Loss is minimized and bytes are a hard upper bound. The loss values must be
the same proxy for every method; they are not predictions of end-to-end PPL.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__author__ = "Yuchen Li"

TYPE_ORDER = ("Q3_K", "Q4_K", "Q5_K", "Q8_0")
EPSILON = 1e-12


@dataclass(frozen=True)
class Choice:
    tensor_type: str
    byte_count: int
    loss: float


@dataclass(frozen=True)
class Tensor:
    name: str
    choices: tuple[Choice, ...]


@dataclass(frozen=True)
class Problem:
    tensors: tuple[Tensor, ...]
    budgets: tuple[int, ...]
    metadata: dict[str, Any]

    @property
    def minimum_bytes(self) -> int:
        return sum(min(choice.byte_count for choice in tensor.choices) for tensor in self.tensors)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_problem(path: Path) -> Problem:
    """Read and validate a per-tensor choice table."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object")
    raw_tensors = data.get("tensors")
    raw_budgets = data.get("budgets")
    if not isinstance(raw_tensors, list) or not raw_tensors:
        raise ValueError("Input must contain a non-empty 'tensors' array")
    if not isinstance(raw_budgets, list) or not raw_budgets:
        raise ValueError("Input must contain a non-empty 'budgets' array")

    tensors: list[Tensor] = []
    names: set[str] = set()
    for index, raw_tensor in enumerate(raw_tensors):
        if not isinstance(raw_tensor, dict):
            raise ValueError(f"tensors[{index}] must be an object")
        name = raw_tensor.get("name")
        raw_choices = raw_tensor.get("choices")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"tensors[{index}].name must be a non-empty string")
        if name in names:
            raise ValueError(f"Duplicate tensor name: {name}")
        names.add(name)
        if not isinstance(raw_choices, dict) or not raw_choices:
            raise ValueError(f"Tensor {name!r} must have a non-empty 'choices' object")
        unknown = set(raw_choices) - set(TYPE_ORDER)
        if unknown:
            raise ValueError(f"Tensor {name!r} has unsupported types: {sorted(unknown)}")
        choices: list[Choice] = []
        for tensor_type in TYPE_ORDER:
            if tensor_type not in raw_choices:
                continue
            value = raw_choices[tensor_type]
            if not isinstance(value, dict):
                raise ValueError(f"{name}/{tensor_type} must provide bytes and loss")
            byte_count = value.get("bytes")
            loss = value.get("loss")
            if not _is_int(byte_count) or byte_count <= 0:
                raise ValueError(f"{name}/{tensor_type}.bytes must be a positive integer")
            if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss):
                raise ValueError(f"{name}/{tensor_type}.loss must be a finite number")
            choices.append(Choice(tensor_type, byte_count, float(loss)))
        choices.sort(key=lambda choice: (choice.byte_count, choice.loss, TYPE_ORDER.index(choice.tensor_type)))
        tensors.append(Tensor(name, tuple(choices)))

    budgets: list[int] = []
    for index, budget in enumerate(raw_budgets):
        if not _is_int(budget) or budget < 0:
            raise ValueError(f"budgets[{index}] must be a non-negative integer byte count")
        budgets.append(budget)
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object when provided")
    return Problem(tuple(tensors), tuple(budgets), metadata)


def _allocation(problem: Problem, selected: tuple[Choice, ...] | list[Choice]) -> dict[str, Any]:
    byte_count = sum(choice.byte_count for choice in selected)
    loss = sum(choice.loss for choice in selected)
    return {
        "allocation": {tensor.name: choice.tensor_type for tensor, choice in zip(problem.tensors, selected)},
        "bytes": byte_count,
        "objective": loss,
    }


def _result(
    problem: Problem,
    budget: int,
    method: str,
    selected: tuple[Choice, ...] | list[Choice] | None,
    status: str,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    if selected is None:
        return {
            "method": method,
            "status": status,
            "feasible": False,
            "budget_bytes": budget,
            "minimum_possible_bytes": problem.minimum_bytes,
            "objective": None,
            "bytes": None,
            "allocation": None,
            "diagnostics": diagnostics,
        }
    values = _allocation(problem, selected)
    feasible = values["bytes"] <= budget
    return {
        "method": method,
        "status": status if feasible else "infeasible",
        "feasible": feasible,
        "budget_bytes": budget,
        **values,
        "diagnostics": diagnostics,
    }


def solve_milp(problem: Problem, budget: int, time_limit: float | None, mip_rel_gap: float | None) -> dict[str, Any]:
    """Solve the exact multiple-choice knapsack MILP using SciPy/HiGHS."""
    started = time.perf_counter()
    method = "milp"
    if problem.minimum_bytes > budget:
        return _result(problem, budget, method, None, "infeasible", {
            "minimum_possible_bytes": problem.minimum_bytes,
            "solve_time_seconds": time.perf_counter() - started,
        })
    flat_choices = [choice for tensor in problem.tensors for choice in tensor.choices]
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError:
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:
            return _result(problem, budget, method, None, "dependency_missing", {
                "message": f"Install scipy>=1.9 or gurobipy to solve the MILP: {exc}",
                "solve_time_seconds": time.perf_counter() - started,
            })

        try:
            model = gp.Model("qalloc_budget_allocation")
            model.Params.OutputFlag = 0
            if time_limit is not None:
                model.Params.TimeLimit = time_limit
            if mip_rel_gap is not None:
                model.Params.MIPGap = mip_rel_gap
            variables = [
                [model.addVar(vtype=GRB.BINARY, obj=choice.loss, name=f"x_{i}_{j}")
                 for j, choice in enumerate(tensor.choices)]
                for i, tensor in enumerate(problem.tensors)
            ]
            model.update()
            for item_vars in variables:
                model.addConstr(gp.quicksum(item_vars) == 1)
            model.addConstr(gp.quicksum(
                choice.byte_count * variable
                for tensor, item_vars in zip(problem.tensors, variables)
                for choice, variable in zip(tensor.choices, item_vars)
            ) <= budget)
            model.optimize()
        except Exception as exc:
            return _result(problem, budget, method, None, "solver_error", {
                "message": str(exc),
                "solver": "gurobipy",
                "solve_time_seconds": time.perf_counter() - started,
            })

        status = {
            GRB.OPTIMAL: "optimal",
            GRB.TIME_LIMIT: "limit_reached",
            GRB.INFEASIBLE: "infeasible",
            GRB.INTERRUPTED: "limit_reached",
        }.get(model.Status, "solver_error")
        diagnostics = {
            "solver": "Gurobi",
            "solver_status_code": int(model.Status),
            "solver_message": status,
            "solve_time_seconds": time.perf_counter() - started,
            "time_limit_seconds": time_limit,
            "requested_relative_gap": mip_rel_gap,
            "gurobi_version": ".".join(map(str, gp.gurobi.version())),
        }
        if model.SolCount == 0:
            return _result(problem, budget, method, None, status, diagnostics)
        selected = [
            tensor.choices[max(range(len(item_vars)), key=lambda j: item_vars[j].X)]
            for tensor, item_vars in zip(problem.tensors, variables)
        ]
        diagnostics["best_bound"] = float(model.ObjBound)
        diagnostics["relative_gap"] = float(model.MIPGap)
        diagnostics["node_count"] = int(model.NodeCount)
        diagnostics["incumbent_objective"] = float(model.ObjVal)
        solver_status = status
    else:
        offsets = [0]
        for tensor in problem.tensors:
            offsets.append(offsets[-1] + len(tensor.choices))
        row_count = len(problem.tensors) + 1
        matrix = np.zeros((row_count, len(flat_choices)), dtype=float)
        lower = np.full(row_count, -np.inf, dtype=float)
        upper = np.full(row_count, np.inf, dtype=float)
        for index in range(len(problem.tensors)):
            matrix[index, offsets[index]:offsets[index + 1]] = 1.0
            lower[index] = upper[index] = 1.0
        matrix[-1, :] = [choice.byte_count for choice in flat_choices]
        upper[-1] = float(budget)
        options: dict[str, Any] = {}
        if time_limit is not None:
            options["time_limit"] = time_limit
        if mip_rel_gap is not None:
            options["mip_rel_gap"] = mip_rel_gap

        try:
            result = milp(
                c=np.asarray([choice.loss for choice in flat_choices], dtype=float),
                integrality=np.ones(len(flat_choices), dtype=int),
                bounds=Bounds(np.zeros(len(flat_choices)), np.ones(len(flat_choices))),
                constraints=LinearConstraint(matrix, lower, upper),
                options=options,
            )
        except Exception as exc:
            return _result(problem, budget, method, None, "solver_error", {
                "message": str(exc),
                "solver": "scipy.optimize.milp (HiGHS)",
                "solve_time_seconds": time.perf_counter() - started,
            })

        solver_status = {0: "optimal", 1: "limit_reached", 2: "infeasible", 3: "unbounded", 4: "solver_error"}.get(
            int(result.status), "solver_error"
        )
        diagnostics = {
            "solver": "scipy.optimize.milp (HiGHS)",
            "solver_status_code": int(result.status),
            "solver_message": str(result.message),
            "solve_time_seconds": time.perf_counter() - started,
            "time_limit_seconds": time_limit,
            "requested_relative_gap": mip_rel_gap,
        }
        for source, target in (("mip_dual_bound", "best_bound"), ("mip_gap", "relative_gap"), ("mip_node_count", "node_count")):
            value = getattr(result, source, None)
            if value is not None and isinstance(value, (int, float)) and math.isfinite(float(value)):
                diagnostics[target] = float(value) if target != "node_count" else int(value)

        if result.x is None:
            return _result(problem, budget, method, None, solver_status, diagnostics)
        selected = []
        for index, tensor in enumerate(problem.tensors):
            segment = result.x[offsets[index]:offsets[index + 1]]
            selected.append(tensor.choices[int(np.argmax(segment))])
    values = _allocation(problem, selected)
    if values["bytes"] > budget:
        diagnostics["incumbent_rejected"] = "returned incumbent violates byte budget after decoding"
        return _result(problem, budget, method, None, "solver_error", diagnostics)
    diagnostics["incumbent_objective"] = values["objective"]
    return _result(problem, budget, method, selected, solver_status, diagnostics)


def _selected_from_indices(problem: Problem, indices: list[int]) -> list[Choice]:
    return [tensor.choices[index] for tensor, index in zip(problem.tensors, indices)]


def _repair_to_budget(problem: Problem, indices: list[int], budget: int) -> tuple[list[int], int]:
    """Downgrade to save bytes at the smallest proxy-loss increase per byte."""
    selected = indices.copy()
    total_bytes = sum(tensor.choices[index].byte_count for tensor, index in zip(problem.tensors, selected))
    downgrades = 0
    while total_bytes > budget:
        best: tuple[float, float, int, int, int] | None = None
        for item_index, (tensor, current_index) in enumerate(zip(problem.tensors, selected)):
            current = tensor.choices[current_index]
            for next_index, candidate in enumerate(tensor.choices):
                saved = current.byte_count - candidate.byte_count
                if saved <= 0:
                    continue
                added_loss = candidate.loss - current.loss
                key = (added_loss / saved, added_loss, -saved, item_index, next_index)
                if best is None or key < best:
                    best = key
        if best is None:
            return selected, downgrades
        _, _, neg_saved, item_index, next_index = best
        current = problem.tensors[item_index].choices[selected[item_index]]
        candidate = problem.tensors[item_index].choices[next_index]
        total_bytes += candidate.byte_count - current.byte_count
        selected[item_index] = next_index
        downgrades += 1
    return selected, downgrades


def _best_local_move(problem: Problem, selected: list[int], budget: int) -> tuple[list[tuple[int, int]], float] | None:
    """Find the best feasible one-tensor change or two-tensor exchange."""
    current_choices = _selected_from_indices(problem, selected)
    current_bytes = sum(choice.byte_count for choice in current_choices)
    slack = budget - current_bytes
    changes: list[tuple[int, int, int, float]] = []
    best_single: tuple[float, int, int] | None = None
    for item_index, (tensor, current_index) in enumerate(zip(problem.tensors, selected)):
        current = tensor.choices[current_index]
        for next_index, candidate in enumerate(tensor.choices):
            if next_index == current_index:
                continue
            delta_bytes = candidate.byte_count - current.byte_count
            delta_loss = candidate.loss - current.loss
            changes.append((item_index, next_index, delta_bytes, delta_loss))
            if delta_loss < -EPSILON and delta_bytes <= slack:
                move = (delta_loss, item_index, next_index)
                if best_single is None or move < best_single:
                    best_single = move

    best_pair: tuple[float, int, int, int, int] | None = None
    if len(changes) >= 2:
        sorted_changes = sorted(changes, key=lambda change: (change[2], change[3], change[0], change[1]))
        prefix_best: list[tuple[tuple[int, int, int, float] | None, tuple[int, int, int, float] | None]] = []
        first: tuple[int, int, int, float] | None = None
        second: tuple[int, int, int, float] | None = None
        for change in sorted_changes:
            if first is None or change[3] < first[3]:
                if first is not None and change[0] != first[0]:
                    second = first
                first = change
            elif change[0] != first[0] and (second is None or change[3] < second[3]):
                second = change
            prefix_best.append((first, second))

        costs = [change[2] for change in sorted_changes]
        for change in changes:
            limit_index = bisect.bisect_right(costs, slack - change[2]) - 1
            if limit_index < 0:
                continue
            candidate_a, candidate_b = prefix_best[limit_index]
            partner = candidate_a if candidate_a[0] != change[0] else candidate_b
            if partner is None or partner[0] == change[0]:
                continue
            delta_loss = change[3] + partner[3]
            if delta_loss >= -EPSILON:
                continue
            move = (delta_loss, change[0], change[1], partner[0], partner[1])
            if best_pair is None or move < best_pair:
                best_pair = move

    if best_single is None and best_pair is None:
        return None
    if best_single is not None and (best_pair is None or best_single[0] <= best_pair[0]):
        return [(best_single[1], best_single[2])], best_single[0]
    assert best_pair is not None
    return [(best_pair[1], best_pair[2]), (best_pair[3], best_pair[4])], best_pair[0]


def _local_improve(problem: Problem, indices: list[int], budget: int) -> tuple[list[int], int, int, bool]:
    selected = indices.copy()
    one_item_moves = 0
    two_item_moves = 0
    move_limit = max(20, 10 * len(problem.tensors))
    hit_move_limit = True
    for _ in range(move_limit):
        move = _best_local_move(problem, selected, budget)
        if move is None:
            hit_move_limit = False
            break
        changes, _ = move
        for item_index, next_index in changes:
            selected[item_index] = next_index
        if len(changes) == 1:
            one_item_moves += 1
        else:
            two_item_moves += 1
    if hit_move_limit:
        hit_move_limit = _best_local_move(problem, selected, budget) is not None
    return selected, one_item_moves, two_item_moves, hit_move_limit


def solve_lagrangian(problem: Problem, budget: int) -> dict[str, Any]:
    """Lagrangian relaxation, byte repair, and one/two-item local exchanges."""
    started = time.perf_counter()
    method = "lagrangian_repair_exchange"
    if problem.minimum_bytes > budget:
        return _result(problem, budget, method, None, "infeasible", {
            "minimum_possible_bytes": problem.minimum_bytes,
            "solve_time_seconds": time.perf_counter() - started,
        })

    def relaxed(lagrange_multiplier: float) -> tuple[list[int], float]:
        selected: list[int] = []
        relaxed_value = -lagrange_multiplier * budget
        for tensor in problem.tensors:
            best_index = min(
                range(len(tensor.choices)),
                key=lambda index: (
                    tensor.choices[index].loss + lagrange_multiplier * tensor.choices[index].byte_count,
                    tensor.choices[index].byte_count,
                    tensor.choices[index].loss,
                ),
            )
            choice = tensor.choices[best_index]
            selected.append(best_index)
            relaxed_value += choice.loss + lagrange_multiplier * choice.byte_count
        return selected, relaxed_value

    dual_bound = -math.inf
    evaluated = 0
    best_under: list[int] | None = None
    best_under_loss = math.inf
    last_over: list[int] | None = None

    def consider(multiplier: float) -> list[int]:
        nonlocal dual_bound, evaluated, best_under, best_under_loss, last_over
        selection, bound = relaxed(multiplier)
        evaluated += 1
        dual_bound = max(dual_bound, bound)
        byte_count = sum(tensor.choices[index].byte_count for tensor, index in zip(problem.tensors, selection))
        loss = sum(tensor.choices[index].loss for tensor, index in zip(problem.tensors, selection))
        if byte_count <= budget:
            if loss < best_under_loss:
                best_under, best_under_loss = selection.copy(), loss
        else:
            last_over = selection.copy()
        return selection

    consider(0.0)
    high = 1.0
    high_selection = consider(high)
    while sum(tensor.choices[index].byte_count for tensor, index in zip(problem.tensors, high_selection)) > budget and high < 1e18:
        high *= 2.0
        high_selection = consider(high)
    low = 0.0
    if sum(tensor.choices[index].byte_count for tensor, index in zip(problem.tensors, high_selection)) <= budget:
        for _ in range(80):
            middle = (low + high) / 2.0
            if middle == low or middle == high:
                break
            selection = consider(middle)
            byte_count = sum(tensor.choices[index].byte_count for tensor, index in zip(problem.tensors, selection))
            if byte_count > budget:
                low = middle
                last_over = selection.copy()
            else:
                high = middle

    starts: list[tuple[list[int], int]] = []
    if best_under is not None:
        starts.append((best_under, 0))
    if last_over is not None:
        repaired, downgrades = _repair_to_budget(problem, last_over, budget)
        if sum(tensor.choices[index].byte_count for tensor, index in zip(problem.tensors, repaired)) <= budget:
            starts.append((repaired, downgrades))
    min_cost_indices = [
        min(range(len(tensor.choices)), key=lambda index: (tensor.choices[index].byte_count, tensor.choices[index].loss))
        for tensor in problem.tensors
    ]
    starts.append((min_cost_indices, 0))

    best: list[int] | None = None
    best_loss = math.inf
    best_counts = (0, 0, 0, False)
    for initial, repaired_count in starts:
        improved, one_moves, two_moves, hit_move_limit = _local_improve(problem, initial, budget)
        loss = sum(tensor.choices[index].loss for tensor, index in zip(problem.tensors, improved))
        if loss < best_loss:
            best, best_loss = improved, loss
            best_counts = (repaired_count, one_moves, two_moves, hit_move_limit)
    assert best is not None
    diagnostics = {
        "lagrange_dual_bound": dual_bound if math.isfinite(dual_bound) else None,
        "lagrange_multiplier": high,
        "relaxations_evaluated": evaluated,
        "downgrade_repair_moves": best_counts[0],
        "one_item_improvement_moves": best_counts[1],
        "two_item_exchange_moves": best_counts[2],
        "local_search_move_limit": max(20, 10 * len(problem.tensors)),
        "local_search_hit_move_limit": best_counts[3],
        "solve_time_seconds": time.perf_counter() - started,
    }
    return _result(problem, budget, method, _selected_from_indices(problem, best), "feasible", diagnostics)


def solve_greedy(problem: Problem, budget: int) -> dict[str, Any]:
    """Feasible marginal proxy-gain per additional byte greedy baseline."""
    started = time.perf_counter()
    method = "greedy_gain_per_byte"
    if problem.minimum_bytes > budget:
        return _result(problem, budget, method, None, "infeasible", {
            "minimum_possible_bytes": problem.minimum_bytes,
            "solve_time_seconds": time.perf_counter() - started,
        })
    selected = [
        min(range(len(tensor.choices)), key=lambda index: (tensor.choices[index].byte_count, tensor.choices[index].loss))
        for tensor in problem.tensors
    ]
    current_bytes = sum(tensor.choices[index].byte_count for tensor, index in zip(problem.tensors, selected))
    moves = 0
    while True:
        best: tuple[float, float, int, int] | None = None
        slack = budget - current_bytes
        for item_index, (tensor, current_index) in enumerate(zip(problem.tensors, selected)):
            current = tensor.choices[current_index]
            for next_index, candidate in enumerate(tensor.choices):
                extra_bytes = candidate.byte_count - current.byte_count
                gain = current.loss - candidate.loss
                if extra_bytes <= 0 or extra_bytes > slack or gain <= EPSILON:
                    continue
                key = (-gain / extra_bytes, -gain, item_index, next_index)
                if best is None or key < best:
                    best = key
        if best is None:
            break
        _, _, item_index, next_index = best
        current_bytes += problem.tensors[item_index].choices[next_index].byte_count - problem.tensors[item_index].choices[selected[item_index]].byte_count
        selected[item_index] = next_index
        moves += 1
    return _result(problem, budget, method, _selected_from_indices(problem, selected), "feasible", {
        "upgrade_moves": moves,
        "solve_time_seconds": time.perf_counter() - started,
    })


def solve_all(problem: Problem, time_limit: float | None = None, mip_rel_gap: float | None = None) -> dict[str, Any]:
    """Solve every requested budget and return a JSON-ready report."""
    results = []
    for budget in problem.budgets:
        results.append({
            "budget_bytes": budget,
            "methods": {
                "milp": solve_milp(problem, budget, time_limit, mip_rel_gap),
                "lagrangian_repair_exchange": solve_lagrangian(problem, budget),
                "greedy_gain_per_byte": solve_greedy(problem, budget),
            },
        })
    return {
        "metadata": {
            "author": __author__,
            "schema_version": 1,
            "source_metadata": problem.metadata,
            "objective": "minimize additive proxy loss",
            "budget_semantics": "sum of per-tensor bytes <= budget_bytes",
        },
        "candidate_types": list(TYPE_ORDER),
        "results": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve tensor-type allocations for multiple byte budgets")
    parser.add_argument("--input", type=Path, required=True, help="JSON tensor cost/loss table")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--time-limit", type=float, default=None, help="MILP time limit in seconds")
    parser.add_argument("--mip-rel-gap", type=float, default=None, help="MILP relative optimality-gap target")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.time_limit is not None and args.time_limit <= 0:
        raise SystemExit("--time-limit must be positive")
    if args.mip_rel_gap is not None and args.mip_rel_gap < 0:
        raise SystemExit("--mip-rel-gap must be non-negative")
    problem = load_problem(args.input)
    report = solve_all(problem, args.time_limit, args.mip_rel_gap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(report['results'])} budget results to {args.output}")
    if any(method["status"] == "dependency_missing" for row in report["results"] for method in row["methods"].values()):
        print("MILP was skipped because SciPy is unavailable; install it with: python -m pip install scipy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
