"""Solve the five measured Q3_K/Q4_K/Q5_K/Q6_K allocation budgets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__author__ = "Yuchen Li"
TYPE_ORDER = ("Q3_K", "Q4_K", "Q5_K", "Q6_K")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--mip-rel-gap", type=float, default=0.0)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    from classical_baselines import budget_optimizer

    problem = budget_optimizer.load_problem(args.input)
    if len(problem.tensors) != 196 or len(problem.budgets) != 5:
        raise ValueError("Expected 196 tensors and the five previous budgets")
    if any({choice.tensor_type for choice in tensor.choices} != set(TYPE_ORDER) for tensor in problem.tensors):
        raise ValueError("All tensors must support all four Q3/Q4/Q5/Q6 choices")
    report = budget_optimizer.solve_all(problem, time_limit=args.time_limit, mip_rel_gap=args.mip_rel_gap)
    if any(not row["methods"]["milp"]["feasible"] for row in report["results"]):
        raise RuntimeError("A MILP solution was unavailable for at least one budget")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    allocation_dir = args.output.parent / "allocations"
    allocation_dir.mkdir(exist_ok=True)
    for index, row in enumerate(report["results"], start=1):
        result = row["methods"]["milp"]
        allocation = {
            "author": __author__,
            "model": problem.metadata["model"],
            "method": "milp",
            "budget_target_bytes": row["budget_bytes"],
            "budget_full_file_bytes": row["budget_bytes"] + problem.metadata["fixed_non_target_bytes"],
            "expected_target_bytes": result["bytes"],
            "objective": result["objective"],
            "allocation": result["allocation"],
        }
        (allocation_dir / f"budget_{index:02d}_milp.json").write_text(
            json.dumps(allocation, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        print(row["budget_bytes"], result["status"], result["bytes"], result["objective"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
