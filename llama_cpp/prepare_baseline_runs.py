"""Export distinct solver allocations with measured full-file budget checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

__author__ = "Yuchen Li"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def prepare(table: dict[str, Any], solved: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    metadata = table["metadata"]
    fixed_bytes = metadata["fixed_non_target_bytes"]
    budgets = metadata["uniform_full_file_budget_bytes"]
    costs = {row["name"]: row["choices"] for row in table["tensors"]}
    if len(budgets) != len(solved["results"]):
        raise ValueError("Budget counts differ between the cost table and solver report")
    runs: list[dict[str, Any]] = []
    distinct: dict[str, dict[str, Any]] = {}
    for index, (full_budget, result) in enumerate(zip(budgets, solved["results"])):
        target_budget = result["budget_bytes"]
        if target_budget + fixed_bytes != full_budget:
            raise ValueError(f"Budget {index} does not match measured fixed bytes")
        for method, solution in result["methods"].items():
            if not solution["feasible"] or solution["allocation"] is None:
                raise ValueError(f"Missing feasible solution for {method} at budget {index}")
            allocation = solution["allocation"]
            if set(allocation) != set(costs):
                raise ValueError(f"Tensor inventory differs for {method} at budget {index}")
            actual_target_bytes = sum(costs[name][choice]["bytes"] for name, choice in allocation.items())
            if actual_target_bytes != solution["bytes"]:
                raise ValueError(f"Byte calculation differs for {method} at budget {index}")
            expected_full_bytes = fixed_bytes + actual_target_bytes
            if expected_full_bytes > full_budget:
                raise ValueError(f"Allocation exceeds full-file budget for {method} at budget {index}")
            encoded = json.dumps(allocation, sort_keys=True, separators=(",", ":")).encode("utf-8")
            digest = hashlib.sha256(encoded).hexdigest()[:12]
            plan_name = f"allocation_{digest}"
            plan_path = output_dir / f"{plan_name}.json"
            if digest not in distinct:
                _write_json(plan_path, {"author": __author__, "allocation": allocation})
                distinct[digest] = {
                    "id": plan_name,
                    "allocation_file": str(plan_path),
                    "model_file": f"{plan_name}.gguf",
                    "expected_full_file_bytes": expected_full_bytes,
                    "uses": [],
                }
            distinct[digest]["uses"].append({"budget_index": index, "method": method})
            runs.append({
                "budget_index": index,
                "method": method,
                "allocation_id": plan_name,
                "budget_full_file_bytes": full_budget,
                "expected_full_file_bytes": expected_full_bytes,
                "proxy_objective": solution["objective"],
            })
    return {
        "author": __author__,
        "model": metadata["model"],
        "loss_definition": metadata["loss_definition"],
        "fixed_non_target_bytes": fixed_bytes,
        "runs": runs,
        "distinct_allocations": list(distinct.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--solved", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    table = json.loads(args.table.read_text(encoding="utf-8"))
    solved = json.loads(args.solved.read_text(encoding="utf-8"))
    manifest = prepare(table, solved, args.output_dir)
    _write_json(args.output_dir / "run_manifest.json", manifest)
    print(f"Prepared {len(manifest['runs'])} comparisons, {len(manifest['distinct_allocations'])} distinct GGUF models")


if __name__ == "__main__":
    main()
