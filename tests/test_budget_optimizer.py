"""Synthetic checks for the mixed-type budget allocation baselines."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from classical_baselines import budget_optimizer as optimizer
from classical_baselines.budget_optimizer import Choice, Problem, Tensor

__author__ = "Yuchen Li"


class BudgetOptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem_data = {
            "metadata": {"model": "synthetic"},
            "budgets": [5, 7, 9],
            "tensors": [
                {"name": "a", "choices": {
                    "Q3_K": {"bytes": 1, "loss": 10.0},
                    "Q4_K": {"bytes": 2, "loss": 4.0},
                    "Q5_K": {"bytes": 3, "loss": 1.0},
                    "Q8_0": {"bytes": 4, "loss": 0.0},
                }},
                {"name": "b", "choices": {
                    "Q3_K": {"bytes": 1, "loss": 9.0},
                    "Q4_K": {"bytes": 2, "loss": 3.0},
                    "Q5_K": {"bytes": 3, "loss": 1.2},
                    "Q8_0": {"bytes": 4, "loss": 0.0},
                }},
                {"name": "c", "choices": {
                    "Q3_K": {"bytes": 1, "loss": 8.0},
                    "Q4_K": {"bytes": 2, "loss": 2.5},
                    "Q5_K": {"bytes": 3, "loss": 0.8},
                    "Q8_0": {"bytes": 4, "loss": 0.0},
                }},
            ],
        }
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False)
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        json.dump(self.problem_data, handle)
        handle.close()
        self.problem = optimizer.load_problem(Path(handle.name))

    def test_milp_matches_enumerated_optimum_for_each_budget(self) -> None:
        report = optimizer.solve_all(self.problem)
        for row in report["results"]:
            budget = row["budget_bytes"]
            candidates = []
            for a in self.problem.tensors[0].choices:
                for b in self.problem.tensors[1].choices:
                    for c in self.problem.tensors[2].choices:
                        allocation = (a, b, c)
                        size = sum(option.byte_count for option in allocation)
                        if size <= budget:
                            candidates.append((sum(option.loss for option in allocation), size))
            milp_result = row["methods"]["milp"]
            self.assertIn(milp_result["status"], {"optimal", "dependency_missing"})
            if milp_result["feasible"]:
                self.assertEqual(milp_result["objective"], min(candidates)[0])
                self.assertLessEqual(milp_result["bytes"], budget)

    def test_lagrangian_and_greedy_are_feasible_and_have_valid_allocations(self) -> None:
        for budget in self.problem.budgets:
            for result in (optimizer.solve_lagrangian(self.problem, budget), optimizer.solve_greedy(self.problem, budget)):
                self.assertTrue(result["feasible"])
                self.assertLessEqual(result["bytes"], budget)
                self.assertEqual(set(result["allocation"]), {"a", "b", "c"})
                self.assertTrue(set(result["allocation"].values()) <= set(optimizer.TYPE_ORDER))

    def test_report_metadata_identifies_author_and_methods(self) -> None:
        report = optimizer.solve_all(self.problem)
        self.assertEqual(report["metadata"]["author"], "Yuchen Li")
        self.assertEqual(report["candidate_types"], ["Q3_K", "Q4_K", "Q5_K", "Q8_0"])
        self.assertEqual(len(report["results"]), 3)
        self.assertEqual(
            set(report["results"][0]["methods"]),
            {"milp", "lagrangian_repair_exchange", "greedy_gain_per_byte"},
        )

    def test_q6_candidate_is_loaded_without_changing_legacy_q8_problem(self) -> None:
        q6_data = json.loads(json.dumps(self.problem_data))
        for tensor in q6_data["tensors"]:
            tensor["choices"]["Q6_K"] = tensor["choices"].pop("Q8_0")
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "q6.json"
            input_path.write_text(json.dumps(q6_data), encoding="utf-8")
            problem = optimizer.load_problem(input_path)
        report = optimizer.solve_all(problem)
        self.assertEqual(report["candidate_types"], ["Q3_K", "Q4_K", "Q5_K", "Q6_K"])
        self.assertTrue(all("Q6_K" in {choice.tensor_type for choice in tensor.choices}
                            for tensor in problem.tensors))

    def test_cli_output_persists_author_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            output_path = Path(directory) / "output.json"
            input_path.write_text(json.dumps(self.problem_data), encoding="utf-8")
            self.assertEqual(
                optimizer.main(["--input", str(input_path), "--output", str(output_path)]),
                0,
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["metadata"]["author"], "Yuchen Li")

    def test_infeasible_budget_is_reported_without_claiming_an_allocation(self) -> None:
        result = optimizer.solve_greedy(self.problem, 2)
        self.assertFalse(result["feasible"])
        self.assertEqual(result["status"], "infeasible")
        self.assertIsNone(result["allocation"])
        self.assertEqual(result["minimum_possible_bytes"], 3)

    def test_invalid_input_rejects_non_integral_byte_cost(self) -> None:
        bad = json.loads(json.dumps(self.problem_data))
        bad["tensors"][0]["choices"]["Q3_K"]["bytes"] = 1.5
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as handle:
            json.dump(bad, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "positive integer"):
                optimizer.load_problem(Path(handle.name))

    def test_two_item_exchange_pairs_a_downgrade_with_a_higher_gain_upgrade(self) -> None:
        problem = Problem(
            tensors=(
                Tensor("a", (Choice("Q3_K", 1, 1.4), Choice("Q4_K", 5, 1.2))),
                Tensor("b", (Choice("Q3_K", 1, 0.5), Choice("Q4_K", 5, 0.0))),
            ),
            budgets=(6,),
            metadata={},
        )
        move = optimizer._best_local_move(problem, [1, 0], budget=6)
        self.assertIsNotNone(move)
        assert move is not None
        changes, objective_delta = move
        self.assertEqual({item_index for item_index, _ in changes}, {0, 1})
        self.assertAlmostEqual(objective_delta, -0.3)


if __name__ == "__main__":
    unittest.main()
